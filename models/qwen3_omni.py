import json
import time
import random
import torch
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

ASR_EXAMPLE_PROMPTS = [
    "هل تقدر تعطيني نص هالصوت؟",
    "ممكن ترسل لي تفريغ هالتسجيل؟",
    "ماذا يقول المتحدث؟",
    "تعطيني نسخة مكتوبة من المقطع؟",
    "تكدر تحوّل هالصوت لنص؟",
    "سَوِ تحويل للصوت إلى نص مكتوب.",
    "أعطني تفريغ كلام المتكلّم بالصوت.",
    "هات لي نص التسجيل.",
    "أبغى نص الكلام الموجود بالصوت.",
    "حَوِّل هذا المقطع الصوتي إلى نص.",
]
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
)
model.to("cuda:0")
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)


def process_and_infer(audio, prompt):
    """
    Arguments
    ---------
    audio: str
        audio file path
    prompt: str
        An ASR instruction to be followed

    Output
    ---------
    Transcription for the audio file
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt}
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs,
        max_new_tokens=512,
        speaker="Ethan", 
        thinker_return_dict_in_generate=True,
        use_audio_in_video=True)

    text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return text


def run_qwen3_omni(data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    data_manifest: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_manifest: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    with open(data_manifest, 'r') as f:
        with open(output_manifest, 'w') as fout:
            all_inference_time = 0
            all_audio_duration = 0
            all_inference_memory = []
            count = 0
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                pred = process_and_infer(in_path, random.choice(ASR_EXAMPLE_PROMPTS)+" أَخْرِج التَّفْرِيغ الصَّوتي فَقَط.")
                end_time = time.time()
                
                inference_time = end_time - start_time
                if count > 4:
                    all_inference_time += inference_time
                    all_audio_duration += duration
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                all_inference_memory.append(peak_memory-initial_memory)        
                count += 1

                metadata = {
                    "audio_filepath": in_path,
                    "text": item["text"],
                    "pred_text": pred[0],
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))