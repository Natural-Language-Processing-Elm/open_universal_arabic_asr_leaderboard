import json
import time
import torch
from tqdm import tqdm
from transformers import VoxtralForConditionalGeneration, AutoProcessor

device = "cuda:0"


def run_voxtral(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        mistralai/Voxtral-Small-24B-2507 or mistralai/Voxtral-Mini-3B-2507
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
    processor = AutoProcessor.from_pretrained(model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    with open(data_manifest, "r") as f:
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
                inputs = processor.apply_transcription_request(
                    language="ar",
                    audio=in_path,
                    model_id=repo_id,
                )
                inputs = inputs.to(device, dtype=torch.bfloat16)

                outputs = model.generate(**inputs, max_new_tokens=500)
                decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
                    "pred_text": decoded_outputs[0],
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))