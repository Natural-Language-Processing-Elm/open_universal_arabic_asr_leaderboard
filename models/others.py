import json
import time
import torch
import torchaudio
from tqdm import tqdm
from huggingsound import SpeechRecognitionModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2BertForCTC, pipeline, AutoProcessor, SeamlessM4Tv2ForSpeechToText

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def run_xlsr(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        Huggingface model name: "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
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
    model = SpeechRecognitionModel(model_id, "cuda")

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(data_manifest, "r") as f:
        with open(output_manifest, 'w') as fout:
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                transcription = model.transcribe([in_path])[0]["transcription"]
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
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))


def run_w2v_bert(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        Huggingface model name: "whitefox123/w2v-bert-2.0-arabic-4"
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
    model = Wav2Vec2BertForCTC.from_pretrained(model_id)
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(data_manifest, "r") as f:
        with open(output_manifest, 'w') as fout:
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                transcription = pipe(in_path)["text"]
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
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))


def run_seamless(model_id, language_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        Huggingface model name: "facebook/seamless-m4t-v2-large"
    language_id: str
        Seamless supports 3 arabic dialects ["arb", "ary", "arz"]
        We run 3 inferences separately and report the best WER
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
    model =  SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id)
    model.to(device)

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(data_manifest, "r") as f:
        with open(output_manifest, 'w') as fout:
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                audio, orig_freq = torchaudio.load(in_path)
                audio_inputs = processor(audios=audio, return_tensors="pt").to(device)
                output_tokens = model.generate(
                    **audio_inputs,
                    tgt_lang=language_id, # "arb", "ary", "arz"
                ) 
                translated_text_from_audio = processor.decode(output_tokens[0], skip_special_tokens=True)
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
                    "pred_text": translated_text_from_audio,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))


def run_mms(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        Huggingface model name: "facebook/mms-1b-all"
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
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to(device)
    processor.tokenizer.set_target_lang("ara")
    model.load_adapter("ara")

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(data_manifest, "r") as f:
        with open(output_manifest, 'w') as fout:
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                audio, orig_freq = torchaudio.load(in_path)
                audio = audio.squeeze()
                inputs = processor(audio, sampling_rate=16_000, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs).logits
                ids = torch.argmax(outputs, dim=-1)[0]
                transcription = processor.decode(ids)
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
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))
