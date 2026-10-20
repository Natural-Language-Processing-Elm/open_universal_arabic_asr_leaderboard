import json
import time
import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

device = "cuda:0"


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000, quality="HQ")
    return wav.astype(np.float32)


def run_jais_asr(model_id, data_manifest, data_folder, output_manifest, batch_size=8):
    """
    Arguments
    ---------
    model_id: str
        IFM/Jais-ASR
    data_manifest: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_manifest: str
        The output manifest path
    batch_size: int
        Number of clips decoded together

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map=device
    ).eval()
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio"}]}], tokenize=False, add_generation_prompt=True
    )

    with open(data_manifest, "r") as f:
        items = [json.loads(line) for line in f]
    order = sorted(range(len(items)), key=lambda i: items[i]["duration"])
    preds = [None] * len(items)

    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
    all_inference_time = 0
    all_audio_duration = 0
    for count, start in enumerate(tqdm(range(0, len(order), batch_size))):
        idx = order[start:start + batch_size]
        wavs = [load_audio(items[i]["audio_filepath"].format(data_folder=data_folder)) for i in idx]

        start_time = time.time()
        inputs = processor(
            text=[prompt] * len(wavs), audio=wavs, sampling_rate=16000, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs)
        results = processor.post_process_transcription(outputs, input_lengths=inputs["input_ids"].shape[1])
        end_time = time.time()

        if count > 4:
            all_inference_time += end_time - start_time
            all_audio_duration += sum(items[i]["duration"] for i in idx)
        for i, result in zip(idx, results):
            preds[i] = (result.text or "").strip()
    peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

    with open(output_manifest, 'w') as fout:
        for item, pred in zip(items, preds):
            metadata = {
                "audio_filepath": item["audio_filepath"].format(data_folder=data_folder),
                "text": item["text"],
                "pred_text": pred,
            }
            json.dump(metadata, fout, ensure_ascii=False)
            fout.write('\n')

    print("average rtf : ", all_inference_time/all_audio_duration if all_audio_duration else "n/a")
    print("model memory : ", initial_memory)
    print("peak inference memory : ", peak_memory-initial_memory)
