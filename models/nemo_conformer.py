import nemo.collections.asr as nemo_asr
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
import json
import torch
from tqdm import tqdm
import time


def run_conformer_greedy(asr_model_path, manifest_path, data_folder, output_path):
    """
    Arguments
    ---------
    asr_model_path: str
        local path to .nemo model, which can be downloaded from
        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ar_ar_conformer/files
    manifest_path: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_path: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    asr_model = nemo_asr.models.ASRModel.restore_from(asr_model_path, map_location=torch.device("cuda"))

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(manifest_path, 'r') as manifest_file:
        with open(output_path, 'w') as fout:
            for line in tqdm(manifest_file):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                with torch.no_grad():
                    pred = asr_model.transcribe([in_path])[0]
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
                    "pred_text": pred,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))


def run_conformer_lm(asr_model_path, lm_path, manifest_path, data_folder, output_path):
    """
    Arguments
    ---------
    asr_model_path: str
        local path to .nemo model, which can be downloaded from
        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ar_ar_conformer/files
    lm_path: str
        local path to .arpa language model, which can be downloaded from
        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/speechtotext_ar_ar_lm/files
    manifest_path: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_path: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    asr_model = nemo_asr.models.ASRModel.restore_from(asr_model_path, map_location=torch.device("cuda"))
    decoder = build_ctcdecoder(
        asr_model.decoder.vocabulary,
        lm_path,
        alpha=0.5,  # tuned on a val set
        beta=1.0,  # tuned on a val set
    )

    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    count = 0
    with open(manifest_path, 'r') as manifest_file:
        with open(output_path, 'w') as fout:

            for line in tqdm(manifest_file):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                with torch.no_grad():
                    logits_hyps = asr_model.transcribe([in_path], return_hypotheses=True)[0]
                logits = logits_hyps.alignments.numpy()
                pred = decoder.decode(logits)
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
                    "pred_text": pred,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))
