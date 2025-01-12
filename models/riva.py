"""
This script contains the Riva client to transcribe an entire dataset.
The inference is based on Riva quick start 2.18.0, please first launch
the Riva quick start container according to the guide:
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart
The models should be specified in config.sh
"""

import os
import json
import torch
from tqdm import tqdm
import time


def run_riva_models(manifest_path, data_folder, log_path, output_path):
    """
    Arguments
    ---------
    manifest_path: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    log_path: str
        The path to the store riva responses
    output_path: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """

    def get_transcript(file):
        """
        Arguments
        ---------
        file: str
            Path of a log file where riva reponses are preserved
        Output
        ---------
            The riva model transcription
        """
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_line = lines[-1]
            return last_line.replace("Final transcript: ", "").strip()

    count = 0
    all_inference_time = 0
    all_audio_duration = 0
    all_inference_memory = []
    with open(manifest_path, "r") as f:
        with open(output_path, 'w') as fout:
            for line in tqdm(f):
                item = json.loads(line)
                in_path = item["audio_filepath"].format(data_folder=data_folder)
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                os.system(f"python riva_quickstart/examples/transcribe_file_offline.py --input-file {in_path} > {log_path}/{count}.txt")
                end_time = time.time()

                inference_time = end_time - start_time
                if count > 4:
                    all_inference_time += inference_time
                    all_audio_duration += duration
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                all_inference_memory.append(peak_memory-initial_memory)        
                
                metadata = {
                    "audio_filepath": in_path,
                    "text": item["text"],
                    "pred_text": get_transcript(f"{log_path}/{count}.txt"),
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

                count += 1

    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))



