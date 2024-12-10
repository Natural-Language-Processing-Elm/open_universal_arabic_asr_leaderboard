import json
import time
import torch
from tqdm import tqdm
from speechbrain.pretrained import EncoderASR


def run_speechbrain(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        "asafaya/hubert-large-arabic-transcribe" for hubert large model,
        "speechbrain/asr-wav2vec2-commonvoice-14-ar" for wav2vec model
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
    asr_model = EncoderASR.from_hparams(
        source=model_id,
        savedir="pretrained_models/",
        run_opts={"device":"cuda"},
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
                transcription = asr_model.transcribe_file(in_path)
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

