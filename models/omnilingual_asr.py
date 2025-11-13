import json
import time
import torch
from tqdm import tqdm
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def run_omnilingual(model_id, data_manifest, data_folder, output_manifest):
    """
    Arguments
    ---------
    model_id: str
        ["omniASR_CTC_300M", "omniASR_LLM_300M", "omniASR_CTC_1B", "omniASR_LLM_1B", "omniASR_CTC_3B", "omniASR_LLM_3B", "omniASR_CTC_7B", "omniASR_LLM_7B"]
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
    pipeline = ASRInferencePipeline(model_card=model_id)
    
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
                try:
                    # lang = ['acm_Arab', 'acw_Arab', 'aeb_Arab', 'aec_Arab', 'afb_Arab', 'apc_Arab', 'apd_Arab', 'arb_Arab', 'arq_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'ayl_Arab', 'ayp_Arab', 'aze_Arab', 'bcc_Arab', 'bft_Arab', 'bgp_Arab', 'bqi_Arab', 'brh_Arab', 'bsh_Arab', 'btv_Arab', 'ckb_Arab', 'dcc_Arab', 'dmk_Arab', 'dml_Arab', 'fas_Arab', 'ggg_Arab', 'gig_Arab', 'gjk_Arab', 'gju_Arab', 'glk_Arab', 'gwc_Arab', 'gwt_Arab', 'hno_Arab', 'kas_Arab', 'khw_Arab', 'kmr_Arab', 'kur_Arab', 'kvx_Arab', 'kxp_Arab', 'lrk_Arab', 'lss_Arab', 'mki_Arab', 'mve_Arab', 'mvy_Arab', 'odk_Arab', 'oru_Arab', 'pbt_Arab', 'pbu_Arab', 'phl_Arab', 'phr_Arab', 'plk_Arab', 'pnb_Arab', 'pst_Arab', 'pus_Arab', 'rif_Arab', 'sbn_Arab', 'scl_Arab', 'skr_Arab', 'snd_Arab', 'ssi_Arab', 'trw_Arab', 'tuk_Arab', 'uig_Arab', 'urd_Arab', 'ush_Arab', 'xhe_Arab', 'xka_Arab', 'ydg_Arab']
                    transcriptions = pipeline.transcribe([in_path], lang=['arb_Arab'], batch_size=1)
                    
                except Exception as err:
                    print(f"{err} with file {in_path}")
                    continue
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
                    "pred_text": transcriptions[0],
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')
