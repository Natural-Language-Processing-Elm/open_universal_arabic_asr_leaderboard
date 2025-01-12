# Open Universal Arabic ASR Leaderboard

This repository contains the evaluation code for the Open Universal Arabic ASR Leaderboard, a continuous benchmark project for open-source multi-dialectal Arabic ASR models across various multi-dialectal datasets. The leaderboard is hosted at [elmresearchcenter/open_universal_arabic_asr_leaderboard](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard). For more detailed analysis such as models' robustness, speaker adaption, model efficiency and memory usage, please check our [paper](https://arxiv.org/pdf/2412.13788).

# Updates

- [2025/01/11]: New model included: [Nvidia Parakeet-CTC-XXL-1.1B-Universal](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-1-1b-unified-ml-cs-universal)
- [2025/01/11]: New model included: [Nvidia Parakeet-CTC-XXL-1.1B-Concat](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-1-1b-unified-ml-cs-concat)
- [2023/01/11]: New dataset included: [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca)

# Datasets

Please first download the following test sets

| Test Set                                                                                        | Num Dialects   | Test (h)    |
|-------------------------------------------------------------------------------------------------|----------------|-------------|
| [SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022)                                      | 10             | 10.7        |
| [Common Voice 18.0](https://commonvoice.mozilla.org/en/datasets)                                | 25             | 12.6        |
| [MASC (Clean-Test)](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus)    | 7              | 10.5        |
| [MASC (Noisy-Test)](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus)    | 8              | 14.9        |
| [MGB-2](http://www.mgb-challenge.org/MGB-2.html)                                                | Unspecified    | 9.6         |
| [Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca)                                | 8              | 7.7         |

# Requirements

We collected models from different toolkits, such as HuggingFace, SpeechBrain, Nvidia-NeMo, etc. Requirements for each library can be installed to evaluate a desired model. To intall all the dependencies, run:
```bash
pip install -r requirements.txt
```

# Evaluate a model

We provide easy-to-use inference functions, to run an ASR model:
1. Go to `models/`, and run the corresponding model inference function to generate an output manifest file containing ground-truths and predictions.
2. Run the `calculate_wer` function in `eval.py` on the output manifest file.
3. Details can be found in the methods' docstrings.

# Add a new model

Please run the above evaluation for all the 5 test sets under `datasets/`, calculate the average WER/CER, then launch an issue or PR letting us know about your model, training data, and its performance.

We welcome models that:
1. with a model architecture that is not present in the leaderboard.
2. avoid using training sets in the same dataset as the test sets to avoid the in-domain issue.

# Citation 

```bibtex
@article{wang2024open,
  title={Open Universal Arabic ASR Leaderboard},
  author={Wang, Yingzhi and Alhmoud, Anas and Alqurishi, Muhammad},
  journal={arXiv preprint arXiv:2412.13788},
  year={2024}
}
```
