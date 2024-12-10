# Open Universal Arabic ASR Leaderboard

This repository contains the evaluation code for the Open Universal Arabic ASR Leaderboard, which is a continuous benchmark project for open-source multi-dialectal Arabic ASR models across various multi-dialectal datasets. The leaderboard is hosted at [elmresearchcenter/open_universal_arabic_asr_leaderboard](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard). For more detailed analysis such as models' robustness, speaker adaption, model efficiency and memory usage, please check our [paper]().

# Requirements

We collected models from different toolkits, such as HuggingFace, SpeechBrain, Nvidia-NeMo, etc. Requirements for each library can be installed to evaluate a desired model. To intall all the dependencies, run:
bash```
pip install -r requirements/requirements.txt
```

# Evaluate a model

1. Go to `models/`, run the corresponding model inference function, which will generate an output manifest file.
2. Run the `calculate_wer` function in `eval.py`

# Add a new model

Please run the above evaluation for all the 5 test sets, calculate the average WER/CER, then launch an issue or PR letting us know about your model, training data, and its performance.

# Citation 

```bibtex
to be added
```