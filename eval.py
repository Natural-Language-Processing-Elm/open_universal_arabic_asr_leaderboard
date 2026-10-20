from tqdm import tqdm
import json
from nemo.collections.asr.metrics.wer import word_error_rate
from tqdm import tqdm
import re

def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[\u064B-\u0652]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # Attach the conjunction و to the following word
    text = re.sub(r'(^|\s)و\s+', r'\1و', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_wer(output_manifest):
    """
    Arguments
    ---------
    output_manifest: str
        path to the output manifest of the model inference

    Output
    ---------
    WER/CER
    """
    predictions = []
    target_transcripts = []
    with open(output_manifest, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            target_transcripts.append(normalize_arabic_text(item['text']))
            predictions.append(normalize_arabic_text(item['pred_text']))
    wer = word_error_rate(predictions, target_transcripts)
    print("wer : ", wer)
    cer = word_error_rate(predictions, target_transcripts, use_cer=True)
    print("cer : ", cer)
    return wer, cer


def calculate_wer_per_dialect(output_manifest, data_manifest):
    """
    Compute WER/CER per dialect for test sets whose data manifest carries
    a "dialect" field per sample (e.g. datasets/casablanca_test.json).
    The output manifest is joined to the data manifest by audio file name.

    Arguments
    ---------
    output_manifest: str
        path to the output manifest of the model inference
    data_manifest: str
        path to the data manifest under datasets/ with dialect labels

    Output
    ---------
    dict mapping each dialect to its {"wer": ..., "cer": ...}
    """
    dialect_labels = {}
    with open(data_manifest, "r") as f:
        for line in f:
            item = json.loads(line)
            if "dialect" in item:
                file_name = item['audio_filepath'].split("/")[-1]
                dialect_labels[file_name] = item["dialect"]

    predictions = {}
    target_transcripts = {}
    with open(output_manifest, "r") as f:
        for line in tqdm(f):
            item = json.loads(line)
            file_name = item['audio_filepath'].split("/")[-1]
            dialect = dialect_labels.get(file_name)
            if dialect is None:
                continue
            target_transcripts.setdefault(dialect, []).append(normalize_arabic_text(item['text']))
            predictions.setdefault(dialect, []).append(normalize_arabic_text(item['pred_text']))

    results = {}
    for dialect in sorted(predictions):
        wer = word_error_rate(predictions[dialect], target_transcripts[dialect])
        cer = word_error_rate(predictions[dialect], target_transcripts[dialect], use_cer=True)
        print(dialect, "({} samples)".format(len(predictions[dialect])), "wer : ", wer, "cer : ", cer)
        results[dialect] = {"wer": wer, "cer": cer}
    return results
