"""
Arabic tweet cleaning pipeline.
"""

import re
import pyarabic.araby as araby


def clean_arabic_tweet(text: str) -> str:
    """
    Clean an Arabic tweet for sentiment analysis.

    Steps:
    1. Remove URLs
    2. Remove @mentions, strip # keep hashtag word
    3. Keep only Arabic characters + spaces
    4. Normalize Alef variants and Ta marbuta
    5. Remove diacritics (tashkeel)
    6. Collapse character repetitions (>2 -> 1)
    7. Strip extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)       # 1. URLs
    text = re.sub(r'@\w+', '', text)                    # 2. mentions
    text = re.sub(r'#', '', text)                       # 2. hashtag symbol
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)    # 3. Arabic only
    text = araby.normalize_alef(text)                   # 4. Alef normalization
    text = araby.normalize_teh(text)                    # 4. Ta marbuta
    text = araby.strip_tashkeel(text)                   # 5. diacritics
    text = re.sub(r'(.)\1{2,}', r'\1', text)           # 6. repetitions
    text = re.sub(r'\s+', ' ', text).strip()            # 7. whitespace
    return text


if __name__ == "__main__":
    # Quick sanity check
    # Run with:  python src/preprocess.py
    examples = [
        "تغريدة مع رابط http://example.com نهاية",
        "@user هذا النص جيد جداً!!",
        "مممممتاز جداً #عربي هذا الفيلم",
        "ما عجبني الفيلم أبداً",
    ]
    print("── Preprocessing sanity check ──\n")
    for raw in examples:
        cleaned = clean_arabic_tweet(raw)
        print(f"  Raw:   {raw}")
        print(f"  Clean: {cleaned}")
        print()
