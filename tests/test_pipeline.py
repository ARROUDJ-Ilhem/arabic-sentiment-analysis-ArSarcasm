"""
Unit tests for preprocessing and dataset.
Run locally with:  pytest tests/ -v

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import torch
from src.preprocess import clean_arabic_tweet
from src.dataset import LABEL2ID, ID2LABEL, NUM_LABELS
from src.evaluate import compute_metrics_from_arrays
import numpy as np


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestCleanArabicTweet:

    def test_removes_url(self):
        raw = "تغريدة مع رابط http://example.com نهاية"
        cleaned = clean_arabic_tweet(raw)
        assert 'http' not in cleaned
        assert 'تغريدة' in cleaned

    def test_removes_mention(self):
        raw = "@user هذا النص جيد"
        cleaned = clean_arabic_tweet(raw)
        assert '@' not in cleaned
        assert 'النص' in cleaned

    def test_removes_hashtag_symbol(self):
        raw = "هذا #رائع جدا"
        cleaned = clean_arabic_tweet(raw)
        assert '#' not in cleaned

    def test_collapses_repeated_chars(self):
        raw = "مممممتاز"
        cleaned = clean_arabic_tweet(raw)
        # After collapse: no char repeated more than once consecutively
        import re
        assert not re.search(r'(.)\1{2,}', cleaned)

    def test_empty_string_input(self):
        assert clean_arabic_tweet("") == ""

    def test_none_input(self):
        assert clean_arabic_tweet(None) == ""

    def test_latin_only_becomes_empty(self):
        raw = "hello world 123"
        cleaned = clean_arabic_tweet(raw)
        # All non-Arabic chars removed → empty or whitespace only
        assert cleaned.strip() == ""

    def test_output_is_string(self):
        assert isinstance(clean_arabic_tweet("أي نص"), str)


# ── Dataset / label tests ─────────────────────────────────────────────────────

class TestLabels:

    def test_label2id_has_four_classes(self):
        assert NUM_LABELS == 4

    def test_label2id_id2label_inverse(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_all_expected_labels_present(self):
        expected = {'Positive', 'Negative', 'Neutral', 'Mixed'}
        assert set(LABEL2ID.keys()) == expected


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3])
        metrics = compute_metrics_from_arrays(y, y)
        assert metrics['macro_f1'] == pytest.approx(1.0)
        assert metrics['accuracy'] == pytest.approx(1.0)

    def test_macro_f1_range(self):
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 2, 2, 3, 1, 1])
        metrics = compute_metrics_from_arrays(y_true, y_pred)
        assert 0.0 <= metrics['macro_f1'] <= 1.0
        assert 0.0 <= metrics['accuracy'] <= 1.0


# ── Run with pytest ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Also runnable directly: python tests/test_pipeline.py
    import unittest
    unittest.main()
