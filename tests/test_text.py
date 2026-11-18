"""Unit tests for the text pipeline (no torch, no model weights needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.text import (  # noqa: E402
    detect_lang, extract_constraints, normalize, tokenize,
)


class LanguageTests(unittest.TestCase):
    def test_turkish(self):
        self.assertEqual(detect_lang("vejetaryen, 30 dakikadan kısa"), "tr")
        self.assertEqual(detect_lang("fırında falafel"), "tr")

    def test_english(self):
        self.assertEqual(detect_lang("vegetarian pizza under 30 minutes"), "en")
        self.assertEqual(detect_lang("quick chocolate dessert"), "en")

    def test_loanword_does_not_flip_language(self):
        # "vegan" is spelled the same in both languages; it must not make
        # an English query look Turkish (regression test).
        self.assertEqual(detect_lang("quick vegan dinner"), "en")


class ConstraintTests(unittest.TestCase):
    def test_english_time(self):
        self.assertEqual(extract_constraints("under 30 minutes").max_minutes, 30)
        self.assertEqual(extract_constraints("20 minutes or less").max_minutes, 20)
        self.assertEqual(extract_constraints("no more than 2 hours").max_minutes, 120)
        self.assertEqual(extract_constraints("max 60 minutes").max_minutes, 60)

    def test_turkish_time(self):
        self.assertEqual(extract_constraints("30 dakikadan kısa").max_minutes, 30)
        self.assertEqual(extract_constraints("en fazla 20 dakika").max_minutes, 20)
        self.assertEqual(extract_constraints("1 saatten kısa").max_minutes, 60)

    def test_quick_means_thirty(self):
        self.assertEqual(extract_constraints("quick pancakes").max_minutes, 30)
        self.assertEqual(extract_constraints("hızlı pankek").max_minutes, 30)

    def test_diet(self):
        self.assertEqual(extract_constraints("vegetarian pizza").diet, {"vegetarian"})
        self.assertEqual(extract_constraints("etsiz suşi").diet, {"vegetarian"})
        # vegan implies vegetarian
        self.assertEqual(extract_constraints("vegan ramen").diet, {"vegan", "vegetarian"})


class NormalizationTests(unittest.TestCase):
    def test_turkish_dotted_i(self):
        # "İ" always lower-cases to "i"; "I" becomes "ı" once the text is Turkish.
        self.assertEqual(normalize("İstanbul"), "istanbul")
        self.assertEqual(normalize("ISPANAK çorbası"), "ıspanak çorbası")

    def test_tokens_are_translated_and_stemmed(self):
        tokens = tokenize("hızlı pankek tarifi")
        self.assertIn("quick", tokens)
        self.assertTrue(any(t.startswith("pancak") for t in tokens))


if __name__ == "__main__":
    unittest.main()
