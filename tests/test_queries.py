"""Validate data/queries.jsonl: schema, constraint parsing, relevance counts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_queries, load_recipes, relevant_ids  # noqa: E402
from src.retrieval.text import extract_constraints  # noqa: E402


class QuerySetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = load_queries()
        cls.recipes = load_recipes()

    def test_size_and_languages(self):
        self.assertGreaterEqual(len(self.queries), 30)
        langs = [q["lang"] for q in self.queries]
        self.assertEqual(langs.count("tr"), langs.count("en"))

    def test_ids_are_unique(self):
        ids = [q["id"] for q in self.queries]
        self.assertEqual(len(ids), len(set(ids)))

    def test_stored_constraints_match_the_text(self):
        """If the regex stops parsing a query, the benchmark would silently change."""
        for q in self.queries:
            c = extract_constraints(q["text"])
            expected = set(q["diet"]) | ({"vegetarian"} if "vegan" in q["diet"] else set())
            with self.subTest(q=q["id"]):
                self.assertEqual(c.max_minutes, q["max_minutes"])
                self.assertEqual(c.diet, expected)

    def test_every_query_has_enough_relevant_recipes(self):
        for q in self.queries:
            with self.subTest(q=q["id"]):
                self.assertGreaterEqual(len(relevant_ids(q, self.recipes)), 3)


if __name__ == "__main__":
    unittest.main()
