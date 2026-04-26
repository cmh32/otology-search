#!/usr/bin/env python3
"""Regression tests for final-answer citation validation."""

import os
import unittest

os.environ.setdefault("MEILI_URL", "http://localhost:7700")
os.environ.setdefault("MEILI_INDEX", "test")
os.environ.setdefault("MEILI_SEARCH_KEY", "test")
os.environ.setdefault("GEMINI_API_KEY", "test")

from agent.server import (  # noqa: E402
    extracted_citations,
    normalize_citation_markdown,
    prepare_citation_response,
    topic_penalty_for_hit,
)


class CitationTests(unittest.TestCase):
    def test_normalizes_double_bracket_citations(self):
        reply = "[[Ototoxicity in childhood (2022)](https://pubmed.ncbi.nlm.nih.gov/35872300/)"

        normalized = normalize_citation_markdown(reply)
        citations = extracted_citations(normalized)

        self.assertEqual(
            normalized,
            "[Ototoxicity in childhood (2022)](https://pubmed.ncbi.nlm.nih.gov/35872300/)",
        )
        self.assertEqual(citations, [{
            "label": "Ototoxicity in childhood (2022)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/35872300/",
        }])

    def test_prepare_response_dedupes_and_accepts_retrieved_urls(self):
        retrieved = {"https://pubmed.ncbi.nlm.nih.gov/35872300/"}
        reply = (
            "[[Ototoxicity in childhood (2022)](https://pubmed.ncbi.nlm.nih.gov/35872300/) "
            "and [Ototoxicity again (2022)](https://pubmed.ncbi.nlm.nih.gov/35872300/)"
        )

        checked_reply, missing_urls, citations, format_warnings = prepare_citation_response(reply, retrieved)

        self.assertIn("[Ototoxicity in childhood (2022)]", checked_reply)
        self.assertEqual(missing_urls, [])
        self.assertEqual(format_warnings, [])
        self.assertEqual(len(citations), 1)

    def test_warns_on_unretrieved_pubmed_url(self):
        reply = "[Other paper (2024)](https://pubmed.ncbi.nlm.nih.gov/12345678/)"

        checked_reply, missing_urls, citations, format_warnings = prepare_citation_response(reply, set())

        self.assertIn("Citation check:", checked_reply)
        self.assertEqual(missing_urls, ["https://pubmed.ncbi.nlm.nih.gov/12345678/"])
        self.assertEqual(citations[0]["url"], "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        self.assertEqual(format_warnings, [])

    def test_warns_when_retrieval_happened_but_no_valid_citations(self):
        retrieved = {"https://pubmed.ncbi.nlm.nih.gov/35872300/"}
        reply = "Evidence supports monitoring [Ototoxicity in childhood (2022)]."

        _checked_reply, missing_urls, citations, format_warnings = prepare_citation_response(reply, retrieved)

        self.assertEqual(missing_urls, [])
        self.assertEqual(citations, [])
        self.assertEqual(len(format_warnings), 1)


class TopicPenaltyTests(unittest.TestCase):
    def test_penalizes_cochlear_implant_hit_for_ossiculoplasty_query(self):
        penalty = topic_penalty_for_hit(
            "What predicts hearing outcomes after ossiculoplasty with PORP or TORP?",
            "Hearing outcomes after cochlear implantation",
            "Speech perception improved after cochlear implant surgery.",
            ["Cochlear Implantation", "Hearing"],
        )

        self.assertGreater(penalty, 0)

    def test_penalizes_missing_ossiculoplasty_marker_for_ossiculoplasty_query(self):
        penalty = topic_penalty_for_hit(
            "What predicts hearing outcomes after ossiculoplasty with PORP or TORP?",
            "Predictive factors of successful tympanoplasty",
            "Adults with dry tympanic membrane perforation underwent tympanoplasty.",
            ["Tympanoplasty", "Hearing"],
        )

        self.assertGreater(penalty, 0)

    def test_does_not_penalize_ossicular_reconstruction_hit(self):
        penalty = topic_penalty_for_hit(
            "What predicts hearing outcomes after ossiculoplasty with PORP or TORP?",
            "PORP vs. TORP: a meta-analysis",
            "Ossicular reconstruction with partial and total ossicular replacement prostheses.",
            ["Ossicular Prosthesis", "Tympanoplasty"],
        )

        self.assertEqual(penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
