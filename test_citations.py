#!/usr/bin/env python3
"""Regression tests for final-answer citation validation."""

import os
import contextlib
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MEILI_URL", "http://localhost:7700")
os.environ.setdefault("MEILI_INDEX", "test")
os.environ.setdefault("MEILI_SEARCH_KEY", "test")
os.environ.setdefault("GEMINI_API_KEY", "test")

from agent import server  # noqa: E402
from agent.server import (  # noqa: E402
    append_conversation_message,
    create_conversation,
    delete_conversation_for_user,
    enforce_citation_urls,
    apply_clinical_contradiction_guardrails,
    detects_aom_under_two_overstatement,
    expand_query_variants,
    extracted_citations,
    generate_content_with_retry,
    get_conversation,
    get_conversation_db,
    journal_matches,
    journal_match_score,
    list_conversations_for_user,
    load_messages_for_conversation,
    normalize_conversation_title,
    normalize_citation_markdown,
    prepare_citation_response,
    topic_penalty_for_hit,
    upsert_user,
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

    def test_enforce_citation_urls_repairs_title_only_citation(self):
        retrieved = {"https://pubmed.ncbi.nlm.nih.gov/35872300/"}
        sources = {
            "https://pubmed.ncbi.nlm.nih.gov/35872300/": {
                "title": "Ototoxicity in childhood",
                "year": 2022,
            }
        }
        reply = "Evidence supports monitoring [Ototoxicity in childhood (2022)]."
        repaired = (
            "Evidence supports monitoring "
            "[Ototoxicity in childhood (2022)](https://pubmed.ncbi.nlm.nih.gov/35872300/)."
        )

        with patch("agent.server.repair_citation_markdown", return_value=repaired):
            checked_reply, missing_urls, citations, format_warnings, repair_attempted = enforce_citation_urls(
                reply,
                retrieved,
                sources,
            )

        self.assertTrue(repair_attempted)
        self.assertEqual(missing_urls, [])
        self.assertEqual(format_warnings, [])
        self.assertEqual(citations, [{
            "label": "Ototoxicity in childhood (2022)",
            "url": "https://pubmed.ncbi.nlm.nih.gov/35872300/",
        }])
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/35872300/", checked_reply)

    def test_enforce_citation_urls_keeps_answer_if_repair_fails(self):
        retrieved = {"https://pubmed.ncbi.nlm.nih.gov/35872300/"}
        sources = {
            "https://pubmed.ncbi.nlm.nih.gov/35872300/": {
                "title": "Ototoxicity in childhood",
                "year": 2022,
            }
        }
        reply = "Evidence supports monitoring [Ototoxicity in childhood (2022)]."

        with patch("agent.server.repair_citation_markdown", side_effect=RuntimeError("upstream 500")):
            checked_reply, missing_urls, citations, format_warnings, repair_attempted = enforce_citation_urls(
                reply,
                retrieved,
                sources,
            )

        self.assertTrue(repair_attempted)
        self.assertEqual(checked_reply, reply)
        self.assertEqual(missing_urls, [])
        self.assertEqual(citations, [])
        self.assertEqual(len(format_warnings), 2)
        self.assertIn("Citation URL repair failed", format_warnings[1])


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


class ClinicalGuardrailTests(unittest.TestCase):
    def test_detects_overbroad_under_two_aom_watchful_waiting_claim(self):
        reply = (
            "Consequently, clinical practice guidelines mandate prompt treatment for all children "
            "under 2 years of age with acute otitis media rather than watchful waiting, regardless "
            "of whether the AOM is unilateral or bilateral, or mild or severe."
        )

        self.assertTrue(detects_aom_under_two_overstatement(reply))

    def test_does_not_flag_qualified_under_two_aom_guidance(self):
        reply = (
            "This premise is overbroad: children 6-23 months with nonsevere unilateral AOM may be "
            "managed with observation/watchful waiting when follow-up is reliable, while bilateral "
            "AOM, otorrhea, or severe symptoms warrant antibiotics."
        )

        self.assertFalse(detects_aom_under_two_overstatement(reply))

    def test_does_not_flag_qualified_aom_guidance_when_nonsevere_precedes_unilateral(self):
        reply = (
            "All children under 2 with bilateral AOM need antibiotics. Nonsevere unilateral cases "
            "may use observation/watchful waiting when follow-up is reliable."
        )

        self.assertFalse(detects_aom_under_two_overstatement(reply))

    def test_does_not_flag_qualified_aom_guidance_with_without_severe_wording(self):
        reply = (
            "Watchful waiting may be considered for children 6 months to 2 years with unilateral "
            "AOM without severe symptoms."
        )

        self.assertFalse(detects_aom_under_two_overstatement(reply))

    def test_does_not_flag_benign_all_the_studies_phrase(self):
        reply = (
            "All the studies on children under 2 with AOM compared antibiotic therapy versus "
            "watchful waiting and reported subgroup results."
        )

        self.assertFalse(detects_aom_under_two_overstatement(reply))

    def test_detects_rationale_answer_that_accepts_under_two_prompt_treatment_premise(self):
        reply = (
            "The rationale for prompt antibiotic treatment in children under 2 years of age, "
            "as opposed to watchful waiting, is driven by greater therapeutic benefit and a "
            "higher risk profile in AOM."
        )

        self.assertTrue(detects_aom_under_two_overstatement(reply))

    def test_detects_prioritizing_prompt_treatment_over_watchful_waiting_phrase(self):
        reply = (
            "The rationale for prioritizing prompt antibiotic treatment over watchful waiting in "
            "children under 2 years of age with AOM is based on higher treatment efficacy."
        )

        self.assertTrue(detects_aom_under_two_overstatement(reply))

    def test_guardrail_appends_aom_under_two_correction_with_retrieved_aap_link(self):
        reply = (
            "Guidelines require prompt antibiotic treatment for all children under 2 with AOM "
            "rather than watchful waiting."
        )
        sources = {
            "https://pubmed.ncbi.nlm.nih.gov/23439909/": {
                "title": "The diagnosis and management of acute otitis media",
                "year": 2013,
            }
        }

        guarded, warnings = apply_clinical_contradiction_guardrails(reply, sources)

        self.assertEqual(warnings, [
            "Corrected overbroad AOM watchful-waiting claim for children under 2."
        ])
        self.assertIn("overbroad", guarded)
        self.assertIn("6-23 months with nonsevere unilateral AOM", guarded)
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/23439909/", guarded)

    def test_aom_query_expansion_adds_pediatric_guideline_variants(self):
        variants = expand_query_variants(
            "Why do children under 2 require prompt treatment rather than watchful waiting for AOM?"
        )

        self.assertTrue(any("AAP AAFP acute otitis media guideline" in variant for variant in variants))


class ModelRetryTests(unittest.TestCase):
    def test_retries_transient_model_error(self):
        expected = object()

        with (
            patch("agent.server.MODEL_RETRY_ATTEMPTS", 2),
            patch("agent.server.MODEL_RETRY_BASE_DELAY_SECONDS", 0),
            patch("agent.server.random.uniform", return_value=0),
            patch("agent.server.time.sleep") as sleep,
            patch(
                "agent.server.client.models.generate_content",
                side_effect=[RuntimeError("500 INTERNAL"), expected],
            ) as generate,
        ):
            result = generate_content_with_retry(
                phase="test",
                model="gemma-4-31b-it",
                contents="hello",
                config=None,
            )

        self.assertIs(result, expected)
        self.assertEqual(generate.call_count, 2)
        sleep.assert_called_once()

    def test_does_not_retry_non_transient_model_error(self):
        with (
            patch("agent.server.MODEL_RETRY_ATTEMPTS", 3),
            patch("agent.server.time.sleep") as sleep,
            patch(
                "agent.server.client.models.generate_content",
                side_effect=ValueError("invalid request"),
            ) as generate,
        ):
            with self.assertRaises(ValueError):
                generate_content_with_retry(
                    phase="test",
                    model="gemma-4-31b-it",
                    contents="hello",
                    config=None,
                )

        self.assertEqual(generate.call_count, 1)
        sleep.assert_not_called()


class JournalFilterTests(unittest.TestCase):
    def test_matches_abbreviated_jama_otolaryngology_title(self):
        self.assertTrue(
            journal_matches(
                "JAMA Otolaryngol Head Neck Surg",
                "JAMA Otolaryngology-- Head & Neck Surgery",
            )
        )

    def test_matches_abbreviated_otology_neurotology_title(self):
        self.assertTrue(
            journal_matches(
                "Otol Neurotol",
                "Otology & Neurotology",
            )
        )

    def test_rejects_unrelated_journal(self):
        self.assertLess(
            journal_match_score("Otolaryngology Head Neck Surgery", "The Laryngoscope"),
            0.8,
        )
        self.assertFalse(journal_matches("Otolaryngology Head Neck Surgery", "The Laryngoscope"))

    def test_distinctive_journal_token_must_match(self):
        self.assertFalse(
            journal_matches(
                "JAMA Otolaryngol Head Neck Surg",
                "Archives of otolaryngology--head & neck surgery",
            )
        )
        self.assertFalse(
            journal_matches(
                "Otolaryngology Head Neck Surgery",
                "Archives of otolaryngology--head & neck surgery",
            )
        )


class ConversationStoreTests(unittest.TestCase):
    def test_title_truncation_marks_omitted_text(self):
        title = normalize_conversation_title("x" * 80)

        self.assertEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))

    def test_create_list_load_and_delete_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.server.CONVERSATION_DB_PATH",
            os.path.join(tmpdir, "conversations.sqlite"),
        ):
            with contextlib.closing(get_conversation_db()) as db:
                upsert_user(db, "user-a")
                conversation_id = create_conversation(db, "user-a", "  First   clinical question  ")
                append_conversation_message(db, conversation_id, "user", "First clinical question")
                append_conversation_message(db, conversation_id, "assistant", "Answer")
                db.commit()

                conversations = list_conversations_for_user(db, "user-a")
                messages = load_messages_for_conversation(db, conversation_id)
                deleted = delete_conversation_for_user(db, "user-a", conversation_id)
                remaining_messages = load_messages_for_conversation(db, conversation_id)

            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0]["title"], "First clinical question")
            self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
            self.assertTrue(deleted)
            self.assertEqual(remaining_messages, [])

    def test_ownership_checks_do_not_return_other_users_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.server.CONVERSATION_DB_PATH",
            os.path.join(tmpdir, "conversations.sqlite"),
        ):
            with contextlib.closing(get_conversation_db()) as db:
                upsert_user(db, "user-a")
                upsert_user(db, "user-b")
                conversation_id = create_conversation(db, "user-a", "Question")
                db.commit()

                self.assertIsNone(get_conversation(db, "user-b", conversation_id))
                self.assertFalse(delete_conversation_for_user(db, "user-b", conversation_id))
                self.assertIsNotNone(get_conversation(db, "user-a", conversation_id))


class ConversationApiTests(unittest.TestCase):
    def test_chat_creates_conversation_and_persists_visible_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.server.CONVERSATION_DB_PATH",
            os.path.join(tmpdir, "conversations.sqlite"),
        ), patch("agent.server.run_agent", return_value={"reply": "Stored answer", "citations": []}):
            with server.app.test_client() as client:
                response = client.post(
                    "/chat",
                    json={"user_id": "user-a", "message": "What about tympanostomy tubes?"},
                )
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                conversation_id = payload["conversation_id"]

                loaded = client.get(f"/api/conversations/{conversation_id}?user_id=user-a")
                loaded_payload = loaded.get_json()

            self.assertEqual(loaded.status_code, 200)
            self.assertEqual(
                [m["role"] for m in loaded_payload["messages"]],
                ["user", "assistant"],
            )
            self.assertEqual(loaded_payload["messages"][1]["content"], "Stored answer")

    def test_wrong_user_get_delete_and_chat_return_404(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.server.CONVERSATION_DB_PATH",
            os.path.join(tmpdir, "conversations.sqlite"),
        ), patch("agent.server.run_agent", return_value={"reply": "Answer", "citations": []}):
            with server.app.test_client() as client:
                created = client.post(
                    "/chat",
                    json={"user_id": "user-a", "message": "Question"},
                ).get_json()
                conversation_id = created["conversation_id"]

                get_response = client.get(f"/api/conversations/{conversation_id}?user_id=user-b")
                delete_response = client.delete(f"/api/conversations/{conversation_id}?user_id=user-b")
                chat_response = client.post(
                    "/chat",
                    json={
                        "user_id": "user-b",
                        "conversation_id": conversation_id,
                        "message": "Follow up",
                    },
                )

            self.assertEqual(get_response.status_code, 404)
            self.assertEqual(delete_response.status_code, 404)
            self.assertEqual(chat_response.status_code, 404)

    def test_chat_context_uses_latest_30_visible_messages(self):
        captured = {}

        def fake_run_agent(contents, include_trace=False):
            captured["texts"] = [part.text for content in contents for part in content.parts if part.text]
            return {"reply": "Answer", "citations": []}

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agent.server.CONVERSATION_DB_PATH",
            os.path.join(tmpdir, "conversations.sqlite"),
        ), patch("agent.server.run_agent", side_effect=fake_run_agent):
            with contextlib.closing(get_conversation_db()) as db:
                upsert_user(db, "user-a")
                conversation_id = create_conversation(db, "user-a", "Question 0")
                for i in range(35):
                    append_conversation_message(db, conversation_id, "user", f"prior {i}")
                db.commit()

            with server.app.test_client() as client:
                response = client.post(
                    "/chat",
                    json={
                        "user_id": "user-a",
                        "conversation_id": conversation_id,
                        "message": "newest",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured["texts"]), 30)
        self.assertEqual(captured["texts"][0], "prior 6")
        self.assertEqual(captured["texts"][-1], "newest")


if __name__ == "__main__":
    unittest.main()
