#!/usr/bin/env python3
"""Flask backend for the Otology Literature Research Agent."""

import json
import contextlib
import math
import os
import random
import re
import hashlib
import sqlite3
import time
import urllib.request
import urllib.error
import datetime
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SEARCH_APP_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "search-app"))

app = Flask(__name__)

MEILI_URL = os.environ["MEILI_URL"].rstrip("/")
MEILI_INDEX = os.environ["MEILI_INDEX"]
MEILI_SEARCH_KEY = os.environ["MEILI_SEARCH_KEY"]

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_INSTRUCTION = """You are a clinical literature research assistant for an APRN specializing in otology.
Use precise clinical terminology for a clinician peer; avoid generic patient-facing disclaimers and do not give patient-specific medical advice.
Your only source of truth is the literature retrieved with the tool. If evidence is thin or absent, say so clearly.
Cite each source you rely on as a markdown hyperlink: [Title (Year)](URL).
Do not cite by title alone, and do not group multiple titles inside one bracket. Each cited paper must be its own complete Markdown link with its PubMed URL.
Use exactly one opening bracket per citation link, like [Title (Year)](URL), never [[Title (Year)](URL).
Only cite papers returned by the search_papers tool. Never fabricate a title, year, or URL.

Scope: answer otology, hearing, vestibular, ear surgery, and closely related neurotology questions.
If the user asks about a clearly non-otology topic such as rhinology, laryngology, ophthalmology, or general medicine, say that it is outside this otology literature index and do not search.

You have access to a search_papers tool over a PubMed otology index.
Before answering, call the tool one or more times with focused queries.
When making a tool call, output only the function call — no surrounding text or explanation.
You have at most 5 tool-call turns. Spend them deliberately: use the first turns for broad coverage of the main clinical question and evidence hierarchy, and reserve a later turn for refinement if results are sparse, off-topic, or missing a key evidence type.
For complex questions, decompose the question and search each angle separately.
Use MeSH terms, year filters, and journal filters when they improve retrieval.
Use publication-type filters when the user asks for guidelines, recommendations, systematic reviews, meta-analyses, or randomized trials.
If a search returns few or no relevant results, broaden before answering: remove strict filters, use fewer terms, try synonyms or abbreviations, and search for guideline or review terms separately.
Search broadly first, then narrow.

Choose evidence based on the user's intent:
- The user is in the United States. For guideline-based management or standard-of-care questions, prioritize U.S. society or public-health guidance when relevant (for example AAP, AAFP, AAO-HNS, CDC, ACR, CNS) over newer international consensus statements; mention international differences only after the U.S. position is clear.
- For current indications, recommendations, guidelines, or standard-of-care questions, prioritize authoritative clinical practice guidelines and consensus statements from major societies, then high-level reviews before individual studies.
- For treatment-evidence questions, always issue at least one search targeting clinical practice guidelines or consensus statements on the topic in addition to searching for primary studies. Guidelines synthesize the evidence and define standard of care — they should anchor the answer even when the question is framed around individual studies or treatment comparisons.
- Rank primary evidence by study design: systematic reviews/meta-analyses, randomized trials, prospective comparative studies, then retrospective cohorts/case series.
- Use lower-level studies only when higher-level evidence is absent, conflicting, or too sparse.
- Do not present uncontrolled, single-center, or older cohort studies as strong evidence when higher-level evidence is weak or uncertain.
- When multiple guidelines on the same topic were retrieved, cite each relevant one in the answer — do not anchor on a single guideline when others address the same point. A guideline from a major US society (AAP, AAO-HNS, AAFP) warrants citation even when a more recent international consensus was also retrieved.
- If a guideline and a newer systematic review, meta-analysis, or trial point in different directions, state the conflict directly, prioritize the guideline for current standard-of-care framing, and explain whether the newer evidence is strong enough to qualify or challenge that guidance.
- For follow-up questions that ask "why" a previously stated clinical rule is true, first verify the premise against retrieved guideline evidence. If the premise is false, overbroad, or missing exceptions, correct it explicitly before explaining the narrower true rule.
- Avoid absolute clinical claims such as "all", "never", "always", "regardless", or "mandate" unless the retrieved guideline actually states that rule without relevant exceptions.
- When a recommendation depends on age, severity, laterality, otorrhea, prior treatment, or allergy status, preserve that decision matrix in the answer rather than collapsing it into a single prose rule.

In the final answer, distinguish when relevant between:
- strength of evidence
- magnitude of effect
- role in refractory or end-stage disease

If the user asks about treatment evidence or recommendations, organize the answer in this order:
1. Overall evidence quality.
2. Best-supported interventions or recommendations, grouped by clinical endpoint when relevant.
3. Recommendations or practices supported mainly by weak or very low-certainty evidence.
4. Major tradeoffs, harms, and residual uncertainty.

If the user asks for current indications or guideline-based management, organize the answer in this order:
1. Relevant guidelines and consensus positions (include issuing body and year for each).
2. Main indications or action statements.
3. Important situations where the guideline recommends against intervention or suggests observation first.
4. Important uncertainty, exceptions, or at-risk subgroups.

When summarizing evidence, include study design, sample size, and evidence quality when available.
Keep the answer under 300 words unless the user asks for more depth.
Do not use tables unless the user explicitly asks."""

FINAL_SYSTEM_INSTRUCTION = """You are a clinical literature research assistant for an APRN specializing in otology.
Use precise clinical terminology for a clinician peer; avoid generic patient-facing disclaimers and do not give patient-specific medical advice.
Your only source of truth is the literature already retrieved in this conversation. If evidence is thin or absent, say so clearly.
Only cite papers returned by the search_papers tool. Never fabricate a title, year, or URL.
Cite each source you rely on as a markdown hyperlink: [Title (Year)](URL).
Do not cite by title alone, and do not group multiple titles inside one bracket. Each cited paper must be its own complete Markdown link with its PubMed URL.
Use exactly one opening bracket per citation link, like [Title (Year)](URL), never [[Title (Year)](URL).

Write a concise synthesis from the retrieved papers. Do not call or request tools.
The user is in the United States. For guideline-based management or standard-of-care questions, prioritize U.S. society or public-health guidance when relevant over newer international consensus statements; mention international differences only after the U.S. position is clear.
When multiple guidelines on the same topic were retrieved, cite each relevant one — do not anchor on a single guideline when others address the same point.
If a guideline and a newer systematic review, meta-analysis, or trial point in different directions, state the conflict directly, prioritize the guideline for current standard-of-care framing, and explain whether the newer evidence is strong enough to qualify or challenge that guidance.
For follow-up questions that ask "why" a previously stated clinical rule is true, first verify the premise against retrieved guideline evidence. If the premise is false, overbroad, or missing exceptions, correct it explicitly before explaining the narrower true rule.
Avoid absolute clinical claims such as "all", "never", "always", "regardless", or "mandate" unless the retrieved guideline actually states that rule without relevant exceptions.
When a recommendation depends on age, severity, laterality, otorrhea, prior treatment, or allergy status, preserve that decision matrix in the answer rather than collapsing it into a single prose rule.

If the user asks about treatment evidence or recommendations, organize the answer in this order:
1. Overall evidence quality.
2. Best-supported interventions or recommendations, grouped by clinical endpoint when relevant.
3. Recommendations or practices supported mainly by weak or very low-certainty evidence.
4. Major tradeoffs, harms, and residual uncertainty.

If the user asks for current indications or guideline-based management, organize the answer in this order:
1. Relevant guidelines and consensus positions (include issuing body and year for each).
2. Main indications or action statements.
3. Important situations where the guideline recommends against intervention or suggests observation first.
4. Important uncertainty, exceptions, or at-risk subgroups.

Keep the answer under 300 words unless the user asks for more depth.
Do not use tables unless the user explicitly asks."""

CITATION_REPAIR_SYSTEM_INSTRUCTION = """You repair citation Markdown in otology literature answers.
Return only the revised answer text.
Do not change clinical wording except to repair citation markup.
Every source citation must be a complete Markdown link: [Title (Year)](https://pubmed.ncbi.nlm.nih.gov/PMID/).
Use only URLs from the provided retrieved source list.
If a bracketed source title in the answer matches a retrieved source, convert it to a Markdown link with that source's PubMed URL.
If a claim clearly cites a retrieved source by title without a URL, add the PubMed URL.
Never invent a title, year, PMID, or URL."""

SEARCH_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_papers",
        description=(
            "Search the PubMed otology literature database. "
            "Use focused keyword queries (e.g. 'cholesteatoma recurrence canal wall down'). "
            "Call multiple times for multi-part questions."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING",
                    description="Focused keyword search query",
                ),
                "mesh_terms": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="Optional PubMed MeSH terms to filter on when known",
                ),
                "publication_types": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description=(
                        "Optional PubMed publication types to filter on, such as "
                        "'Practice Guideline', 'Guideline', 'Systematic Review', "
                        "'Meta-Analysis', or 'Randomized Controlled Trial'"
                    ),
                ),
                "year_from": types.Schema(
                    type="INTEGER",
                    description="Optional lower bound publication year",
                ),
                "year_to": types.Schema(
                    type="INTEGER",
                    description="Optional upper bound publication year",
                ),
                "journal": types.Schema(
                    type="STRING",
                    description="Optional journal name or abbreviation to constrain results",
                ),
                "max_results": types.Schema(
                    type="INTEGER",
                    description="Optional number of reranked papers to return (default 10, max 12)",
                ),
            },
            required=["query"],
        ),
    )
])

AGENT_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[SEARCH_TOOL],
)
FINAL_CONFIG = types.GenerateContentConfig(system_instruction=FINAL_SYSTEM_INSTRUCTION)
CITATION_REPAIR_CONFIG = types.GenerateContentConfig(system_instruction=CITATION_REPAIR_SYSTEM_INSTRUCTION)

MAX_TOOL_TURNS = 5
FETCH_LIMIT = 60   # candidates pulled from Meilisearch
RERANK_LIMIT = 12  # returned to the model after semantic re-ranking
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "text-embedding-3-large" if EMBEDDING_PROVIDER == "openai" else "gemini-embedding-001",
)
EMBEDDING_RETRY_DELAY_SECONDS = 2
MODEL_RETRY_ATTEMPTS = int(os.environ.get("MODEL_RETRY_ATTEMPTS", "3"))
MODEL_RETRY_BASE_DELAY_SECONDS = float(os.environ.get("MODEL_RETRY_BASE_DELAY_SECONDS", "1"))
EMBEDDING_CACHE_PATH = os.environ.get("EMBEDDING_CACHE_PATH", "data/runtime/embedding-cache.sqlite")
CONVERSATION_DB_PATH = os.environ.get("CONVERSATION_DB_PATH", "data/runtime/conversations.sqlite")
CONVERSATION_CONTEXT_MESSAGE_LIMIT = int(os.environ.get("CONVERSATION_CONTEXT_MESSAGE_LIMIT", "30"))
DISABLE_EMBEDDING_CACHE = os.environ.get("DISABLE_EMBEDDING_CACHE", "").lower() in {"1", "true", "yes"}
MIN_FILTERED_HITS = 3
MEILI_HYBRID_SEARCH = os.environ.get("MEILI_HYBRID_SEARCH", "1").lower() in {"1", "true", "yes"}
MEILI_HYBRID_EMBEDDER = os.environ.get("MEILI_HYBRID_EMBEDDER", "otology_openai_large")
MEILI_HYBRID_PROVIDER = os.environ.get("MEILI_HYBRID_PROVIDER", "openai").strip().lower()
MEILI_HYBRID_MODEL = os.environ.get("MEILI_HYBRID_MODEL", "text-embedding-3-large")
MEILI_HYBRID_SEMANTIC_RATIO = float(os.environ.get("MEILI_HYBRID_SEMANTIC_RATIO", "0.3"))
BOOST_TOPIC_GATE_THRESHOLD = float(os.environ.get("BOOST_TOPIC_GATE_THRESHOLD", "0.55"))
BOOST_TOPIC_GATE_FACTOR = float(os.environ.get("BOOST_TOPIC_GATE_FACTOR", "0.25"))
OSSICULOPLASTY_QUERY_TERMS = {
    "ossiculoplasty",
    "ossicular",
    "porp",
    "torp",
}
OSSICULOPLASTY_ON_TOPIC_MARKERS = {
    "ossiculoplasty",
    "ossicular",
    "ossicular chain",
    "ossicular replacement",
    "ossicular reconstruction",
    "ossicular prosthesis",
    "partial ossicular",
    "total ossicular",
    "porp",
    "torp",
    "malleus",
    "incus",
    "stapes",
}
OSSICULOPLASTY_OFF_TOPIC_MARKERS = {
    "cochlear implant",
    "cochlear implantation",
    "vestibular schwannoma",
    "hearing preservation after cochlear",
    "single-sided deafness",
    "speech perception",
}
OSSICULOPLASTY_OFF_TOPIC_PENALTY = 0.18
OSSICULOPLASTY_MISSING_TOPIC_PENALTY = 0.12

OUT_OF_SCOPE_TERMS = {
    "rhinosinusitis",
    "sinusitis",
    "nasal polyp",
    "nasal polyps",
    "tonsillectomy",
    "adenoid hypertrophy",
    "laryngology",
    "dysphonia",
    "vocal cord",
    "glaucoma",
}

URL_PATTERN = re.compile(r"https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?")
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]\((https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?)\)")
DOUBLE_BRACKET_CITATION_PATTERN = re.compile(r"\[\[([^\]]+)\]\((https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?)\)")
CITATION_LIKE_BRACKET_PATTERN = re.compile(r"\[[^\]\n]{12,}\]")
AOM_UNDER_TWO_AGE_PATTERN = re.compile(
    r"\b(under|younger than|less than)\s+(?:2|two)\b|"
    r"\b6\s*(?:-|to|through)\s*23\s*months\b|"
    r"\b6\s*months\s*(?:to|through)\s*(?:2|two)\s*years\b",
    re.IGNORECASE,
)
AOM_ABSOLUTE_TREATMENT_PATTERN = re.compile(
    r"\ball\s+(?:children|infants|patients|cases)\b|"
    r"\b(always|never|regardless|mandate|mandates|contraindication|contraindications)\b|"
    r"\brequire(?:s|d)?\s+prompt\b|"
    r"\bprompt\s+(?:antibiotic\s+)?treatment\s+.*\bregardless\b",
    re.IGNORECASE,
)
AOM_PREMISE_CORRECTION_SKIP_PATTERN = re.compile(
    r"\b(false premise|overbroad|not all|does not require|do not require|"
    r"(?:unilateral\s+nonsevere|nonsevere\s+unilateral).*(?:observation|watchful waiting)|"
    r"(?:observation|watchful waiting).*(?:unilateral\s+nonsevere|nonsevere\s+unilateral))\b",
    re.IGNORECASE,
)

QUERY_EXPANSIONS = {
    "ssnhl": "sudden sensorineural hearing loss",
    "shl": "sudden hearing loss",
    "ome": "otitis media with effusion",
    "aom": "acute otitis media",
    "raom": "recurrent acute otitis media",
    "bppv": "benign paroxysmal positional vertigo",
    "vemp": "vestibular evoked myogenic potential",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "cwu": "canal wall up",
    "cwd": "canal wall down",
}

GUIDELINE_INTENT_TERMS = {
    "current",
    "guideline",
    "guidelines",
    "recommend",
    "recommendations",
    "indications",
    "management",
    "standard",
}

EVIDENCE_INTENT_TERMS = {
    "evidence",
    "systematic",
    "meta-analysis",
    "meta",
    "trial",
    "randomized",
    "versus",
    "vs",
    "compare",
    "comparison",
}

US_GUIDELINE_MARKERS = {
    "aao-hns",
    "aao-hnsf",
    "aap",
    "aap/aafp",
    "aafp",
    "american academy of otolaryngology",
    "american academy of otolaryngology-head and neck surgery",
    "american academy of pediatrics",
    "american academy of family physicians",
    "american college of radiology",
    "acr appropriateness criteria",
    "congress of neurological surgeons",
    "centers for disease control",
    "cdc",
}

GUIDELINE_PHRASE_MARKERS = {
    "clinical practice guideline",
    "practice guideline",
    "guideline update",
    "appropriateness criteria",
    "consensus statement",
}


def _quote_filter(value: str) -> str:
    return json.dumps(value)


JOURNAL_TOKEN_ALIASES = {
    "am": "american",
    "ann": "annals",
    "arch": "archives",
    "assoc": "association",
    "clin": "clinical",
    "int": "international",
    "intl": "international",
    "j": "journal",
    "laryngol": "laryngology",
    "med": "medicine",
    "neurotol": "neurotology",
    "otol": "otology",
    "otolaryngol": "otolaryngology",
    "otorhinolaryngol": "otorhinolaryngology",
    "pediatr": "pediatric",
    "surg": "surgery",
}

JOURNAL_TOKEN_STOPWORDS = {
    "a",
    "and",
    "de",
    "der",
    "for",
    "in",
    "of",
    "the",
}

JOURNAL_DISTINCTIVE_TOKENS = {
    "archives",
    "bmj",
    "cochrane",
    "jama",
    "lancet",
    "laryngoscope",
}


def journal_tokens(value: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", (value or "").lower()):
        token = JOURNAL_TOKEN_ALIASES.get(token, token)
        if token and token not in JOURNAL_TOKEN_STOPWORDS:
            tokens.add(token)
    return tokens


def journal_match_score(requested: str, actual: str) -> float:
    requested_tokens = journal_tokens(requested)
    actual_tokens = journal_tokens(actual)
    if not requested_tokens or not actual_tokens:
        return 0.0
    if requested_tokens <= actual_tokens:
        return 1.0
    if actual_tokens <= requested_tokens:
        return 1.0
    if requested_tokens == actual_tokens:
        return 1.0

    overlap = requested_tokens & actual_tokens
    recall = len(overlap) / len(requested_tokens)
    precision = len(overlap) / len(actual_tokens)
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def journal_matches(requested: str, actual: str) -> bool:
    requested_tokens = journal_tokens(requested)
    actual_tokens = journal_tokens(actual)
    if not requested_tokens or not actual_tokens:
        return False

    distinctive_requested = requested_tokens & JOURNAL_DISTINCTIVE_TOKENS
    if distinctive_requested and not distinctive_requested <= actual_tokens:
        return False
    distinctive_actual = actual_tokens & JOURNAL_DISTINCTIVE_TOKENS
    if distinctive_actual and not distinctive_actual <= requested_tokens:
        return False

    if requested_tokens <= actual_tokens:
        return True
    if actual_tokens <= requested_tokens:
        return True

    score = journal_match_score(requested, actual)
    if len(requested_tokens) <= 2:
        return score >= 0.95
    return score >= 0.85


def is_out_of_scope(message: str) -> bool:
    normalized = message.lower()
    return any(term in normalized for term in OUT_OF_SCOPE_TERMS)


def expand_query_variants(query: str, publication_types: list[str] | None = None) -> list[str]:
    normalized = query.lower()
    variants = [query]

    expanded_terms = []
    for short, long in QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(short)}\b", normalized) and long not in normalized:
            expanded_terms.append(long)
    if expanded_terms:
        variants.append(f"{query} {' '.join(expanded_terms)}")

    terms = set(re.findall(r"[a-z0-9-]+", normalized))
    pub_types = {pub_type.lower() for pub_type in publication_types or []}
    if terms & GUIDELINE_INTENT_TERMS or {"practice guideline", "guideline"} & pub_types:
        for suffix in [
            "clinical practice guideline consensus statement",
            "AAO-HNS guideline",
        ]:
            if suffix.lower() not in normalized:
                variants.append(f"{query} {suffix}")

    if ("otitis" in terms and "media" in terms) or "aom" in terms:
        for suffix in [
            "AAP AAFP acute otitis media guideline",
            "pediatric acute otitis media watchful waiting observation",
        ]:
            if suffix.lower() not in normalized:
                variants.append(f"{query} {suffix}")

    if terms & EVIDENCE_INTENT_TERMS or {"systematic review", "meta-analysis", "randomized controlled trial"} & pub_types:
        for suffix in [
            "systematic review meta-analysis randomized trial",
            "Cochrane review",
            "clinical practice guideline",
        ]:
            if suffix.lower() not in normalized:
                variants.append(f"{query} {suffix}")

    deduped = []
    seen = set()
    for variant in variants:
        compact = " ".join(variant.split())
        key = compact.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(compact)
    return deduped[:5]


def merge_ranked_hits(hit_sets: list[tuple[str, list]]) -> list:
    merged = {}
    for query_index, (variant, hits) in enumerate(hit_sets):
        query_weight = 1.0 / (query_index + 1)
        for rank, hit in enumerate(hits, start=1):
            pmid = hit.get("pmid") or hit.get("id") or f"{variant}:{rank}"
            contribution = query_weight / (60 + rank)
            if pmid not in merged:
                enriched = dict(hit)
                enriched["_rrf_score"] = 0.0
                enriched["_matched_queries"] = []
                merged[pmid] = enriched
            merged[pmid]["_rrf_score"] += contribution
            merged[pmid]["_matched_queries"].append(variant)
    return sorted(merged.values(), key=lambda hit: hit.get("_rrf_score", 0.0), reverse=True)


def query_intent(query: str) -> set[str]:
    normalized = query.lower()
    terms = set(re.findall(r"[a-z0-9-]+", normalized))
    intents = set()
    if terms & GUIDELINE_INTENT_TERMS:
        intents.add("guideline")
    if terms & EVIDENCE_INTENT_TERMS:
        intents.add("evidence")
    return intents


def publication_type_boost(publication_types: list[str]) -> float:
    normalized = {pub_type.lower() for pub_type in publication_types}
    boost = 0.0
    if "practice guideline" in normalized:
        boost = max(boost, 0.12)
    if "guideline" in normalized:
        boost = max(boost, 0.09)
    if "systematic review" in normalized:
        boost = max(boost, 0.07)
    if "meta-analysis" in normalized:
        boost = max(boost, 0.07)
    if "randomized controlled trial" in normalized:
        boost = max(boost, 0.05)
    return boost


def guideline_source_boost(title: str, abstract: str, journal: str) -> float:
    haystack = f"{title} {abstract[:1000]} {journal}".lower()
    boost = 0.0
    if any(marker in haystack for marker in US_GUIDELINE_MARKERS):
        boost += 0.16
    if any(marker in haystack for marker in GUIDELINE_PHRASE_MARKERS):
        boost += 0.06
    if "official journal of american academy of otolaryngology" in haystack:
        boost += 0.04
    if "clinical practice guideline" in title.lower() and "update" in title.lower():
        boost += 0.06
    return boost


def recency_boost_for_year(year: int | None, guideline_intent: bool) -> float:
    if not isinstance(year, int):
        return 0.0
    current_year = datetime.date.today().year
    window = 20 if guideline_intent else 15
    max_boost = 0.18 if guideline_intent else 0.08
    return max(0.0, 1 - min(current_year - year, window) / window) * max_boost


def topic_penalty_for_hit(query: str, title: str, abstract: str, mesh_terms: list[str]) -> float:
    query_terms = set(re.findall(r"[a-z0-9-]+", query.lower()))
    if not (query_terms & OSSICULOPLASTY_QUERY_TERMS):
        return 0.0

    haystack = f"{title} {abstract[:1000]} {' '.join(mesh_terms)}".lower()
    if any(marker in haystack for marker in OSSICULOPLASTY_ON_TOPIC_MARKERS):
        return 0.0
    if any(marker in haystack for marker in OSSICULOPLASTY_OFF_TOPIC_MARKERS):
        return OSSICULOPLASTY_OFF_TOPIC_PENALTY
    return OSSICULOPLASTY_MISSING_TOPIC_PENALTY


def fetch_papers(
    query: str,
    mesh_terms: list[str] | None = None,
    publication_types: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    journal: str | None = None,
    limit: int = FETCH_LIMIT,
) -> list:
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/search"
    filters = []
    if year_from is not None:
        filters.append(f"year >= {int(year_from)}")
    if year_to is not None:
        filters.append(f"year <= {int(year_to)}")
    if mesh_terms:
        quoted = ", ".join(_quote_filter(term) for term in mesh_terms if term)
        if quoted:
            filters.append(f"mesh_terms IN [{quoted}]")
    if publication_types:
        quoted = ", ".join(_quote_filter(pub_type) for pub_type in publication_types if pub_type)
        if quoted:
            filters.append(f"publication_type IN [{quoted}]")

    search_query = f"{query} {journal}".strip() if journal else query
    payload_obj = {
        "q": search_query,
        "limit": max(1, min(int(limit or FETCH_LIMIT), FETCH_LIMIT)),
        "attributesToRetrieve": [
            "pmid",
            "title",
            "abstract",
            "authors",
            "journal",
            "year",
            "pubdate",
            "mesh_terms",
            "publication_type",
            "url",
        ],
    }
    if MEILI_HYBRID_SEARCH:
        query_vector = hybrid_query_embedding(query)
        payload_obj["vector"] = query_vector
        payload_obj["hybrid"] = {
            "embedder": MEILI_HYBRID_EMBEDDER,
            "semanticRatio": MEILI_HYBRID_SEMANTIC_RATIO,
        }
    if filters:
        payload_obj["filter"] = filters

    payload = json.dumps(payload_obj).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {MEILI_SEARCH_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        hits = data.get("hits", [])
        if journal:
            matched_hits = []
            for hit in hits:
                score = journal_match_score(journal, hit.get("journal", ""))
                if journal_matches(journal, hit.get("journal", "")):
                    enriched = dict(hit)
                    enriched["_journal_match_score"] = round(score, 4)
                    matched_hits.append(enriched)
            hits = matched_hits
        mode = "hybrid" if MEILI_HYBRID_SEARCH else "bm25"
        journal_note = f" journal~={journal!r}" if journal else ""
        print(f"  [meilisearch:{mode}] '{search_query}' filters={filters or 'none'}{journal_note} → {len(hits)} hits")
        return hits


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / denom if denom else 0.0


class EmbeddingProvider:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self.provider == "gemini":
            return self._embed_gemini(texts, task_type)
        if self.provider == "openai":
            return self._embed_openai(texts)
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER={self.provider!r}")

    def _embed_gemini(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [embedding.values for embedding in response.embeddings]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        payload = json.dumps({
            "model": self.model,
            "input": texts,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        embeddings_by_index = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in embeddings_by_index]


class EmbeddingCache:
    def __init__(self, path: str, provider: EmbeddingProvider):
        self.path = Path(path)
        self.provider = provider
        self.disabled = DISABLE_EMBEDDING_CACHE
        if not self.disabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self.disabled:
            return self._embed_with_retry_uncached(texts, task_type)

        results: list[list[float] | None] = [None] * len(texts)
        missed_indexes = []
        missed_texts = []
        with self._connect() as conn:
            for index, text in enumerate(texts):
                cached = self._get(conn, text, task_type)
                if cached is None:
                    missed_indexes.append(index)
                    missed_texts.append(text)
                else:
                    results[index] = cached

        if missed_texts:
            fresh = self._embed_with_retry_uncached(missed_texts, task_type)
            if len(fresh) != len(missed_texts):
                raise RuntimeError(
                    f"Embedding provider returned {len(fresh)} vectors for {len(missed_texts)} texts"
                )
            with self._connect() as conn:
                for index, text, embedding in zip(missed_indexes, missed_texts, fresh):
                    results[index] = embedding
                    self._put(conn, text, task_type, embedding)
                conn.commit()

        return [embedding for embedding in results if embedding is not None]

    def _embed_with_retry_uncached(self, texts: list[str], task_type: str) -> list[list[float]]:
        for attempt in range(2):
            try:
                return self.provider.embed(texts, task_type)
            except Exception as e:
                if "429" not in str(e) or attempt == 1:
                    raise
                print(f"  [embedding rate limit] retrying after {EMBEDDING_RETRY_DELAY_SECONDS}s")
                time.sleep(EMBEDDING_RETRY_DELAY_SECONDS)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (provider, model, task_type, content_hash)
                )
                """
            )
            conn.commit()

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get(self, conn, text: str, task_type: str) -> list[float] | None:
        row = conn.execute(
            """
            SELECT embedding_json
            FROM embeddings
            WHERE provider = ? AND model = ? AND task_type = ? AND content_hash = ?
            """,
            (self.provider.provider, self.provider.model, task_type, self._content_hash(text)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, conn, text: str, task_type: str, embedding: list[float]) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
                (provider, model, task_type, content_hash, embedding_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.provider.provider,
                self.provider.model,
                task_type,
                self._content_hash(text),
                json.dumps(embedding),
                int(time.time()),
            ),
        )


embedding_provider = EmbeddingProvider(EMBEDDING_PROVIDER, EMBEDDING_MODEL)
embedding_cache = EmbeddingCache(EMBEDDING_CACHE_PATH, embedding_provider)
hybrid_embedding_provider = EmbeddingProvider(MEILI_HYBRID_PROVIDER, MEILI_HYBRID_MODEL)
hybrid_embedding_cache = EmbeddingCache(EMBEDDING_CACHE_PATH, hybrid_embedding_provider)


def hybrid_query_embedding(query: str) -> list[float]:
    return hybrid_embedding_cache.embed([query], "retrieval_query")[0]


def semantic_rerank(query: str, hits: list) -> list:
    """Re-rank hits by embedding cosine similarity; falls back to keyword order on error."""
    if not hits:
        return hits
    try:
        intents = query_intent(query)
        guideline_intent = "guideline" in intents
        snippets = [
            " ".join(
                part for part in [
                    h.get("title", ""),
                    h.get("abstract", "")[:1200],
                    " ".join(h.get("mesh_terms", [])[:8]),
                    " ".join(h.get("publication_type", [])[:5]),
                    h.get("journal", ""),
                ] if part
            )
            for h in hits
        ]
        query_embedding = embed_texts(
            texts=[query],
            task_type="retrieval_query",
        )[0]
        document_embeddings = embed_texts(
            texts=snippets,
            task_type="retrieval_document",
        )
        query_emb = query_embedding
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for hit, emb in zip(hits, document_embeddings):
            semantic_score = _cosine(query_emb, emb)
            raw_title = hit.get("title", "")
            raw_abstract = hit.get("abstract", "")
            journal = hit.get("journal", "")
            title = raw_title.lower()
            abstract = raw_abstract.lower()
            mesh_terms = [term.lower() for term in hit.get("mesh_terms", [])]
            publication_types = [term.lower() for term in hit.get("publication_type", [])]
            lexical_overlap = sum(term in title or term in abstract for term in query_terms)
            mesh_overlap = sum(term in mesh for term in query_terms for mesh in mesh_terms)
            publication_type_overlap = sum(
                term in pub_type for term in query_terms for pub_type in publication_types
            )
            year = hit.get("year")
            raw_recency_boost = recency_boost_for_year(year, guideline_intent)
            raw_hierarchy_boost = publication_type_boost(hit.get("publication_type", []))
            raw_source_boost = guideline_source_boost(raw_title, raw_abstract, journal) if guideline_intent else 0.0
            topic_boost_factor = (
                1.0 if semantic_score >= BOOST_TOPIC_GATE_THRESHOLD else BOOST_TOPIC_GATE_FACTOR
            )
            recency_boost = raw_recency_boost * topic_boost_factor
            hierarchy_boost = raw_hierarchy_boost * topic_boost_factor
            source_boost = raw_source_boost * topic_boost_factor
            lexical_component = 0.03 * lexical_overlap
            mesh_component = 0.02 * mesh_overlap
            publication_type_component = 0.04 * publication_type_overlap
            rrf_component = 2.0 * hit.get("_rrf_score", 0.0)
            topic_penalty = topic_penalty_for_hit(query, raw_title, raw_abstract, mesh_terms)
            score = (
                semantic_score
                + lexical_component
                + mesh_component
                + publication_type_component
                + rrf_component
                + hierarchy_boost
                + source_boost
                + recency_boost
                - topic_penalty
            )
            enriched = dict(hit)
            enriched["_score"] = round(score, 4)
            enriched["_semantic_score"] = round(semantic_score, 4)
            enriched["_hierarchy_boost"] = round(hierarchy_boost, 4)
            enriched["_raw_hierarchy_boost"] = round(raw_hierarchy_boost, 4)
            enriched["_source_boost"] = round(source_boost, 4)
            enriched["_raw_source_boost"] = round(raw_source_boost, 4)
            enriched["_recency_boost"] = round(recency_boost, 4)
            enriched["_raw_recency_boost"] = round(raw_recency_boost, 4)
            enriched["_topic_boost_factor"] = round(topic_boost_factor, 4)
            enriched["_lexical_overlap"] = lexical_overlap
            enriched["_mesh_overlap"] = mesh_overlap
            enriched["_publication_type_overlap"] = publication_type_overlap
            enriched["_lexical_component"] = round(lexical_component, 4)
            enriched["_mesh_component"] = round(mesh_component, 4)
            enriched["_publication_type_component"] = round(publication_type_component, 4)
            enriched["_rrf_component"] = round(rrf_component, 4)
            enriched["_topic_penalty"] = round(topic_penalty, 4)
            scored.append(enriched)

        scored.sort(key=lambda hit: hit["_score"], reverse=True)
        reranked = scored[:RERANK_LIMIT]
        print(f"  [rerank] top result: {reranked[0].get('title', '')[:60]!r}")
        return reranked
    except Exception as e:
        print(f"  [rerank error] {e} — using keyword order")
        return lexical_policy_rerank(query, hits)


def lexical_policy_rerank(query: str, hits: list) -> list:
    intents = query_intent(query)
    guideline_intent = "guideline" in intents
    query_terms = {term.lower() for term in query.split() if len(term) > 2}
    scored = []
    for rank, hit in enumerate(hits, start=1):
        raw_title = hit.get("title", "")
        raw_abstract = hit.get("abstract", "")
        journal = hit.get("journal", "")
        title = raw_title.lower()
        abstract = raw_abstract.lower()
        mesh_terms = [term.lower() for term in hit.get("mesh_terms", [])]
        publication_types = [term.lower() for term in hit.get("publication_type", [])]
        lexical_overlap = sum(term in title or term in abstract for term in query_terms)
        mesh_overlap = sum(term in mesh for term in query_terms for mesh in mesh_terms)
        publication_type_overlap = sum(
            term in pub_type for term in query_terms for pub_type in publication_types
        )
        hierarchy_boost = publication_type_boost(hit.get("publication_type", []))
        source_boost = guideline_source_boost(raw_title, raw_abstract, journal) if guideline_intent else 0.0
        recency_boost = recency_boost_for_year(hit.get("year"), guideline_intent)
        lexical_component = 0.03 * lexical_overlap
        mesh_component = 0.02 * mesh_overlap
        publication_type_component = 0.04 * publication_type_overlap
        rrf_component = 2.0 * hit.get("_rrf_score", 0.0)
        topic_penalty = topic_penalty_for_hit(query, raw_title, raw_abstract, mesh_terms)
        score = (
            (1.0 / (60 + rank))
            + lexical_component
            + mesh_component
            + publication_type_component
            + rrf_component
            + hierarchy_boost
            + source_boost
            + recency_boost
            - topic_penalty
        )
        enriched = dict(hit)
        enriched["_score"] = round(score, 4)
        enriched["_semantic_score"] = None
        enriched["_hierarchy_boost"] = round(hierarchy_boost, 4)
        enriched["_raw_hierarchy_boost"] = round(hierarchy_boost, 4)
        enriched["_source_boost"] = round(source_boost, 4)
        enriched["_raw_source_boost"] = round(source_boost, 4)
        enriched["_recency_boost"] = round(recency_boost, 4)
        enriched["_raw_recency_boost"] = round(recency_boost, 4)
        enriched["_topic_boost_factor"] = None
        enriched["_lexical_overlap"] = lexical_overlap
        enriched["_mesh_overlap"] = mesh_overlap
        enriched["_publication_type_overlap"] = publication_type_overlap
        enriched["_lexical_component"] = round(lexical_component, 4)
        enriched["_mesh_component"] = round(mesh_component, 4)
        enriched["_publication_type_component"] = round(publication_type_component, 4)
        enriched["_rrf_component"] = round(rrf_component, 4)
        enriched["_topic_penalty"] = round(topic_penalty, 4)
        scored.append(enriched)

    scored.sort(key=lambda hit: hit["_score"], reverse=True)
    return scored[:RERANK_LIMIT]


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    return embedding_cache.embed(texts, task_type)


def search_and_rerank(
    query: str,
    mesh_terms: list[str] | None = None,
    publication_types: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    journal: str | None = None,
    max_results: int | None = None,
    seen_pmids: set[str] | None = None,
    skip_rerank: bool = False,
) -> dict:
    requested = max(1, min(int(max_results or 10), RERANK_LIMIT))
    query_variants = expand_query_variants(query, publication_types)
    hit_sets = []
    recovery_notes = []
    for variant in query_variants:
        variant_hits = fetch_papers(
            query=variant,
            mesh_terms=mesh_terms,
            publication_types=publication_types,
            year_from=year_from,
            year_to=year_to,
            journal=journal,
        )
        hit_sets.append((variant, variant_hits))

    filtered_hit_count = sum(len(hits) for _, hits in hit_sets)
    if filtered_hit_count < MIN_FILTERED_HITS and (mesh_terms or publication_types or journal):
        relaxed_parts = []
        relaxed_mesh_terms = mesh_terms
        relaxed_publication_types = publication_types
        relaxed_journal = journal
        if journal:
            relaxed_journal = None
            relaxed_parts.append("journal")
        elif mesh_terms:
            relaxed_mesh_terms = None
            relaxed_parts.append("mesh_terms")
        elif publication_types:
            relaxed_publication_types = None
            relaxed_parts.append("publication_types")

        if relaxed_parts:
            recovery_notes.append(
                f"Only {filtered_hit_count} filtered hit(s); retried without {', '.join(relaxed_parts)}."
            )
            print(f"  [recovery] {recovery_notes[-1]}")
            for variant in query_variants:
                relaxed_hits = fetch_papers(
                    query=variant,
                    mesh_terms=relaxed_mesh_terms,
                    publication_types=relaxed_publication_types,
                    year_from=year_from,
                    year_to=year_to,
                    journal=relaxed_journal,
                )
                hit_sets.append((variant, relaxed_hits))

    hits = merge_ranked_hits(hit_sets)
    if seen_pmids:
        before = len(hits)
        hits = [h for h in hits if h.get("pmid") not in seen_pmids]
        removed = before - len(hits)
        if removed:
            print(f"  [dedupe] dropped {removed} previously returned PMID(s)")
    if skip_rerank:
        hits = hits[:RERANK_LIMIT]
    else:
        hits = semantic_rerank(query, hits)
    hits = hits[:requested]
    papers = []
    for h in hits:
        if seen_pmids is not None and h.get("pmid"):
            seen_pmids.add(h["pmid"])
        papers.append({
            "pmid": h.get("pmid"),
            "title": h.get("title", ""),
            "authors": h.get("authors", []),
            "journal": h.get("journal", ""),
            "year": h.get("year"),
            "pubdate": h.get("pubdate", ""),
            "mesh_terms": h.get("mesh_terms", []),
            "publication_type": h.get("publication_type", []),
            "url": h.get("url", ""),
            "abstract": h.get("abstract", ""),
            "score": h.get("_score"),
            "semantic_score": h.get("_semantic_score"),
            "rrf_score": h.get("_rrf_score"),
            "rrf_component": h.get("_rrf_component"),
            "lexical_overlap": h.get("_lexical_overlap"),
            "mesh_overlap": h.get("_mesh_overlap"),
            "publication_type_overlap": h.get("_publication_type_overlap"),
            "lexical_component": h.get("_lexical_component"),
            "mesh_component": h.get("_mesh_component"),
            "publication_type_component": h.get("_publication_type_component"),
            "hierarchy_boost": h.get("_hierarchy_boost"),
            "raw_hierarchy_boost": h.get("_raw_hierarchy_boost"),
            "source_boost": h.get("_source_boost"),
            "raw_source_boost": h.get("_raw_source_boost"),
            "recency_boost": h.get("_recency_boost"),
            "raw_recency_boost": h.get("_raw_recency_boost"),
            "topic_boost_factor": h.get("_topic_boost_factor"),
            "topic_penalty": h.get("_topic_penalty"),
            "journal_match_score": h.get("_journal_match_score"),
        })

    return {
        "query": query,
        "filters": {
            "mesh_terms": mesh_terms or [],
            "publication_types": publication_types or [],
            "year_from": year_from,
            "year_to": year_to,
            "journal": journal,
        },
        "query_variants": query_variants,
        "recovery_notes": recovery_notes,
        "count": len(papers),
        "papers": papers,
    }


def normalize_citation_markdown(reply: str) -> str:
    """Repair harmless citation Markdown glitches before validation."""
    return DOUBLE_BRACKET_CITATION_PATTERN.sub(r"[\1](\2)", reply or "")


def filter_unretrieved_citations(reply: str, retrieved_urls: set[str]) -> tuple[str, list[str]]:
    cited_urls = set(URL_PATTERN.findall(reply or ""))
    missing = sorted(url for url in cited_urls if url.rstrip("/") + "/" not in retrieved_urls and url.rstrip("/") not in retrieved_urls)
    if not missing:
        return reply, []

    warning = (
        "\n\nCitation check: the answer contained citation URL(s) that were not returned "
        "by the literature search and should be independently verified: "
        + ", ".join(missing)
    )
    return (reply or "") + warning, missing


def extracted_citations(reply: str) -> list[dict]:
    seen = set()
    result = []
    for label, url in CITATION_PATTERN.findall(reply or ""):
        normalized = url.rstrip("/") + "/"
        if normalized not in seen:
            seen.add(normalized)
            result.append({"label": label.lstrip("[").strip(), "url": normalized})
    return result


def retrieved_source_markdown_link(
    retrieved_sources: dict[str, dict],
    title_markers: list[str],
) -> str:
    for url, source in retrieved_sources.items():
        title = (source.get("title") or "").lower()
        if all(marker in title for marker in title_markers):
            year = source.get("year")
            label = source.get("title") or "Retrieved guideline"
            if year:
                label = f"{label} ({year})"
            return f"[{label}]({url})"
    return ""


def detects_aom_under_two_overstatement(reply: str) -> bool:
    normalized = reply or ""
    lower = normalized.lower()
    if "otitis media" not in lower and "aom" not in lower:
        return False
    if "watchful waiting" not in lower and "observation" not in lower:
        return False
    if not AOM_UNDER_TWO_AGE_PATTERN.search(normalized):
        return False
    if AOM_PREMISE_CORRECTION_SKIP_PATTERN.search(normalized):
        return False
    return bool(AOM_ABSOLUTE_TREATMENT_PATTERN.search(normalized))


def apply_clinical_contradiction_guardrails(
    reply: str,
    retrieved_sources: dict[str, dict],
) -> tuple[str, list[str]]:
    """Patch narrow, high-confidence clinical overstatements before returning a reply."""
    warnings = []
    if not detects_aom_under_two_overstatement(reply):
        return reply, warnings

    aap_link = retrieved_source_markdown_link(
        retrieved_sources,
        ["diagnosis", "management", "acute otitis media"],
    )
    citation = f" {aap_link}" if aap_link else ""
    correction = (
        "\n\nClinical accuracy check: the under-2 rule is overbroad. "
        "U.S. AAP/AAFP guidance allows observation with close follow-up for children "
        "6-23 months with nonsevere unilateral AOM; prompt antibiotics are recommended "
        "for severe AOM, otorrhea, or bilateral AOM in this age group."
        f"{citation}"
    )
    warnings.append("Corrected overbroad AOM watchful-waiting claim for children under 2.")
    return (reply or "") + correction, warnings


def is_transient_model_error(error: Exception) -> bool:
    msg = str(error).lower()
    transient_markers = (
        "500",
        "internal",
        "503",
        "unavailable",
        "resource_exhausted",
        "429",
        "rate limit",
        "timeout",
        "temporarily",
        "connection reset",
    )
    return any(marker in msg for marker in transient_markers)


def generate_content_with_retry(*, phase: str, model: str, contents, config):
    attempts = max(1, MODEL_RETRY_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if not is_transient_model_error(e) or attempt == attempts:
                raise
            delay = MODEL_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            delay += random.uniform(0, 0.25)
            print(
                f"  [model retry] phase={phase} attempt={attempt}/{attempts} "
                f"after transient error: {str(e)[:160]}"
            )
            time.sleep(delay)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")


def normalize_conversation_title(text: str) -> str:
    title = re.sub(r"\s+", " ", (text or "").strip())
    if not title:
        return "Untitled"
    if len(title) > 60:
        return title[:59] + "…"
    return title


def validate_user_id(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        raise ValueError("user_id is required")
    if len(user_id) > 128:
        raise ValueError("user_id is too long")
    return user_id


def conversation_db_path() -> Path:
    return Path(CONVERSATION_DB_PATH)


def get_conversation_db() -> sqlite3.Connection:
    path = conversation_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    init_conversation_db(db)
    return db


def init_conversation_db(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
            ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
            ON messages(conversation_id, created_at);
    """)


def upsert_user(db: sqlite3.Connection, user_id: str) -> None:
    now = utc_now_iso()
    db.execute(
        """
        INSERT INTO users (id, created_at, last_seen_at)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET last_seen_at = excluded.last_seen_at
        """,
        (user_id, now, now),
    )


def create_conversation(db: sqlite3.Connection, user_id: str, first_message: str) -> str:
    conversation_id = str(uuid.uuid4())
    now = utc_now_iso()
    db.execute(
        """
        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conversation_id, user_id, normalize_conversation_title(first_message), now, now),
    )
    return conversation_id


def get_conversation(db: sqlite3.Connection, user_id: str, conversation_id: str):
    return db.execute(
        """
        SELECT id, user_id, title, created_at, updated_at
        FROM conversations
        WHERE id = ? AND user_id = ?
        """,
        (conversation_id, user_id),
    ).fetchone()


def list_conversations_for_user(db: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = db.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def load_messages_for_conversation(db: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at, rowid
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def load_context_messages(db: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = db.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (conversation_id, CONVERSATION_CONTEXT_MESSAGE_LIMIT),
    ).fetchall()
    return [dict(row) for row in reversed(rows)]


def append_conversation_message(
    db: sqlite3.Connection,
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    now = utc_now_iso()
    db.execute(
        """
        INSERT INTO messages (id, conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), conversation_id, role, content, now),
    )
    db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )


def delete_conversation_for_user(db: sqlite3.Connection, user_id: str, conversation_id: str) -> bool:
    result = db.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    return result.rowcount > 0


def build_model_contents(messages: list[dict]) -> list:
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    return contents


def run_agent(contents: list, include_trace: bool = False) -> dict:
    seen_pmids = set()
    retrieved_urls = set()
    retrieved_sources = {}
    skip_rerank_for_request = False
    trace = {
        "out_of_scope": False,
        "tool_calls": [],
        "forced_final": False,
        "rerank_disabled": False,
        "citation_repair_attempted": False,
        "clinical_guardrail_warnings": [],
    }

    for turn in range(MAX_TOOL_TURNS):
        response = generate_content_with_retry(
            phase=f"agent_turn_{turn + 1}",
            model="gemma-4-31b-it",
            contents=contents,
            config=AGENT_CONFIG,
        )

        candidate = response.candidates[0]
        function_calls = [p for p in candidate.content.parts if p.function_call]

        if not function_calls:
            guarded_text, clinical_warnings = apply_clinical_contradiction_guardrails(
                response.text,
                retrieved_sources,
            )
            reply, missing_urls, citations, format_warnings, repair_attempted = enforce_citation_urls(
                guarded_text,
                retrieved_urls,
                retrieved_sources,
            )
            trace["citation_repair_attempted"] = repair_attempted
            trace["clinical_guardrail_warnings"].extend(clinical_warnings)
            response_obj = {
                "reply": reply,
                "citation_warnings": missing_urls,
                "citation_format_warnings": format_warnings,
                "clinical_guardrail_warnings": clinical_warnings,
                "citations": citations,
            }
            if include_trace:
                response_obj["trace"] = trace
            return response_obj

        print(f"  [agent turn {turn + 1}] {len(function_calls)} search(es)")
        contents.append(types.Content(role="model", parts=function_calls))

        tool_parts = []
        for part in function_calls:
            fc = part.function_call
            query = fc.args.get("query", "")
            mesh_terms = fc.args.get("mesh_terms") or []
            publication_types = fc.args.get("publication_types") or []
            year_from = fc.args.get("year_from")
            year_to = fc.args.get("year_to")
            journal = fc.args.get("journal")
            max_results = fc.args.get("max_results")
            print(
                f"    → query={query!r} mesh_terms={mesh_terms!r} "
                f"publication_types={publication_types!r} "
                f"year_from={year_from!r} year_to={year_to!r} journal={journal!r}"
            )
            try:
                result = search_and_rerank(
                    query=query,
                    mesh_terms=mesh_terms,
                    publication_types=publication_types,
                    year_from=year_from,
                    year_to=year_to,
                    journal=journal,
                    max_results=max_results,
                    seen_pmids=seen_pmids,
                    skip_rerank=skip_rerank_for_request,
                )
            except Exception as e:
                if "429" not in str(e):
                    raise
                print("  [rerank disabled] embedding quota hit; using keyword order for this request")
                skip_rerank_for_request = True
                trace["rerank_disabled"] = True
                result = search_and_rerank(
                    query=query,
                    mesh_terms=mesh_terms,
                    publication_types=publication_types,
                    year_from=year_from,
                    year_to=year_to,
                    journal=journal,
                    max_results=max_results,
                    seen_pmids=seen_pmids,
                    skip_rerank=True,
                )
            for paper in result.get("papers", []):
                url = paper.get("url")
                if url:
                    normalized_url = url.rstrip("/") + "/"
                    retrieved_urls.add(normalized_url)
                    retrieved_sources[normalized_url] = {
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                    }
            trace["tool_calls"].append({
                "turn": turn + 1,
                "query": query,
                "filters": result.get("filters", {}),
                "query_variants": result.get("query_variants", []),
                "recovery_notes": result.get("recovery_notes", []),
                "count": result.get("count", 0),
                "papers": [
                    {
                        "pmid": paper.get("pmid"),
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "publication_type": paper.get("publication_type", []),
                        "score": paper.get("score"),
                        "semantic_score": paper.get("semantic_score"),
                        "topic_penalty": paper.get("topic_penalty"),
                        "url": paper.get("url"),
                    }
                    for paper in result.get("papers", [])
                ],
            })
            tool_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={"result": result},
                )
            ))

        contents.append(types.Content(role="user", parts=tool_parts))

    response = generate_content_with_retry(
        phase="forced_final",
        model="gemma-4-31b-it",
        contents=contents,
        config=FINAL_CONFIG,
    )
    guarded_text, clinical_warnings = apply_clinical_contradiction_guardrails(
        response.text,
        retrieved_sources,
    )
    reply, missing_urls, citations, format_warnings, repair_attempted = enforce_citation_urls(
        guarded_text,
        retrieved_urls,
        retrieved_sources,
    )
    trace["forced_final"] = True
    trace["citation_repair_attempted"] = repair_attempted
    trace["clinical_guardrail_warnings"].extend(clinical_warnings)
    response_obj = {
        "reply": reply,
        "citation_warnings": missing_urls,
        "citation_format_warnings": format_warnings,
        "clinical_guardrail_warnings": clinical_warnings,
        "citations": citations,
    }
    if include_trace:
        response_obj["trace"] = trace
    return response_obj


def has_citation_like_brackets(reply: str) -> bool:
    """Detect title-style bracket citations that are missing Markdown URLs."""
    without_valid_links = CITATION_PATTERN.sub("", reply or "")
    return bool(CITATION_LIKE_BRACKET_PATTERN.search(without_valid_links))


def repair_citation_markdown(reply: str, retrieved_sources: dict[str, dict]) -> str:
    sources = [
        {
            "title": source.get("title", ""),
            "year": source.get("year"),
            "url": url,
        }
        for url, source in sorted(retrieved_sources.items())
    ]
    if not sources:
        return reply

    prompt = (
        "Retrieved source list:\n"
        f"{json.dumps(sources, indent=2)}\n\n"
        "Answer to repair:\n"
        f"{reply or ''}"
    )
    response = generate_content_with_retry(
        phase="citation_repair",
        model="gemma-4-31b-it",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=CITATION_REPAIR_CONFIG,
    )
    return response.text or reply


def prepare_citation_response(reply: str, retrieved_urls: set[str]) -> tuple[str, list[str], list[dict], list[str]]:
    normalized_reply = normalize_citation_markdown(reply)
    checked_reply, missing_urls = filter_unretrieved_citations(normalized_reply, retrieved_urls)
    citations = extracted_citations(checked_reply)
    format_warnings = []
    if retrieved_urls and checked_reply.strip() and not citations:
        format_warnings.append(
            "No valid PubMed citation links were parsed even though the literature search returned papers."
        )
    return checked_reply, missing_urls, citations, format_warnings


def enforce_citation_urls(
    reply: str,
    retrieved_urls: set[str],
    retrieved_sources: dict[str, dict],
) -> tuple[str, list[str], list[dict], list[str], bool]:
    checked_reply, missing_urls, citations, format_warnings = prepare_citation_response(
        reply,
        retrieved_urls,
    )
    repair_attempted = False
    if format_warnings and has_citation_like_brackets(checked_reply):
        repair_attempted = True
        try:
            repaired_reply = repair_citation_markdown(checked_reply, retrieved_sources)
        except Exception as e:
            format_warnings.append(f"Citation URL repair failed: {str(e)[:120]}")
            return checked_reply, missing_urls, citations, format_warnings, repair_attempted
        checked_reply, missing_urls, citations, format_warnings = prepare_citation_response(
            repaired_reply,
            retrieved_urls,
        )
        if format_warnings:
            format_warnings.append("Citation URL repair was attempted but did not produce valid PubMed links.")
    return checked_reply, missing_urls, citations, format_warnings, repair_attempted


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(SEARCH_APP_DIR, "chat.html")


@app.route("/search")
def search_ui():
    return send_from_directory(SEARCH_APP_DIR, "index.html")


@app.route("/api/conversations", methods=["GET", "OPTIONS"])
def conversations_api():
    if request.method == "OPTIONS":
        return "", 204
    try:
        user_id = validate_user_id(request.args.get("user_id", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    with contextlib.closing(get_conversation_db()) as db:
        upsert_user(db, user_id)
        conversations = list_conversations_for_user(db, user_id)
        db.commit()
    return jsonify({"conversations": conversations})


@app.route("/api/conversations/<conversation_id>", methods=["GET", "DELETE", "OPTIONS"])
def conversation_api(conversation_id):
    if request.method == "OPTIONS":
        return "", 204
    try:
        user_id = validate_user_id(request.args.get("user_id", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    with contextlib.closing(get_conversation_db()) as db:
        upsert_user(db, user_id)
        conversation = get_conversation(db, user_id, conversation_id)
        if not conversation:
            db.commit()
            return jsonify({"error": "Conversation not found"}), 404
        if request.method == "DELETE":
            delete_conversation_for_user(db, user_id, conversation_id)
            db.commit()
            return "", 204
        messages = load_messages_for_conversation(db, conversation_id)
        db.commit()
    return jsonify({"conversation": dict(conversation), "messages": messages})


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json() or {}
    include_trace = bool(data.get("trace"))
    message = (data.get("message") or "").strip()
    conversation_id = (data.get("conversation_id") or "").strip() or None
    try:
        user_id = validate_user_id(data.get("user_id", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not message:
        return jsonify({"error": "message is required"}), 400

    print(f"\n[user] {message[:80]}{'...' if len(message) > 80 else ''}")

    try:
        with contextlib.closing(get_conversation_db()) as db:
            upsert_user(db, user_id)
            if conversation_id:
                if not get_conversation(db, user_id, conversation_id):
                    db.commit()
                    return jsonify({"error": "Conversation not found"}), 404
            else:
                conversation_id = create_conversation(db, user_id, message)
            append_conversation_message(db, conversation_id, "user", message)
            context_messages = load_context_messages(db, conversation_id)
            db.commit()

        if is_out_of_scope(message):
            response_obj = {
                "conversation_id": conversation_id,
                "reply": (
                    "That question is outside this otology-focused literature index. "
                    "I can help with otology, hearing, vestibular, ear surgery, and closely related neurotology questions."
                ),
            }
            if include_trace:
                response_obj["trace"] = {"out_of_scope": True, "tool_calls": []}
        else:
            response_obj = run_agent(build_model_contents(context_messages), include_trace=include_trace)
            response_obj["conversation_id"] = conversation_id

        with contextlib.closing(get_conversation_db()) as db:
            if get_conversation(db, user_id, conversation_id):
                append_conversation_message(db, conversation_id, "assistant", response_obj["reply"])
            db.commit()
        return jsonify(response_obj)

    except urllib.error.URLError as e:
        print(f"[error] Meilisearch: {e}")
        return jsonify({"error": f"Search index unreachable: {e}"}), 502
    except Exception as e:
        print(f"[error] {e}")
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return jsonify({"error": "Rate limit reached — please wait a moment and try again."}), 429
        return jsonify({"error": f"Model error: {msg[:200]}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Otology Research Agent → http://localhost:{port}")
    print(f"Literature search      → http://localhost:{port}/search\n")
    print(
        "Retrieval config       → "
        f"Meili {'hybrid' if MEILI_HYBRID_SEARCH else 'BM25'} "
        f"({MEILI_HYBRID_EMBEDDER}, semanticRatio={MEILI_HYBRID_SEMANTIC_RATIO}); "
        f"rerank embeddings={EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}\n"
    )
    app.run(debug=True, port=port)
