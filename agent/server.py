#!/usr/bin/env python3
"""Flask backend for the Otology Literature Research Agent."""

import json
import math
import os
import urllib.request
import urllib.error

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
Use precise clinical terminology. Do not provide personal medical advice.
Your only source of truth is the literature retrieved with the tool. If evidence is thin or absent, say so clearly.
Cite each source you rely on as a markdown hyperlink: [Title (Year)](URL).

You have access to a search_papers tool over a PubMed otology index.
Before answering, call the tool one or more times with focused queries.
When making a tool call, output only the function call — no surrounding text or explanation.
For complex questions, decompose the question and search each angle separately.
Use MeSH terms, year filters, and journal filters when they improve retrieval.
Use publication-type filters when the user asks for guidelines, recommendations, systematic reviews, meta-analyses, or randomized trials.
Search broadly first, then narrow.

Choose evidence based on the user's intent:
- For current indications, recommendations, guidelines, or standard-of-care questions, prioritize recent clinical practice guidelines, consensus statements, and high-level reviews before individual studies.
- For treatment-evidence questions, rank evidence by study design: systematic reviews/meta-analyses, randomized trials, prospective comparative studies, then retrospective cohorts/case series.
- Use lower-level studies only when higher-level evidence is absent, conflicting, or too sparse.
- Do not present uncontrolled, single-center, or older cohort studies as strong evidence when higher-level evidence is weak or uncertain.

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
1. Current guideline or consensus position and year.
2. Main indications or action statements.
3. Important situations where the guideline recommends against intervention or suggests observation first.
4. Important uncertainty, exceptions, or at-risk subgroups.

When summarizing evidence, include study design, sample size, and evidence quality when available.
Keep the answer under 300 words unless the user asks for more depth.
Do not use tables unless the user explicitly asks."""

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
                    description="Optional exact journal name filter",
                ),
                "max_results": types.Schema(
                    type="INTEGER",
                    description="Optional number of reranked papers to return (default 8, max 12)",
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
FINAL_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)

MAX_TOOL_TURNS = 5
FETCH_LIMIT = 60   # candidates pulled from Meilisearch
RERANK_LIMIT = 12  # returned to the model after semantic re-ranking


def _quote_filter(value: str) -> str:
    return json.dumps(value)


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
    if journal:
        filters.append(f"journal = {_quote_filter(journal)}")
    if mesh_terms:
        quoted = ", ".join(_quote_filter(term) for term in mesh_terms if term)
        if quoted:
            filters.append(f"mesh_terms IN [{quoted}]")
    if publication_types:
        quoted = ", ".join(_quote_filter(pub_type) for pub_type in publication_types if pub_type)
        if quoted:
            filters.append(f"publication_type IN [{quoted}]")

    payload_obj = {
        "q": query,
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
        print(f"  [meilisearch] '{query}' filters={filters or 'none'} → {len(hits)} hits")
        return hits


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / denom if denom else 0.0


def semantic_rerank(query: str, hits: list) -> list:
    """Re-rank hits by embedding cosine similarity; falls back to keyword order on error."""
    if not hits:
        return hits
    try:
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
        embeddings = client.models.embed_content(
            model="text-embedding-004",
            contents=[query] + snippets,
        ).embeddings
        query_emb = embeddings[0].values
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        current_year = 2026
        for hit, emb in zip(hits, embeddings[1:]):
            semantic_score = _cosine(query_emb, emb.values)
            title = hit.get("title", "").lower()
            abstract = hit.get("abstract", "").lower()
            mesh_terms = [term.lower() for term in hit.get("mesh_terms", [])]
            publication_types = [term.lower() for term in hit.get("publication_type", [])]
            lexical_overlap = sum(term in title or term in abstract for term in query_terms)
            mesh_overlap = sum(term in mesh for term in query_terms for mesh in mesh_terms)
            publication_type_overlap = sum(
                term in pub_type for term in query_terms for pub_type in publication_types
            )
            year = hit.get("year")
            recency_boost = 0.0
            if isinstance(year, int):
                recency_boost = max(0.0, 1 - min(current_year - year, 15) / 15) * 0.08
            score = (
                semantic_score
                + (0.03 * lexical_overlap)
                + (0.02 * mesh_overlap)
                + (0.04 * publication_type_overlap)
                + recency_boost
            )
            enriched = dict(hit)
            enriched["_score"] = round(score, 4)
            enriched["_semantic_score"] = round(semantic_score, 4)
            scored.append(enriched)

        scored.sort(key=lambda hit: hit["_score"], reverse=True)
        reranked = scored[:RERANK_LIMIT]
        print(f"  [rerank] top result: {reranked[0].get('title', '')[:60]!r}")
        return reranked
    except Exception as e:
        print(f"  [rerank error] {e} — using keyword order")
        fallback = []
        for hit in hits[:RERANK_LIMIT]:
            enriched = dict(hit)
            enriched["_score"] = None
            enriched["_semantic_score"] = None
            fallback.append(enriched)
        return fallback


def search_and_rerank(
    query: str,
    mesh_terms: list[str] | None = None,
    publication_types: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    journal: str | None = None,
    max_results: int | None = None,
) -> dict:
    requested = max(1, min(int(max_results or 8), RERANK_LIMIT))
    hits = fetch_papers(
        query=query,
        mesh_terms=mesh_terms,
        publication_types=publication_types,
        year_from=year_from,
        year_to=year_to,
        journal=journal,
    )
    hits = semantic_rerank(query, hits)
    hits = hits[:requested]
    papers = []
    for h in hits:
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
        "count": len(papers),
        "papers": papers,
    }


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(SEARCH_APP_DIR, "chat.html")


@app.route("/search")
def search_ui():
    return send_from_directory(SEARCH_APP_DIR, "index.html")


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json()
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    last_message = messages[-1]["content"]
    print(f"\n[user] {last_message[:80]}{'...' if len(last_message) > 80 else ''}")

    # Build contents from full conversation history
    contents = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=last_message)]))

    try:
        # Agentic tool-calling loop
        for turn in range(MAX_TOOL_TURNS):
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=contents,
                config=AGENT_CONFIG,
            )

            candidate = response.candidates[0]
            function_calls = [p for p in candidate.content.parts if p.function_call]

            # No tool calls → model is done, return the text answer
            if not function_calls:
                return jsonify({"reply": response.text})

            print(f"  [agent turn {turn + 1}] {len(function_calls)} search(es)")
            # Strip any text preamble emitted alongside tool calls to keep context lean
            tool_only_content = types.Content(role="model", parts=function_calls)
            contents.append(tool_only_content)

            # Execute each tool call and collect responses
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
                result = search_and_rerank(
                    query=query,
                    mesh_terms=mesh_terms,
                    publication_types=publication_types,
                    year_from=year_from,
                    year_to=year_to,
                    journal=journal,
                    max_results=max_results,
                )
                tool_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={"result": result},
                    )
                ))

            contents.append(types.Content(role="user", parts=tool_parts))

        # Reached the turn cap — force a final answer without tools
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=contents,
            config=FINAL_CONFIG,
        )
        return jsonify({"reply": response.text})

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
    app.run(debug=True, port=port)
