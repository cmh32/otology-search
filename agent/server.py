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
Use precise anatomical and clinical terminology — your user is a trained clinician.
When summarizing findings, note study design, sample size, and level of evidence when available.
Explicitly flag conflicting or limited evidence.
Cite each paper you draw on with its title, year, and PubMed link as a markdown hyperlink: [Title](URL).
Do not provide personal medical advice. 

You have access to a search_papers tool that queries a PubMed otology database.
Before answering, call it one or more times with focused keyword queries to retrieve relevant evidence.
For complex questions, decompose them and search each angle separately.
Only synthesize your final answer after gathering sufficient literature.
Keep the answer to 300 words or less. Be terse. The user may prompt you to dig deeper--that is fine.
Do not use tables unless the user explicitly asks.
Do not lean on any prior knoweldge you have — the literature you retrieve is your only source of truth. If you don't find evidence, say so clearly."""

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
FETCH_LIMIT = 15   # candidates pulled from Meilisearch
RERANK_LIMIT = 6   # returned to the model after semantic re-ranking


def fetch_papers(query: str) -> list:
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/search"
    payload = json.dumps({
        "q": query,
        "limit": FETCH_LIMIT,
        "attributesToRetrieve": ["title", "abstract", "authors", "journal", "year", "url"],
    }).encode()
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
        print(f"  [meilisearch] '{query}' → {len(hits)} hits")
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
        snippets = [f"{h.get('title', '')} {h.get('abstract', '')[:300]}" for h in hits]
        embeddings = client.models.embed_content(
            model="text-embedding-004",
            contents=[query] + snippets,
        ).embeddings
        query_emb = embeddings[0].values
        scored = sorted(
            zip(hits, embeddings[1:]),
            key=lambda pair: _cosine(query_emb, pair[1].values),
            reverse=True,
        )
        reranked = [h for h, _ in scored[:RERANK_LIMIT]]
        print(f"  [rerank] top result: {reranked[0].get('title', '')[:60]!r}")
        return reranked
    except Exception as e:
        print(f"  [rerank error] {e} — using keyword order")
        return hits[:RERANK_LIMIT]


def search_and_rerank(query: str) -> str:
    hits = fetch_papers(query)
    hits = semantic_rerank(query, hits)
    return format_papers(hits)


def format_papers(hits: list) -> str:
    if not hits:
        return "No papers found."
    parts = []
    for h in hits:
        authors = ", ".join(h.get("authors", [])[:3])
        if len(h.get("authors", [])) > 3:
            authors += " et al."
        parts.append(
            f"Title: {h.get('title', '')}\n"
            f"Authors: {authors}\n"
            f"Journal: {h.get('journal', '')} ({h.get('year', 'n.d.')})\n"
            f"URL: {h.get('url', '')}\n"
            f"Abstract: {h.get('abstract', '')[:500]}"
        )
    return "\n\n---\n\n".join(parts)


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
            contents.append(candidate.content)

            # Execute each tool call and collect responses
            tool_parts = []
            for part in function_calls:
                fc = part.function_call
                query = fc.args.get("query", "")
                print(f"    → {query!r}")
                result = search_and_rerank(query)
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
