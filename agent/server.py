#!/usr/bin/env python3
"""Flask backend for the Otology Literature Research Agent."""

import json
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
You will be given the user's question along with relevant papers retrieved from a PubMed database.
Use precise anatomical and clinical terminology — your user is a trained clinician.
When summarizing findings, note study design, sample size, and level of evidence when available.
Explicitly flag conflicting or limited evidence.
Cite each paper you draw on with its title, year, and PubMed link as a markdown hyperlink: [Title](URL).
Do not provide personal medical advice."""

CHAT_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)


def fetch_papers(query: str) -> list:
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/search"
    payload = json.dumps({
        "q": query,
        "limit": 8,
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

    # Build conversation history for multi-turn context
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    try:
        # Search Meilisearch once, then make a single Gemini call
        hits = fetch_papers(last_message)
        augmented = (
            f"Question: {last_message}\n\n"
            f"Relevant papers from the database:\n\n{format_papers(hits)}"
        )

        chat_session = client.chats.create(
            model="gemma-4-31b-it",
            history=history,
            config=CHAT_CONFIG,
        )
        response = chat_session.send_message(augmented)
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
