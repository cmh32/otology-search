# Otology Literature Research Agent

A clinical literature assistant for an APRN specializing in otology. It answers evidence-based questions about ear disease, hearing loss, vestibular disorders, and ear surgery by retrieving from a curated PubMed index and synthesizing results with a language model.

## What it does

- Retrieves relevant papers from a Meilisearch index of ~16,500 PubMed otology articles
- Uses Meilisearch native hybrid search by default with precomputed OpenAI document vectors
- Runs up to five agentic tool-call turns to decompose multi-part questions
- Returns a synthesized answer with inline citations to PubMed URLs
- Flags any citation URLs that were not actually retrieved (hallucination guard)
- Attempts one citation-format repair when the answer cites retrieved source titles without PubMed Markdown links
- Applies narrow clinical/citation guardrails for known high-risk guideline overstatements and citation-support gaps
- Retries transient model-provider failures such as intermittent Google GenAI `500 INTERNAL` responses
- Rejects out-of-scope questions (rhinology, laryngology, ophthalmology) before searching

## Architecture

```
User question
     │
     ▼
Out-of-scope check (keyword list, no LLM call)
     │
     ▼
Gemma-4-31b-it  ←──────────────────────────────────────┐
(via Google GenAI API, native function-calling)         │
     │ search_papers tool call                          │
     ▼                                                  │
Query expansion (abbreviations + guideline/evidence     │
suffix variants, up to 5 variants per call)             │
     │                                                  │
     ▼                                                  │
Meilisearch fetch (native hybrid BM25 + vector search    │
by default; top 60 per variant)                         │
     │                                                  │
     ▼                                                  │
RRF merge (weighted by variant index, keyed by PMID)    │
     │                                                  │
     ▼                                                  │
PMID deduplication (across tool calls within a request) │
     │                                                  │
     ▼                                                  │
Semantic rerank (configurable embedding provider,       │
asymmetric task types) → composite score → top 10       │
     │                                                  │
     └── tool result back to model ─────────────────────┘
     │   (repeat up to 5 turns)
     ▼
Final answer → clinical/citation guardrails → citation audit → JSON response
```

### Composite rerank score

```
score = cosine_similarity(query, document)
      + 0.03 * lexical_overlap
      + 0.02 * mesh_overlap
      + 0.04 * publication_type_overlap
      + 2.0  * rrf_score
      + hierarchy_boost        # Practice Guideline +0.12, SR/MA +0.07, RCT +0.05
      + source_boost           # AAO-HNS authorship +0.16, guideline phrase +0.06
      + recency_boost          # up to +0.18 for guideline queries (20-year window)
```

Guideline-intent queries (containing "current", "guideline", "indications", etc.) activate the source and recency boosts. Evidence/source/recency boosts are topic-gated: when semantic similarity is below `BOOST_TOPIC_GATE_THRESHOLD` (default `0.55`), those boosts are multiplied by `BOOST_TOPIC_GATE_FACTOR` (default `0.25`) so off-topic guidelines do not outrank direct evidence as easily. Embedding failures fall back to a lexical policy rerank that applies the same boosts without cosine similarity.

### Meilisearch hybrid search

The hosted index has user-provided OpenAI `text-embedding-3-large` vectors for all 16,496 documents under the Meilisearch embedder name `otology_openai_large`. Native hybrid first-stage fetch is enabled by default:

```bash
MEILI_HYBRID_SEARCH=1
MEILI_HYBRID_EMBEDDER=otology_openai_large
MEILI_HYBRID_PROVIDER=openai
MEILI_HYBRID_MODEL=text-embedding-3-large
MEILI_HYBRID_SEMANTIC_RATIO=0.3
```

Hybrid search affects the first-stage candidate pool only. The app still applies its own clinical semantic reranker afterward.

### Embedding cache

Embeddings are cached in a SQLite database at `data/runtime/embedding-cache.sqlite`, keyed by `(provider, model, task_type, sha256(text))`. Set `DISABLE_EMBEDDING_CACHE=1` to bypass it.

## Data

The index is built from PubMed using a broad MeSH-anchored otology query supplemented by targeted pulls for guidelines, systematic reviews, meta-analyses, and RCTs on specific high-priority topics (sudden hearing loss, tympanostomy tubes, otitis media with effusion). Each document includes title, abstract, authors, journal, year, MeSH terms, publication types, and PubMed URL.

## Setup

### Requirements

- Python 3.11+
- A running Meilisearch instance (the class hosts one at `http://search.858.mba:7700`)
- A Google GenAI API key (Gemini API, used for Gemma-4)
- An OpenAI API key if using OpenAI embeddings, retrieval reranking, or Meilisearch hybrid query vectors

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file (or export these directly):

```bash
GEMINI_API_KEY=your_google_genai_key
OPENAI_API_KEY=your_openai_key        # needed for OpenAI embeddings / hybrid query vectors
MEILI_URL=http://search.858.mba:7700
MEILI_INDEX=your_index_name
MEILI_SEARCH_KEY=your_search_only_key
MEILI_WRITE_KEY=your_write_key        # only needed for data upload
```

Optional:

```bash
EMBEDDING_PROVIDER=openai             # default; set to "gemini" to use Gemini embeddings
EMBEDDING_MODEL=text-embedding-3-large # overrides the default per provider
MODEL_RETRY_ATTEMPTS=3                # retries transient model API failures
MODEL_RETRY_BASE_DELAY_SECONDS=1      # exponential backoff base delay
EMBEDDING_CACHE_PATH=data/runtime/embedding-cache.sqlite
DISABLE_EMBEDDING_CACHE=1             # bypass the SQLite cache
MEILI_HYBRID_SEARCH=1                 # default; set to 0 to force BM25 first-stage fetch
MEILI_HYBRID_EMBEDDER=otology_openai_large
MEILI_HYBRID_PROVIDER=openai
MEILI_HYBRID_MODEL=text-embedding-3-large
MEILI_HYBRID_SEMANTIC_RATIO=0.3
BOOST_TOPIC_GATE_THRESHOLD=0.55
BOOST_TOPIC_GATE_FACTOR=0.25
NCBI_API_KEY=your_ncbi_key            # optional; raises PubMed rate limit
PORT=8080                             # default port
```

### Build the index

Fetch articles from PubMed and upload to Meilisearch:

```bash
python3 scripts/fetch_pubmed.py --output my-data/pubmed-otology.json
python3 scripts/upload.py my-data/pubmed-otology.json --reset \
  --filterable year \
  --filterable mesh_terms \
  --filterable publication_type \
  --filterable journal
```

`fetch_pubmed.py` defaults to up to 10,000 broad articles plus supplemental pulls for high-value evidence types. Pass `--max N` to limit the broad fetch or `--no-supplemental` to skip the supplemental queries.

To add or refresh native Meilisearch hybrid vectors for the existing corpus:

```bash
set -a && source .env && set +a
EMBEDDING_MODEL=text-embedding-3-large \
python3 scripts/vectorize_and_upload.py \
  --embedding-batch-size 64 \
  --upload-batch-size 100
```

This script uses the already-fetched JSON corpus; it does not re-scrape PubMed. It stores/reuses document embeddings in `data/runtime/embedding-cache.sqlite`, uploads `_vectors.otology_openai_large` for every document, and configures a Meilisearch `userProvided` embedder with 3072 dimensions.

### Run the server

```bash
set -a && source .env && set +a
python3 agent/server.py
```

Opens at [http://localhost:8080](http://localhost:8080) (chat UI) and [http://localhost:8080/search](http://localhost:8080/search) (Meilisearch browse UI).
Startup logs print the active retrieval mode; the validated default is Meilisearch hybrid plus OpenAI `text-embedding-3-large` rerank embeddings.

If Flask debug/reload fails in a sandboxed environment with `Operation not permitted`, start without the reloader:

```bash
set -a && source .env && set +a
python3 -c "from agent.server import app; app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)"
```

Then open [http://127.0.0.1:8080/](http://127.0.0.1:8080/). This fallback disables auto-reload.

## API

**POST /chat**

```json
{
  "user_id": "anonymous-browser-uuid",
  "conversation_id": "optional-existing-conversation-uuid",
  "message": "What are current indications for tympanostomy tubes?",
  "trace": true
}
```

Response:

```json
{
  "conversation_id": "...",
  "reply": "...",
  "citations": [{"label": "Title (Year)", "url": "https://pubmed.ncbi.nlm.nih.gov/..."}],
  "citation_warnings": [],
  "citation_format_warnings": [],
  "clinical_guardrail_warnings": [],
  "citation_support_warnings": [],
  "trace": {
    "tool_calls": [...],
    "forced_final": false,
    "rerank_disabled": false,
    "out_of_scope": false,
    "citation_repair_attempted": false,
    "clinical_guardrail_warnings": [],
    "citation_support_warnings": []
  }
}
```

`citation_warnings` lists any PubMed URLs in the answer that were not returned by tool calls. `citation_format_warnings` lists citation-format problems such as an answer with retrieved papers but no valid PubMed citation links. `clinical_guardrail_warnings` and `citation_support_warnings` report narrow post-synthesis interventions, such as correcting an overbroad AOM watchful-waiting statement or appending AAP/AAFP support for AOM observation criteria when that guideline was retrieved but not cited near the relevant criterion. If retrieved papers exist but no valid citation links are parsed, the app attempts one citation repair using only retrieved source URLs; `trace.citation_repair_attempted` reports whether that happened. `trace` is only included when `"trace": true` is sent in the request.

The `search_papers` tool accepts: `query` (required), `mesh_terms`, `publication_types`, `year_from`, `year_to`, `journal`, `max_results` (default 10, max 12).

Conversation history is stored server-side in SQLite at `data/runtime/conversations.sqlite` by default. The browser-generated `user_id` is a bearer credential for local/demo use; do not expose this app publicly without real authentication. The model receives the latest 30 user-visible messages from the active conversation; older messages remain visible in the UI but are not included in model context.

Journal constraints are fuzzy identity filters, not broad specialty-family filters. The tool expands common abbreviations and tolerates long official journal names, so `Otol Neurotol` can match `Otology & Neurotology : official publication...`. Distinctive journal identity tokens still have to match: `JAMA Otolaryngol Head Neck Surg` should not match `Archives of Otolaryngology--Head & Neck Surgery` just because both share `otolaryngology`, `head`, `neck`, and `surgery`. If a journal constraint is too narrow, the low-hit recovery path retries without it and reports that in `recovery_notes`.

## Benchmark

`benchmark.md` contains 27 evaluation questions across seven categories:

1. Guidelines / Current Practice
2. Treatment Evidence
3. Procedure / Surgical Questions
4. Diagnosis / Workup
5. Weak / Mixed / Controversial Evidence
6. Population-Specific Questions
7. Negative Controls (out-of-scope, should be declined)

Each answer is scored on six axes (1–5): retrieval quality, evidence ranking, synthesis quality, calibration, citation quality, clinical usefulness.

Run the benchmark harness against the live server:

```bash
python3 scripts/run_benchmark.py --questions 1-8
python3 scripts/run_benchmark.py --questions all
```

Full-agent results are saved to `benchmark-runs/<timestamp>/` as `results.json` (full traces), `summary.md` (tabular overview), and `answers.md` (readable answers, citations, and top retrieved PMIDs).

For faster retrieval-only testing:

```bash
set -a && source .env && set +a
EMBEDDING_PROVIDER=openai \
EMBEDDING_MODEL=text-embedding-3-large \
MEILI_HYBRID_SEARCH=1 \
MEILI_HYBRID_EMBEDDER=otology_openai_large \
MEILI_HYBRID_PROVIDER=openai \
MEILI_HYBRID_MODEL=text-embedding-3-large \
MEILI_HYBRID_SEMANTIC_RATIO=0.3 \
python3 scripts/run_benchmark.py \
  --questions all \
  --retrieval-only \
  --max-results 10 \
  --stop-on-rerank-fallback
```

Retrieval-only runs also write `retrieval.md`, including title, year, publication type, score, semantic score, RRF contribution, lexical/MeSH/publication-type components, boosts, and the topic-gate factor. `--stop-on-rerank-fallback` prevents accidentally treating lexical fallback output as a semantic retrieval baseline.

## Files

```
agent/server.py              Flask backend — retrieval, reranking, agentic loop
scripts/fetch_pubmed.py      PubMed data collection
scripts/upload.py            Meilisearch upload (Python)
scripts/upload.mjs           Meilisearch upload (Node)
scripts/vectorize_and_upload.py
                             Precompute document vectors and upload Meili _vectors
scripts/run_benchmark.py     Benchmark harness
benchmark.md                 Evaluation question set and scoring rubric
agent-review.md              Architecture review: what works, gaps, and planned fixes
search-app/chat.html         Chat UI
search-app/index.html        Meilisearch browse UI
my-data/pubmed-otology.json  Local copy of the indexed corpus
benchmark-runs/              Timestamped benchmark results
data/runtime/                Embedding cache (gitignored)
sample-data/                 Yale SOM course data (original class starter, unused)
```

## Known gaps and planned work

See `agent-review.md` for a full list. Highest-priority open items:

- Reranking weight validation — the composite score coefficients are hand-tuned; a sweep against benchmark citation recall could improve ordering
- Surgical comparison query expansion/corpus coverage — some surgical questions still retrieve broad or adjacent-topic evidence
- System prompt additions: search-budget guidance, 0-hit recovery instruction, conflict-handling guidance (guideline vs. recent meta-analysis)
