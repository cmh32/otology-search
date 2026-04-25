# Otology Literature Research Agent

A clinical literature assistant for an APRN specializing in otology. It answers evidence-based questions about ear disease, hearing loss, vestibular disorders, and ear surgery by retrieving from a curated PubMed index and synthesizing results with a language model.

## What it does

- Retrieves relevant papers from a Meilisearch index of ~16,500 PubMed otology articles
- Runs up to five agentic tool-call turns to decompose multi-part questions
- Returns a synthesized answer with inline citations to PubMed URLs
- Flags any citation URLs that were not actually retrieved (hallucination guard)
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
Meilisearch BM25 fetch (top 60 per variant)             │
     │                                                  │
     ▼                                                  │
RRF merge (weighted by variant index, keyed by PMID)    │
     │                                                  │
     ▼                                                  │
PMID deduplication (across tool calls within a request) │
     │                                                  │
     ▼                                                  │
Semantic rerank (gemini-embedding-001, asymmetric       │
task types) → composite score → top 10                  │
     │                                                  │
     └── tool result back to model ─────────────────────┘
     │   (repeat up to 5 turns)
     ▼
Final answer → citation audit → JSON response
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

Guideline-intent queries (containing "current", "guideline", "indications", etc.) activate the source and recency boosts. Embedding failures fall back to a lexical policy rerank that applies the same boosts without cosine similarity.

### Embedding cache

Embeddings are cached in a SQLite database at `data/runtime/embedding-cache.sqlite`, keyed by `(provider, model, task_type, sha256(text))`. Set `DISABLE_EMBEDDING_CACHE=1` to bypass it.

## Data

The index is built from PubMed using a broad MeSH-anchored otology query supplemented by targeted pulls for guidelines, systematic reviews, meta-analyses, and RCTs on specific high-priority topics (sudden hearing loss, tympanostomy tubes, otitis media with effusion). Each document includes title, abstract, authors, journal, year, MeSH terms, publication types, and PubMed URL.

## Setup

### Requirements

- Python 3.11+
- A running Meilisearch instance (the class hosts one at `http://search.858.mba:7700`)
- A Google GenAI API key (Gemini API, used for both Gemma-4 and embeddings)

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file (or export these directly):

```bash
GEMINI_API_KEY=your_google_genai_key
MEILI_URL=http://search.858.mba:7700
MEILI_INDEX=your_index_name
MEILI_SEARCH_KEY=your_search_only_key
MEILI_WRITE_KEY=your_write_key        # only needed for data upload
```

Optional:

```bash
EMBEDDING_PROVIDER=gemini             # or "openai"
EMBEDDING_MODEL=gemini-embedding-001  # overrides the default per provider
EMBEDDING_CACHE_PATH=data/runtime/embedding-cache.sqlite
DISABLE_EMBEDDING_CACHE=1             # bypass the SQLite cache
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

### Run the server

```bash
python3 agent/server.py
```

Opens at [http://localhost:8080](http://localhost:8080) (chat UI) and [http://localhost:8080/search](http://localhost:8080/search) (Meilisearch browse UI).

## API

**POST /chat**

```json
{
  "messages": [
    {"role": "user", "content": "What are current indications for tympanostomy tubes?"}
  ],
  "trace": true
}
```

Response:

```json
{
  "reply": "...",
  "citations": [{"label": "Title (Year)", "url": "https://pubmed.ncbi.nlm.nih.gov/..."}],
  "citation_warnings": [],
  "trace": {
    "tool_calls": [...],
    "forced_final": false,
    "rerank_disabled": false,
    "out_of_scope": false
  }
}
```

`citation_warnings` lists any PubMed URLs in the answer that were not returned by tool calls. `trace` is only included when `"trace": true` is sent in the request.

The `search_papers` tool accepts: `query` (required), `mesh_terms`, `publication_types`, `year_from`, `year_to`, `journal`, `max_results` (default 10, max 12).

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

Results are saved to `benchmark-runs/<timestamp>/` as `results.json` (full traces) and `summary.md` (tabular overview).

## Files

```
agent/server.py              Flask backend — retrieval, reranking, agentic loop
scripts/fetch_pubmed.py      PubMed data collection
scripts/upload.py            Meilisearch upload (Python)
scripts/upload.mjs           Meilisearch upload (Node)
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
- Meilisearch hybrid search — the first-stage fetch is BM25-only; semantic matches that lack keyword overlap are excluded before the reranker sees them
- Hardcoded `current_year = 2026` in `recency_boost_for_year` should use `datetime.date.today().year`
- System prompt additions: search-budget guidance, 0-hit recovery instruction, conflict-handling guidance (guideline vs. recent meta-analysis)
