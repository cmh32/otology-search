# Otology Agent — Retrieval, Synthesis, and System Prompt Review

Review of `agent/server.py`: how well the pipeline retrieves evidence, how well it synthesizes, what's missing, what's incorrect, and whether the system prompt steers the model effectively.

## Retrieval comprehensiveness

The pipeline is a solid keyword → embedding rerank design, but there are real comprehensiveness gaps.

### What works

- Meilisearch pulls 60 candidates per query variant using native hybrid fetch by default, then `semantic_rerank` scores by cosine + light lexical/MeSH/publication-type overlap + RRF + clinical boosts. The startup default is now Meilisearch hybrid with OpenAI `text-embedding-3-large` rerank embeddings.
- The agent loop allows up to 5 searches per question and supports parallel tool calls in a single turn (`agent/server.py:347-374`), so the model *can* decompose multi-angle questions.
- Filters for year, journal, and MeSH are exposed in the tool schema (`agent/server.py:68-108`).

### What's missing or weak

1. **No publication-type signal.** The system prompt demands evidence be ranked guideline → SR/meta → RCT → cohort → case series (`agent/server.py:42-45`), but the tool returns only title/abstract/MeSH/journal/year. PubMed's `PublicationTypeList` (`"Practice Guideline"`, `"Systematic Review"`, `"Meta-Analysis"`, `"Randomized Controlled Trial"`) is never indexed or filtered. The model must guess study design from the abstract. This is the single biggest retrieval gap for a clinical agent — you're asking the model to rank evidence on a dimension your index doesn't expose.
   - **Fix:** add `publication_type` as a filterable field and a `publication_types` param on the tool; or at minimum boost score when those strings appear in the title.
   - **Status:** fixed. PubMed fetch/upload, Meilisearch filters, the search tool schema, reranking features, and the prompt now expose and use publication types.

2. **Weak rerank for a guideline-first workflow.** Recency gets at most +0.08, and cosine similarity dominates. A 2011 case series with high lexical overlap beats a 2023 AAO-HNS guideline with lower title-match. For "current indications" questions this is actively wrong.
   - **Fix:** stronger recency weight when the query contains {"current", "guideline", "indications", "recommendations"}; or expose a sort mode.
   - **Status:** fixed for guideline/current-practice intent with publication-type hierarchy boosts, AAO-HNS/ACR/CNS source boosts, guideline-update boosts, and stronger currentness weighting. The same policy also applies in lexical fallback when embedding quota is exhausted.

3. **No deduplication across tool calls.** If the model issues 3 parallel searches that overlap, the same paper is presented 3× in context with no indication. Wastes context and over-weights a single source during synthesis.
   - **Status:** fixed. Each `/chat` request tracks returned PMIDs and excludes duplicates from later tool results.

4. **Embedding model is stale.** `text-embedding-004` (`agent/server.py:203`) has been superseded by `gemini-embedding-001`. More importantly, you're not using asymmetric task types — both the query and documents go in as `contents=[query] + snippets` with no `task_type="retrieval_query"` / `"retrieval_document"` distinction.
   - **Status:** fixed. Reranking now uses an embedding provider abstraction with asymmetric retrieval query/document task types. The default server path is OpenAI `text-embedding-3-large`; Gemini embeddings remain available by setting `EMBEDDING_PROVIDER=gemini`.

5. **No RRF / fallback when Meili ranks poorly.** Rerank happens only *within* Meili's top 60. If Meili misses a relevant paper (stemming, synonym, typo), embeddings can't rescue it. Reciprocal Rank Fusion of Meili score + embedding score would help.
   - **Status:** fixed for lightweight query variants. The tool now expands abbreviation/guideline/evidence queries, merges candidates by PMID with RRF-style contributions, and includes the RRF signal in semantic reranking.

6. **No query expansion.** No MeSH expansion, no HyDE, no synonym handling ("otitis media with effusion" vs "OME" vs "glue ear"). When the model picks a narrow query, it gets narrow results.
   - **Status:** partially fixed with lightweight abbreviation expansion and guideline/evidence suffix variants.

7. **Journal filter is exact match** (`agent/server.py:140`). Brittle — "JAMA Otolaryngol Head Neck Surg" vs "JAMA Otolaryngology - Head & Neck Surgery" will silently zero-out.
   - **Status:** fixed. Journal constraints now bias the Meilisearch query and apply normalized fuzzy post-filtering, including common journal abbreviation expansions such as `Otol` → `Otology`, `Neurotol` → `Neurotology`, and `Surg` → `Surgery`.

8. **Silent zero-hit failure.** When filters eliminate all hits, the tool returns `count: 0` with no hint to the model ("your year filter removed 14 hits, your journal filter removed 3"). The model can't self-correct.
   - **Status:** fixed for strict filters. The tool now retries low-hit searches after relaxing journal, MeSH, or publication-type filters and returns `recovery_notes`.

## Synthesis

1. **Only 8 papers by default (max 12).** For a multi-part treatment question that's thin — one systematic review, one guideline, and six cohorts will dominate. Raise default to 10–12 when the model doesn't specify.
   - **Status:** fixed. The tool default is now 10 results, capped at 12.

2. **No guarantee citations come from retrieved papers.** The model can freely hallucinate `[Title (Year)](URL)`. Post-process the reply and verify every URL appears in the tool-call results, or at least warn.
   - **Status:** fixed. `/chat` checks final PubMed URLs against retrieved paper URLs, returns `citation_warnings` for unretrieved URLs, normalizes harmless citation Markdown glitches, deduplicates parsed citations, and returns `citation_format_warnings` when retrieved papers exist but no valid PubMed citation links parse.

3. **No across-turn memory of retrieved papers.** Between HTTP requests (`/chat` calls), only user/assistant *text* is replayed (`agent/server.py:322-325`). In a multi-turn chat, the model can't reference "the Cochrane review I retrieved last turn" — it must re-search. Fine for single-shot Q&A, weak for iterative conversations.

4. **300-word cap** (`agent/server.py:65`) is tight for the rubric in `benchmark.md`. Meniere's treatment evidence genuinely needs more. Raise to 400–500, or make it conditional on question type.

5. **Synthesis anchors on one paper when multiple guidelines are retrieved.** `prioritize recent clinical practice guidelines` created recency bias — the model would cite a 2025 international consensus repeatedly and skip a 2013 AAP guideline that scored higher in retrieval. Compounded by a singular response template ("Current guideline or consensus position") that primed single-paper answers.
   - **Status:** fixed. Replaced "recent" with "authoritative", added an explicit instruction to cite every relevant retrieved guideline (not just the most recent), made the response template plural, and mirrored all changes in `FINAL_SYSTEM_INSTRUCTION`.

6. **Extracted citations not deduplicated.** `extracted_citations` returned one row per inline `[Title (Year)](URL)` mention; a paper cited five times inline produced five identical rows in the `citations` array.
   - **Status:** fixed. `extracted_citations` now keys on normalized URL and skips duplicates.

## Likely bugs / correctness

1. **Verify tool calling with `gemma-4-31b-it`.** The model ID is valid (Gemma 4 exists), but the code relies on native function-calling via `types.Tool`, `function_call`/`function_response` parts (`agent/server.py:331`, `:380`, `:337-374`). Historically Gemma served through the Gemini API has not supported that surface the way Gemini models do — confirm it's actually emitting `function_call` parts rather than silently falling through to `if not function_calls: return response.text` (`agent/server.py:340-341`), which would return answers with no retrieval at all. If tool calls aren't firing, the rest of the review is secondary. If they are firing, carry on.
   - **Status:** fixed. `test_tool_calling.py` verified that Gemma emits native `function_call` parts.

2. **Hardcoded year** `current_year = 2026` (`agent/server.py:209`). Use `datetime.date.today().year`.
   - **Status:** fixed. `recency_boost_for_year` now uses `datetime.date.today().year`.

3. **Forced-final-turn prompt reuse** (`agent/server.py:379-383`). The same system prompt is passed when tools are disabled. It still instructs "Before answering, call the tool", which may confuse the model. Swap in a synthesis-only variant.
   - **Status:** fixed. Final synthesis now uses `FINAL_SYSTEM_INSTRUCTION`, which explicitly tells the model not to call or request tools.

## System prompt effectiveness

Strong on intent and structure, weak on operational grounding.

### Effective parts

- Clear persona and scope (otology APRN).
- Explicit evidence hierarchy and endpoint-aware framing (`agent/server.py:42-46`).
- Separate output structures for "treatment evidence" vs "guidelines" questions matches the benchmark categories.
- "Literature-retrieved is the only source of truth… if evidence is thin, say so" — good calibration framing.

### Weak or missing

- **Tells the model to prefer guidelines/SRs without giving it the means.** Either add `publication_type` to the index/tool and reference it in the prompt, or give the model concrete query patterns (append `"clinical practice guideline"`, `"consensus statement"`, `"AAO-HNS"`, `"Cochrane"` to the query).
  - **Status:** fixed. `publication_types` is now an indexed/filterable tool parameter, and query expansion adds guideline/evidence suffix variants.
- **No examples.** A couple of one-line search exemplars ("for a guideline question → `query='sudden sensorineural hearing loss guideline', year_from=2019`") would sharply improve tool use.
- **No search-budget guidance.** The model doesn't know it only has 5 tool turns. Tell it explicitly, and tell it to reserve at least one turn for refinement.
- **No 0-hit recovery instruction.** Add: "If a search returns few or no hits, broaden the query or drop filters before answering."
- **No anti-hallucination rule for citations.** Add: "Only cite papers returned by the search_papers tool. Never fabricate a title, year, or URL."
  - **Status:** fixed. The system prompts include this rule, and `/chat` audits final PubMed URLs against retrieved papers.
- **"Do not provide personal medical advice"** is mildly mis-targeted — the user is a clinician; the risk is over-claiming or missing nuance, not patient advice. Reframe as "write for a clinician peer; do not add disclaimers about seeing a doctor."
- **Conflict handling is absent.** No guidance for "guideline says X, recent meta-analysis says Y" — exactly the interesting case for an APRN.
- **Out-of-scope handling is absent.** No guidance for when a question drifts to rhinology/laryngology.
  - **Status:** fixed. A pre-tool out-of-scope check declines clearly non-otology questions.

## Highest-leverage fixes

- [x] **Confirm `gemma-4-31b-it` is actually tool-calling** — verified via `test_tool_calling.py`; function_call parts fire correctly.
- [x] **Strip text preamble from tool-call turns** — `agent/server.py:344-346` now appends only function_call parts to `contents`, preventing token waste across turns.
- [x] **Index and filter on `publication_type`** — added to PubMed fetch output, Meilisearch retrieval/filter path, search tool schema, reranking context, and system prompt.
- [x] **Supplement PubMed data with high-value evidence queries** — fetch script now merges broad otology results with guideline/high-level evidence and benchmark-critical query sets; hosted index rebuilt from 16,496 docs.
- [x] **Add an anti-hallucination citation rule** — system prompt now forbids fabricated citations, and `/chat` returns `citation_warnings` when final PubMed URLs were not retrieved by tools.
- [x] **Dedupe papers across tool calls** — each `/chat` request tracks returned PMIDs and drops duplicates before returning later tool results to the model.
- [x] **Switch to `gemini-embedding-001`** with asymmetric `task_type="retrieval_query"` / `"retrieval_document"`; raise default `max_results` to 10.
- [x] **Add embedding provider abstraction and cache** — reranking now supports `EMBEDDING_PROVIDER=gemini` (default) or `openai`, isolates cache entries by provider/model/task/text hash, and stores vectors in ignored SQLite cache storage.
- [x] **Add RRF** between Meili rank and embedding rank; stop relying solely on rerank within top-60 query variants.
- [x] **Swap the forced-final-turn prompt** to a synthesis-only variant — final turn now uses a no-tools synthesis prompt.
- [x] **Boost guideline authority/currentness in reranking** — guideline-intent queries now prioritize practice guidelines, U.S. specialty-society sources, guideline updates, and recent current-practice records.
- [x] **Use Meilisearch's native hybrid search for the first-stage fetch** — existing corpus vectors were uploaded under embedder `otology_openai_large`, index stats verified all 16,496 documents embedded, and server startup now defaults to native hybrid fetch plus OpenAI rerank embeddings. Set `MEILI_HYBRID_SEARCH=0` to force BM25-only first-stage fetch.
- [x] **Add narrow topic guard for ossiculoplasty retrieval drift** — ossiculoplasty/PORP/TORP queries now apply a trace-visible topic penalty to clearly adjacent but off-topic cochlear implant / vestibular schwannoma / generic hearing-outcome papers. Q7 retrieval-only and full-agent checks passed after the change.

- [ ] **Validate and tune the reranking score weights.** The composite score formula (`semantic_score + 0.03*lexical + 0.02*mesh + 2.0*rrf_score + boosts`) is hand-picked with no empirical validation. The `2.0 * rrf_score` coefficient is large enough that BM25 rank still dominates final ordering even after semantic scoring. Run the benchmark across a sweep of weight combinations and pick values that maximize citation recall on known-answer questions.

## Smaller cleanups

- [x] Replace hardcoded `current_year = 2026` (`agent/server.py:209`) with `datetime.date.today().year`.
- [x] Make the journal filter case-insensitive or fuzzy (`agent/server.py:140`).
- [x] When the tool returns `count: 0`, include a hint about which filters eliminated hits.
- [x] Add search-budget guidance to system prompt — prompt now tells the model it has at most 5 tool-call turns, to spend early turns on broad coverage, and to reserve a later turn for refinement when needed.
- [ ] Add 0-hit recovery instruction to system prompt.
- [ ] Reframe "Do not provide personal medical advice" for a clinician audience.
- [ ] Add conflict-handling guidance to system prompt (guideline vs. recent meta-analysis).
- [ ] Consider logging cited vs. retrieved papers to feed the benchmark scoring loop.
- [x] Add out-of-scope handling for non-otology queries before tool use.
- [x] Add a short embedding retry and per-request rerank disable path after embedding 429s.
- [x] Add lightweight query expansion for abbreviations and guideline/evidence-seeking searches.
- [x] Add `DISABLE_EMBEDDING_CACHE=1` escape hatch and ignore `data/runtime/` cache artifacts.
- [x] Fix synthesis recency bias — replaced `prioritize recent` with `prioritize authoritative`, added explicit "cite all retrieved guidelines" instruction, pluralized the response template, mirrored in `FINAL_SYSTEM_INSTRUCTION`.
- [x] Deduplicate extracted citations array — `extracted_citations` now skips duplicate URLs so a paper cited N times inline appears only once in the `citations` response field.
