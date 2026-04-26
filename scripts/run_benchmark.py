#!/usr/bin/env python3
"""Run the otology agent benchmark and save traceable results."""

import argparse
import contextlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import server  # noqa: E402


QUESTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$")


def load_questions(path: Path) -> list[dict]:
    questions = []
    section = ""
    in_questions = False
    for line in path.read_text().splitlines():
        if line.startswith("## Questions"):
            in_questions = True
            continue
        if in_questions and line.startswith("## "):
            break
        if not in_questions:
            continue
        if line.startswith("### "):
            section = line[4:].strip()
            continue
        match = QUESTION_RE.match(line)
        if match:
            questions.append({
                "id": int(match.group(1)),
                "section": section,
                "question": match.group(2),
            })
    return questions


def parse_selection(selection: str, questions: list[dict]) -> list[dict]:
    if selection == "all":
        return questions

    wanted = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(part))

    return [question for question in questions if question["id"] in wanted]


def run_question(client, question: dict) -> dict:
    started = time.time()
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        response = client.post(
            "/chat",
            json={
                "trace": True,
                "messages": [{"role": "user", "content": question["question"]}],
            },
        )
    elapsed = time.time() - started
    payload = response.get_json() or {}
    return {
        **question,
        "status": response.status_code,
        "elapsed_seconds": round(elapsed, 2),
        "reply": payload.get("reply"),
        "citations": payload.get("citations", []),
        "citation_warnings": payload.get("citation_warnings", []),
        "citation_format_warnings": payload.get("citation_format_warnings", []),
        "trace": payload.get("trace", {}),
        "error": payload.get("error"),
        "server_log": stdout_buffer.getvalue().splitlines(),
    }


def run_retrieval_question(question: dict, max_results: int) -> dict:
    started = time.time()
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        result = server.search_and_rerank(
            query=question["question"],
            max_results=max_results,
        )
    elapsed = time.time() - started
    return {
        **question,
        "status": 200,
        "elapsed_seconds": round(elapsed, 2),
        "reply": None,
        "citations": [],
        "citation_warnings": [],
        "citation_format_warnings": [],
        "trace": {
            "retrieval_only": True,
            "tool_calls": [result],
        },
        "error": None,
        "server_log": stdout_buffer.getvalue().splitlines(),
    }


def result_used_rerank_fallback(result: dict) -> bool:
    trace = result.get("trace") or {}
    papers = [
        paper
        for call in trace.get("tool_calls") or []
        for paper in call.get("papers", [])
    ]
    return bool(papers) and all(paper.get("semantic_score") is None for paper in papers)


def summarize_result(result: dict) -> dict:
    trace = result.get("trace") or {}
    tool_calls = trace.get("tool_calls") or []
    retrieved_pmids = []
    papers = []
    for call in tool_calls:
        for paper in call.get("papers", []):
            papers.append(paper)
            pmid = paper.get("pmid")
            if pmid and pmid not in retrieved_pmids:
                retrieved_pmids.append(pmid)
    rerank_fallback = result_used_rerank_fallback(result)

    return {
        "id": result["id"],
        "question": result["question"],
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "reply_chars": len(result.get("reply") or ""),
        "tool_calls": len(tool_calls),
        "citations": len(result.get("citations") or []),
        "citation_warnings": len(result.get("citation_warnings") or []),
        "citation_format_warnings": len(result.get("citation_format_warnings") or []),
        "zero_citations": (
            not trace.get("out_of_scope")
            and not trace.get("retrieval_only")
            and len(result.get("citations") or []) == 0
        ),
        "forced_final": bool(trace.get("forced_final")),
        "rerank_disabled": bool(trace.get("rerank_disabled") or rerank_fallback),
        "out_of_scope": bool(trace.get("out_of_scope")),
        "retrieval_only": bool(trace.get("retrieval_only")),
        "top_pmids": retrieved_pmids[:8],
    }


def write_summary(path: Path, metadata: dict, results: list[dict]) -> None:
    lines = [
        "# Benchmark Run",
        "",
        f"- Date: `{metadata['date']}`",
        f"- Git commit: `{metadata['git_commit']}`",
        f"- Questions: `{metadata['question_selection']}`",
        f"- Embedding provider: `{metadata['embedding_provider']}`",
        f"- Embedding model: `{metadata['embedding_model']}`",
        f"- Mode: `{'retrieval-only' if metadata.get('retrieval_only') else 'full-agent'}`",
        "",
        "| ID | Status | Tools | Cites | Zero Cites | URL Warnings | Format Warnings | Forced | Rerank Fallback | Question |",
        "|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|---|",
    ]
    for result in results:
        trace = result.get("trace") or {}
        rerank_fallback = result_used_rerank_fallback(result)
        out_of_scope = bool(trace.get("out_of_scope"))
        zero_citations = not out_of_scope and not metadata.get("retrieval_only") and len(result.get("citations") or []) == 0
        question = result["question"].replace("|", "\\|")
        lines.append(
            f"| {result['id']} | {result['status']} | "
            f"{len(trace.get('tool_calls') or [])} | "
            f"{len(result.get('citations') or [])} | "
            f"{'Y' if zero_citations else ''} | "
            f"{len(result.get('citation_warnings') or [])} | "
            f"{len(result.get('citation_format_warnings') or [])} | "
            f"{'Y' if trace.get('forced_final') else ''} | "
            f"{'Y' if trace.get('rerank_disabled') or rerank_fallback else ''} | "
            f"{question} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_answers(path: Path, results_path: Path, results: list[dict]) -> None:
    lines = [
        "# Benchmark Answers",
        "",
        f"Source: `{results_path}`",
        "",
    ]
    for result in results:
        trace = result.get("trace") or {}
        tool_calls = trace.get("tool_calls") or []
        lines.extend([
            f"## {result['id']}. {result['question']}",
            "",
            f"- Section: `{result.get('section', '')}`",
            f"- Status: `{result.get('status')}`",
            f"- Elapsed: `{result.get('elapsed_seconds')}s`",
            f"- Tool calls: `{len(tool_calls)}`",
            f"- Parsed citations: `{len(result.get('citations') or [])}`",
            f"- Citation URL warnings: `{len(result.get('citation_warnings') or [])}`",
            f"- Citation format warnings: `{len(result.get('citation_format_warnings') or [])}`",
            "",
        ])

        reply = (result.get("reply") or "").strip()
        lines.extend([reply or f"Error: {result.get('error') or 'No answer generated'}", ""])

        citations = result.get("citations") or []
        if citations:
            lines.extend(["Citations:", ""])
            for citation in citations:
                label = citation.get("label") or "citation"
                url = citation.get("url") or ""
                lines.append(f"- [{label}]({url})")
            lines.append("")

        retrieved = []
        seen_pmids = set()
        for call in tool_calls:
            for paper in call.get("papers", []):
                pmid = paper.get("pmid")
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    retrieved.append(paper)
        if retrieved:
            lines.extend(["Top retrieved PMIDs:", ""])
            for paper in retrieved[:10]:
                lines.append(f"- {paper.get('pmid')}: {paper.get('title', '')}")
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def write_retrieval_report(path: Path, results: list[dict]) -> None:
    lines = [
        "# Retrieval Baseline",
        "",
    ]
    for result in results:
        trace = result.get("trace") or {}
        calls = trace.get("tool_calls") or []
        call = calls[0] if calls else {}
        papers = call.get("papers", [])
        rerank_fallback = result_used_rerank_fallback(result)
        lines.extend([
            f"## {result['id']}. {result['question']}",
            "",
            f"- Hits: `{len(papers)}`",
            f"- Elapsed: `{result.get('elapsed_seconds')}s`",
            f"- Query variants: `{len(call.get('query_variants') or [])}`",
            f"- Rerank fallback: `{'yes' if rerank_fallback else 'no'}`",
        ])
        query_variants = call.get("query_variants") or []
        if query_variants:
            lines.append(f"- Variants: {'; '.join(query_variants)}")
        recovery_notes = call.get("recovery_notes") or []
        if recovery_notes:
            lines.append(f"- Recovery notes: {'; '.join(recovery_notes)}")
        lines.extend([
            "",
            "| Rank | PMID | Year | Pub Type | Score | Semantic | RRF | Lex | MeSH | Pub | Hier | Src | Rec | Gate | Title |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for rank, paper in enumerate(papers, start=1):
            pub_types = ", ".join((paper.get("publication_type") or [])[:3])
            title = (paper.get("title") or "").replace("|", "\\|")
            score = paper.get("score")
            semantic_score = paper.get("semantic_score")
            score_text = "" if score is None else str(score)
            semantic_text = "" if semantic_score is None else str(semantic_score)
            rrf_text = "" if paper.get("rrf_component") is None else str(paper.get("rrf_component"))
            lexical_text = "" if paper.get("lexical_component") is None else str(paper.get("lexical_component"))
            mesh_text = "" if paper.get("mesh_component") is None else str(paper.get("mesh_component"))
            pub_text = "" if paper.get("publication_type_component") is None else str(paper.get("publication_type_component"))
            hierarchy_text = "" if paper.get("hierarchy_boost") is None else str(paper.get("hierarchy_boost"))
            source_text = "" if paper.get("source_boost") is None else str(paper.get("source_boost"))
            recency_text = "" if paper.get("recency_boost") is None else str(paper.get("recency_boost"))
            gate_text = "" if paper.get("topic_boost_factor") is None else str(paper.get("topic_boost_factor"))
            lines.append(
                f"| {rank} | {paper.get('pmid') or ''} | {paper.get('year') or ''} | "
                f"{pub_types.replace('|', '\\|')} | {score_text} | {semantic_text} | "
                f"{rrf_text} | {lexical_text} | {mesh_text} | {pub_text} | "
                f"{hierarchy_text} | {source_text} | {recency_text} | {gate_text} | {title} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run otology benchmark questions through /chat.")
    parser.add_argument(
        "--benchmark-file",
        default=str(ROOT / "benchmark.md"),
        help="Markdown file containing benchmark questions in the benchmark.md format.",
    )
    parser.add_argument("--questions", default="1-8", help="Question ids: all, 1-8, or comma list")
    parser.add_argument("--output-dir", default="benchmark-runs", help="Directory for timestamped run output")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only the first-stage search/rerank path without calling the answer model.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Number of papers to return per retrieval-only question.",
    )
    parser.add_argument(
        "--stop-on-rerank-fallback",
        action="store_true",
        help="Stop immediately if a retrieval-only result falls back to lexical ranking.",
    )
    args = parser.parse_args()

    benchmark_file = Path(args.benchmark_file)
    if not benchmark_file.is_absolute():
        benchmark_file = ROOT / benchmark_file
    all_questions = load_questions(benchmark_file)
    selected = parse_selection(args.questions, all_questions)
    if not selected:
        raise SystemExit(f"No benchmark questions matched {args.questions!r}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "benchmark_file": str(benchmark_file),
        "question_selection": args.questions,
        "embedding_provider": server.EMBEDDING_PROVIDER,
        "embedding_model": server.EMBEDDING_MODEL,
        "retrieval_only": args.retrieval_only,
    }

    results = []
    if args.retrieval_only:
        for question in selected:
            print(f"[{question['id']}] {question['question']}")
            result = run_retrieval_question(question, args.max_results)
            results.append(result)
            rerank_fallback = result_used_rerank_fallback(result)
            print(
                f"  status={result['status']} hits={len(((result.get('trace') or {}).get('tool_calls') or [{}])[0].get('papers') or [])} "
                f"rerank_fallback={'yes' if rerank_fallback else 'no'} "
                f"elapsed={result['elapsed_seconds']}s"
            )
            if args.stop_on_rerank_fallback and rerank_fallback:
                print("  stopping because --stop-on-rerank-fallback is set")
                break
    else:
        with server.app.test_client() as client:
            for question in selected:
                print(f"[{question['id']}] {question['question']}")
                result = run_question(client, question)
                results.append(result)
                print(
                    f"  status={result['status']} tools={len((result.get('trace') or {}).get('tool_calls') or [])} "
                    f"citations={len(result.get('citations') or [])} "
                    f"warnings={len(result.get('citation_warnings') or [])} "
                    f"elapsed={result['elapsed_seconds']}s"
                )

    payload = {
        "metadata": metadata,
        "summary": [summarize_result(result) for result in results],
        "results": results,
    }
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2))
    write_summary(run_dir / "summary.md", metadata, results)
    if args.retrieval_only:
        write_retrieval_report(run_dir / "retrieval.md", results)
    else:
        write_answers(run_dir / "answers.md", results_path, results)
    print(f"\nSaved benchmark run to {run_dir}")


if __name__ == "__main__":
    main()
