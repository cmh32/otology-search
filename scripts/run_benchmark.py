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
        "trace": payload.get("trace", {}),
        "error": payload.get("error"),
        "server_log": stdout_buffer.getvalue().splitlines(),
    }


def summarize_result(result: dict) -> dict:
    trace = result.get("trace") or {}
    tool_calls = trace.get("tool_calls") or []
    retrieved_pmids = []
    for call in tool_calls:
        for paper in call.get("papers", []):
            pmid = paper.get("pmid")
            if pmid and pmid not in retrieved_pmids:
                retrieved_pmids.append(pmid)

    return {
        "id": result["id"],
        "question": result["question"],
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "reply_chars": len(result.get("reply") or ""),
        "tool_calls": len(tool_calls),
        "citations": len(result.get("citations") or []),
        "citation_warnings": len(result.get("citation_warnings") or []),
        "forced_final": bool(trace.get("forced_final")),
        "rerank_disabled": bool(trace.get("rerank_disabled")),
        "out_of_scope": bool(trace.get("out_of_scope")),
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
        "",
        "| ID | Status | Tools | Cites | Warnings | Forced | Rerank Fallback | Question |",
        "|---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for result in results:
        trace = result.get("trace") or {}
        question = result["question"].replace("|", "\\|")
        lines.append(
            f"| {result['id']} | {result['status']} | "
            f"{len(trace.get('tool_calls') or [])} | "
            f"{len(result.get('citations') or [])} | "
            f"{len(result.get('citation_warnings') or [])} | "
            f"{'Y' if trace.get('forced_final') else ''} | "
            f"{'Y' if trace.get('rerank_disabled') else ''} | "
            f"{question} |"
        )
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
    parser.add_argument("--questions", default="1-8", help="Question ids: all, 1-8, or comma list")
    parser.add_argument("--output-dir", default="benchmark-runs", help="Directory for timestamped run output")
    args = parser.parse_args()

    all_questions = load_questions(ROOT / "benchmark.md")
    selected = parse_selection(args.questions, all_questions)
    if not selected:
        raise SystemExit(f"No benchmark questions matched {args.questions!r}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "question_selection": args.questions,
        "embedding_provider": server.EMBEDDING_PROVIDER,
        "embedding_model": server.EMBEDDING_MODEL,
    }

    results = []
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
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    write_summary(run_dir / "summary.md", metadata, results)
    print(f"\nSaved benchmark run to {run_dir}")


if __name__ == "__main__":
    main()
