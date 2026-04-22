#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a JSON array of documents to your hosted Meilisearch index.",
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="sample-data/yale-som-courses-spring-2026.json",
        help="Path to a JSON file containing an array of documents.",
    )
    parser.add_argument("--url", default=os.environ.get("MEILI_URL", ""), help="Meilisearch base URL")
    parser.add_argument("--index", default=os.environ.get("MEILI_INDEX", ""), help="Meilisearch index UID")
    parser.add_argument(
        "--key",
        default=os.environ.get("MEILI_WRITE_KEY", ""),
        help="Meilisearch write-capable API key",
    )
    parser.add_argument("--primary-key", default="", help="Primary key field to use")
    parser.add_argument(
        "--filterable",
        action="append",
        default=[],
        help="Repeat to force specific filterable fields. Example: --filterable category --filterable tags",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the index before uploading documents.",
    )
    return parser.parse_args()


def ensure(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit(f"missing {label}; pass --{label} or set the matching environment variable")
    return value.rstrip("/") if label == "url" else value


def load_documents(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        documents = payload
    elif isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        documents = payload["documents"]
    else:
        raise SystemExit("expected a JSON array or an object with a top-level 'documents' array")
    if not documents:
        raise SystemExit("document list is empty")
    if not all(isinstance(doc, dict) for doc in documents):
        raise SystemExit("every document must be a JSON object")
    return documents


def request(url: str, api_key: str, method: str = "GET", body=None, ok=(200, 202)):
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            if status not in ok:
                raise SystemExit(f"{method} {url} returned unexpected HTTP {status}")
            text = response.read().decode("utf-8")
            return json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {body_text}") from exc


def maybe_request(url: str, api_key: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req) as response:
            text = response.read().decode("utf-8")
            return response.getcode(), json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        return exc.code, None


def wait_for_task(base_url: str, api_key: str, task_uid: int) -> None:
    for _ in range(90):
        task = request(f"{base_url}/tasks/{task_uid}", api_key, ok=(200,))
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            raise SystemExit(f"task {task_uid} failed: {json.dumps(task, indent=2)}")
        time.sleep(1)
    raise SystemExit(f"timed out waiting for task {task_uid}")


def choose_primary_key(documents, preferred: str):
    if preferred:
        return preferred, documents

    candidates = ["id", "uid", "_id", "slug", "url", "courseID", "courseId", "course_id"]
    for candidate in candidates:
        values = [doc.get(candidate) for doc in documents]
        if any(value in (None, "") for value in values):
            continue
        if len({str(value) for value in values}) == len(documents):
            return candidate, documents

    rewritten = []
    for index, doc in enumerate(documents, start=1):
        updated = dict(doc)
        updated["id"] = f"doc-{index:05d}"
        rewritten.append(updated)
    return "id", rewritten


def infer_filterable_attributes(documents, primary_key: str):
    preferred = [
        "category",
        "tags",
        "type",
        "term",
        "courseCategory",
        "courseType",
        "courseSession",
        "section",
        "allowBid",
    ]
    noisy_text_fields = {
        primary_key,
        "title",
        "name",
        "headline",
        "description",
        "summary",
        "abstract",
        "body",
        "content",
        "text",
        "url",
        "link",
        "website",
        "homepage",
    }

    selected = []
    total_docs = len(documents)
    keys = sorted({key for doc in documents for key in doc.keys()})
    for key in preferred + [key for key in keys if key not in preferred]:
        if key in noisy_text_fields or key in selected:
            continue
        values = [doc.get(key) for doc in documents if doc.get(key) not in (None, "", [])]
        if not values:
            continue

        simple_values = []
        valid = True
        for value in values:
            if isinstance(value, bool):
                simple_values.append(str(value).lower())
            elif isinstance(value, (int, float)):
                simple_values.append(str(value))
            elif isinstance(value, str):
                if len(value) > 40:
                    valid = False
                    break
                simple_values.append(value)
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                simple_values.extend(str(item) for item in value)
            else:
                valid = False
                break
        if not valid or not simple_values:
            continue

        unique_values = {value for value in simple_values if str(value).strip()}
        if len(unique_values) <= 1:
            continue
        if len(unique_values) > min(40, max(8, total_docs // 2)) and key not in preferred:
            continue
        selected.append(key)
        if len(selected) >= 10:
            break
    return selected


def recreate_index(base_url: str, api_key: str, index_uid: str, primary_key: str, reset: bool):
    status, _ = maybe_request(f"{base_url}/indexes/{urllib.parse.quote(index_uid)}", api_key)
    if reset and status == 200:
        task = request(
            f"{base_url}/indexes/{urllib.parse.quote(index_uid)}",
            api_key,
            method="DELETE",
            ok=(202, 204),
        )
        if task and task.get("taskUid") is not None:
            wait_for_task(base_url, api_key, task["taskUid"])
        status = 404

    if status == 404:
        task = request(
            f"{base_url}/indexes",
            api_key,
            method="POST",
            body={"uid": index_uid, "primaryKey": primary_key},
            ok=(202,),
        )
        wait_for_task(base_url, api_key, task["taskUid"])


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_file)
    base_url = ensure(args.url, "url")
    index_uid = ensure(args.index, "index")
    api_key = ensure(args.key, "key")

    documents = load_documents(json_path)
    primary_key, documents = choose_primary_key(documents, args.primary_key)
    filterable_attributes = args.filterable or infer_filterable_attributes(documents, primary_key)

    recreate_index(base_url, api_key, index_uid, primary_key, args.reset)

    if filterable_attributes:
        task = request(
            f"{base_url}/indexes/{urllib.parse.quote(index_uid)}/settings/filterable-attributes",
            api_key,
            method="PUT",
            body=filterable_attributes,
            ok=(202,),
        )
        wait_for_task(base_url, api_key, task["taskUid"])

    task = request(
        f"{base_url}/indexes/{urllib.parse.quote(index_uid)}/documents?primaryKey={urllib.parse.quote(primary_key)}",
        api_key,
        method="POST",
        body=documents,
        ok=(202,),
    )
    wait_for_task(base_url, api_key, task["taskUid"])

    stats = request(f"{base_url}/indexes/{urllib.parse.quote(index_uid)}/stats", api_key, ok=(200,))
    print(json.dumps(
        {
            "index": index_uid,
            "primaryKey": primary_key,
            "numberOfDocuments": stats.get("numberOfDocuments"),
            "filterableAttributes": filterable_attributes,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
