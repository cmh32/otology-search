#!/usr/bin/env python3
"""Precompute document embeddings and upload them as Meilisearch vectors."""

import argparse
import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_EMBEDDER_NAME = "otology_openai_large"
DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_CACHE_PATH = "data/runtime/embedding-cache.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed PubMed documents and upload _vectors for Meilisearch hybrid search.",
    )
    parser.add_argument("json_file", nargs="?", default="my-data/pubmed-otology.json")
    parser.add_argument("--url", default=os.environ.get("MEILI_URL", ""), help="Meilisearch base URL")
    parser.add_argument("--index", default=os.environ.get("MEILI_INDEX", ""), help="Meilisearch index UID")
    parser.add_argument("--key", default=os.environ.get("MEILI_WRITE_KEY", ""), help="Meilisearch write key")
    parser.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embedder-name", default=os.environ.get("MEILI_EMBEDDER_NAME", DEFAULT_EMBEDDER_NAME))
    parser.add_argument("--cache-path", default=os.environ.get("EMBEDDING_CACHE_PATH", DEFAULT_CACHE_PATH))
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--upload-batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Limit documents for a smoke test")
    parser.add_argument("--dimensions", type=int, default=0, help="Vector dimensions for --configure-only")
    parser.add_argument("--skip-upload", action="store_true", help="Only populate the local embedding cache")
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Only configure the Meilisearch userProvided embedder; do not embed or upload documents.",
    )
    return parser.parse_args()


def ensure(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit(f"missing {label}; pass --{label} or set the matching environment variable")
    return value.rstrip("/") if label == "url" else value


def load_documents(path: Path, limit: int) -> list[dict]:
    payload = json.loads(path.read_text())
    documents = payload if isinstance(payload, list) else payload.get("documents", [])
    if not documents or not all(isinstance(doc, dict) for doc in documents):
        raise SystemExit("expected a JSON array of document objects")
    return documents[:limit] if limit else documents


def document_embedding_text(doc: dict) -> str:
    parts = [
        doc.get("title", ""),
        (doc.get("abstract", "") or "")[:1200],
        " ".join(doc.get("mesh_terms") or []),
        " ".join(doc.get("publication_type") or []),
        doc.get("journal", ""),
    ]
    return " ".join(part for part in parts if part).strip()


def request_json(url: str, method: str = "GET", api_key: str | None = None, body=None, ok=(200, 202)):
    data = None
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            status = response.getcode()
            if status not in ok:
                raise SystemExit(f"{method} {url} returned unexpected HTTP {status}")
            text = response.read().decode("utf-8")
            return json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {body_text}") from exc


def wait_for_task(base_url: str, api_key: str, task_uid: int) -> None:
    for _ in range(240):
        task = request_json(f"{base_url}/tasks/{task_uid}", api_key=api_key, ok=(200,))
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            raise SystemExit(f"task {task_uid} failed: {json.dumps(task, indent=2)}")
        time.sleep(1)
    raise SystemExit(f"timed out waiting for task {task_uid}")


class EmbeddingCache:
    def __init__(self, path: str, provider: str, model: str):
        self.path = Path(path)
        self.provider = provider
        self.model = model
        self.task_type = "retrieval_document"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.path)

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

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> list[list[float] | None]:
        with self._connect() as conn:
            return [self._get(conn, text) for text in texts]

    def _get(self, conn, text: str) -> list[float] | None:
        row = conn.execute(
            """
            SELECT embedding_json FROM embeddings
            WHERE provider = ? AND model = ? AND task_type = ? AND content_hash = ?
            """,
            (self.provider, self.model, self.task_type, self._hash(text)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(self, items: list[tuple[str, list[float]]]) -> None:
        with self._connect() as conn:
            now = int(time.time())
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (provider, model, task_type, content_hash, embedding_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self.provider,
                        self.model,
                        self.task_type,
                        self._hash(text),
                        json.dumps(embedding),
                        now,
                    )
                    for text, embedding in items
                ],
            )
            conn.commit()


def embed_openai(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    for attempt in range(6):
        try:
            response = request_json(
                "https://api.openai.com/v1/embeddings",
                method="POST",
                api_key=api_key,
                body=payload,
                ok=(200,),
            )
            data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
            embeddings = [item["embedding"] for item in data]
            if len(embeddings) != len(texts):
                raise RuntimeError(f"OpenAI returned {len(embeddings)} embeddings for {len(texts)} texts")
            return embeddings
        except SystemExit as exc:
            if "HTTP 429" not in str(exc) or attempt == 5:
                raise
            delay = min(60, 2 ** attempt)
            print(f"  [openai 429] retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def embed_documents(
    documents: list[dict],
    model: str,
    api_key: str,
    cache: EmbeddingCache,
    batch_size: int,
) -> list[tuple[dict, list[float]]]:
    pairs = [(doc, document_embedding_text(doc)) for doc in documents]
    result: list[tuple[dict, list[float]]] = []
    total = len(pairs)
    for start in range(0, total, batch_size):
        batch = pairs[start:start + batch_size]
        docs = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        cached = cache.get_many(texts)
        missing_indexes = [index for index, embedding in enumerate(cached) if embedding is None]
        if missing_indexes:
            missing_texts = [texts[index] for index in missing_indexes]
            fresh = embed_openai(missing_texts, model, api_key)
            cache.put_many(list(zip(missing_texts, fresh)))
            for index, embedding in zip(missing_indexes, fresh):
                cached[index] = embedding
        result.extend((doc, embedding) for doc, embedding in zip(docs, cached) if embedding is not None)
        print(f"  embedded/cache ready {min(start + batch_size, total)}/{total}")
    return result


def configure_meili_embedder(
    base_url: str,
    api_key: str,
    index_uid: str,
    embedder_name: str,
    dimensions: int,
) -> None:
    quoted_index = urllib.parse.quote(index_uid)
    task = request_json(
        f"{base_url}/indexes/{quoted_index}/settings",
        method="PATCH",
        api_key=api_key,
        body={
            "embedders": {
                embedder_name: {
                    "source": "userProvided",
                    "dimensions": dimensions,
                }
            }
        },
        ok=(202,),
    )
    wait_for_task(base_url, api_key, task["taskUid"])


def upload_vectors(
    base_url: str,
    api_key: str,
    index_uid: str,
    embedder_name: str,
    embedded_docs: list[tuple[dict, list[float]]],
    batch_size: int,
) -> None:
    quoted_index = urllib.parse.quote(index_uid)
    total = len(embedded_docs)
    for start in range(0, total, batch_size):
        batch = []
        for doc, embedding in embedded_docs[start:start + batch_size]:
            updated = dict(doc)
            updated["_vectors"] = {embedder_name: embedding}
            batch.append(updated)
        task = request_json(
            f"{base_url}/indexes/{quoted_index}/documents?primaryKey=id",
            method="POST",
            api_key=api_key,
            body=batch,
            ok=(202,),
        )
        wait_for_task(base_url, api_key, task["taskUid"])
        print(f"  uploaded vectors {min(start + batch_size, total)}/{total}")


def main() -> None:
    args = parse_args()
    base_url = ensure(args.url, "url")
    index_uid = ensure(args.index, "index")
    meili_key = ensure(args.key, "key")
    openai_key = ensure(args.openai_key, "openai-key")

    if args.configure_only:
        if not args.dimensions:
            raise SystemExit("--configure-only requires --dimensions")
        configure_meili_embedder(base_url, meili_key, index_uid, args.embedder_name, args.dimensions)
        print(f"configured Meilisearch embedder: {args.embedder_name}")
        return

    documents = load_documents(Path(args.json_file), args.limit)
    cache = EmbeddingCache(args.cache_path, "openai", args.model)
    embedded_docs = embed_documents(
        documents=documents,
        model=args.model,
        api_key=openai_key,
        cache=cache,
        batch_size=max(1, args.embedding_batch_size),
    )
    if not embedded_docs:
        raise SystemExit("no embeddings produced")

    if args.skip_upload:
        print("skipped vector upload")
        return
    upload_vectors(
        base_url=base_url,
        api_key=meili_key,
        index_uid=index_uid,
        embedder_name=args.embedder_name,
        embedded_docs=embedded_docs,
        batch_size=max(1, args.upload_batch_size),
    )
    dimensions = len(embedded_docs[0][1])
    print(f"embedding dimensions: {dimensions}")
    configure_meili_embedder(base_url, meili_key, index_uid, args.embedder_name, dimensions)
    print(f"configured Meilisearch embedder: {args.embedder_name}")


if __name__ == "__main__":
    main()
