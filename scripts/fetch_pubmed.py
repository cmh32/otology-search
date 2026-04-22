#!/usr/bin/env python3
"""
Fetch ear/otology articles from PubMed and save as JSON for Meilisearch upload.

Usage:
    python3 scripts/fetch_pubmed.py
    python3 scripts/fetch_pubmed.py --max 3000 --output my-data/pubmed-otology.json

Reads NCBI_API_KEY from environment (optional but recommended for higher rate limits).
"""

import argparse
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# MeSH-anchored query covering ear/otology without pulling in unrelated ENT
SEARCH_QUERY = (
    "(otology[MeSH Terms] OR "
    "\"ear diseases\"[MeSH Terms] OR "
    "\"hearing loss\"[MeSH Terms] OR "
    "tinnitus[MeSH Terms] OR "
    "\"cochlear implants\"[MeSH Terms] OR "
    "\"vestibular diseases\"[MeSH Terms] OR "
    "\"otitis media\"[MeSH Terms] OR "
    "\"otitis externa\"[MeSH Terms] OR "
    "\"meniere disease\"[MeSH Terms] OR "
    "\"acoustic neuroma\"[MeSH Terms])"
)


def fetch(url: str, params: dict, api_key: str, retries: int = 3) -> dict:
    if api_key:
        params["api_key"] = api_key
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full_url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise SystemExit(f"Request failed after {retries} attempts: {exc}") from exc
            time.sleep(2 ** attempt)


def search_pmids(query: str, max_results: int, api_key: str) -> list[str]:
    print(f"Searching PubMed for up to {max_results} articles...")
    data = fetch(f"{BASE_URL}/esearch.fcgi", {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }, api_key)
    pmids = data["esearchresult"]["idlist"]
    print(f"Found {len(pmids)} articles.")
    return pmids


def fetch_summaries(pmids: list[str], api_key: str, batch_size: int = 200) -> list[dict]:
    results = []
    total = len(pmids)
    delay = 0.11 if api_key else 0.34

    for start in range(0, total, batch_size):
        batch = pmids[start:start + batch_size]
        print(f"  Fetching summaries {start + 1}–{min(start + batch_size, total)} of {total}...")
        data = fetch(f"{BASE_URL}/esummary.fcgi", {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "json",
        }, api_key)
        results.extend(data.get("result", {}).values())
        time.sleep(delay)

    return [r for r in results if isinstance(r, dict) and r.get("uid")]


def fetch_article_details(pmids: list[str], api_key: str, batch_size: int = 100) -> tuple[dict, dict]:
    """Returns (abstracts, mesh_terms) both keyed by PMID."""
    abstracts = {}
    mesh_by_pmid = {}
    delay = 0.11 if api_key else 0.34
    total = len(pmids)

    for start in range(0, total, batch_size):
        batch = pmids[start:start + batch_size]
        print(f"  Fetching abstracts + MeSH {start + 1}–{min(start + batch_size, total)} of {total}...")
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key
        url = f"{BASE_URL}/efetch.fcgi?{urllib.parse.urlencode(params)}"

        xml = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    xml = resp.read().decode("utf-8")
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    print(f"  Warning: batch failed ({exc}), skipping.")
                time.sleep(2 ** attempt)

        for article in re.split(r"<PubmedArticle>", xml)[1:]:
            pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
            if not pmid_match:
                continue
            pmid = pmid_match.group(1)

            # Collect all AbstractText sections (some abstracts have multiple labeled sections)
            abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
            if abstract_parts:
                clean_parts = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in abstract_parts]
                abstracts[pmid] = " ".join(p for p in clean_parts if p)

            # MeSH descriptor names
            mesh_terms = re.findall(r"<DescriptorName[^>]*>([^<]+)</DescriptorName>", article)
            if mesh_terms:
                mesh_by_pmid[pmid] = [html.unescape(t.strip()) for t in mesh_terms]

        time.sleep(delay)

    return abstracts, mesh_by_pmid


def build_document(summary: dict, abstracts: dict, mesh_by_pmid: dict) -> dict:
    pmid = summary.get("uid", "")

    authors = [
        a.get("name", "") for a in summary.get("authors", [])
        if a.get("authtype") == "Author"
    ]

    pub_date = summary.get("pubdate", "")
    year = pub_date[:4] if pub_date and pub_date[:4].isdigit() else None
    journal = summary.get("fulljournalname") or summary.get("source", "")
    title = html.unescape(summary.get("title", "").rstrip("."))

    return {
        "id": f"pmid-{pmid}",
        "pmid": pmid,
        "title": title,
        "abstract": abstracts.get(pmid, ""),
        "authors": authors,
        "journal": journal,
        "year": int(year) if year else None,
        "pubdate": pub_date,
        "mesh_terms": mesh_by_pmid.get(pmid, []),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch ear/otology articles from PubMed.")
    parser.add_argument("--max", type=int, default=2000, help="Max articles to fetch (default 2000)")
    parser.add_argument("--output", default="my-data/pubmed-otology.json", help="Output JSON path")
    parser.add_argument("--no-abstracts", action="store_true", help="Skip abstract/MeSH fetching (faster)")
    args = parser.parse_args()

    api_key = os.environ.get("NCBI_API_KEY", "")
    if not api_key:
        print("Warning: NCBI_API_KEY not set. Rate limited to 3 req/s (slower).")

    pmids = search_pmids(SEARCH_QUERY, args.max, api_key)
    if not pmids:
        raise SystemExit("No articles found.")

    print("Fetching article summaries...")
    summaries = fetch_summaries(pmids, api_key)

    abstracts, mesh_by_pmid = {}, {}
    if not args.no_abstracts:
        print("Fetching abstracts + MeSH terms...")
        abstracts, mesh_by_pmid = fetch_article_details(pmids, api_key)

    documents = [build_document(s, abstracts, mesh_by_pmid) for s in summaries]
    documents = [d for d in documents if d["title"]]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(documents, indent=2))

    with_mesh = sum(1 for d in documents if d["mesh_terms"])
    with_abstract = sum(1 for d in documents if d["abstract"])
    print(f"\nDone! Saved {len(documents)} articles to {output_path}")
    print(f"  {with_abstract} have abstracts, {with_mesh} have MeSH terms")
    print(f"\nRun: python3 scripts/upload.py {output_path} --reset --filterable year --filterable mesh_terms --filterable journal")


if __name__ == "__main__":
    main()
