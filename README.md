# Search with Meilisearch

This starter repo is for the MGT858 full-text search assignment.

The point of the assignment is simple:

1. put some data you care about into your hosted Meilisearch index;
2. open the static search app locally; and
3. watch your own search engine work in the browser.

You do **not** need to run Meilisearch locally.
The class already hosts it for you.

## What you need from your dashboard

Open the course dashboard and find these values:

- `hw-meilisearch-url`
- `hw-meilisearch-index`
- `hw-meilisearch-write-key`
- `hw-meilisearch-search-key`

## Quick start with the sample Yale SOM course data

From this starter repo, set three environment variables:

```bash
export MEILI_URL='http://search.858.mba:7700'
export MEILI_INDEX='your-index-from-the-dashboard'
export MEILI_WRITE_KEY='your-write-key-from-the-dashboard'
```

Then upload the sample data with either Python or Node:

```bash
python3 scripts/upload.py sample-data/yale-som-courses-spring-2026.json --reset
```

or

```bash
node scripts/upload.mjs sample-data/yale-som-courses-spring-2026.json --reset
```

Then open:

```text
search-app/index.html
```

Paste these values into the page:

- Meilisearch URL
- index name
- **search-only** API key

Then search.

## Customize the static app for your schema

The default `search-app/index.html` makes a generic best effort:

- it tries to find a title from fields like `title`, `name`, or `courseTitle`;
- it tries to find a snippet from fields like `description`, `summary`, or `body`; and
- it shows extra short fields as metadata badges.

If your data uses a different schema, edit `search-app/index.html`.
The two most useful places are:

- `renderHit()` — controls how each result card is displayed
- `startSearch()` — where you can add extra InstantSearch widgets

For example, if your documents have a field called `genre`, you could add a filter widget like:

```js
widgets.panel({ templates: { header: 'Genre' } })(
  widgets.refinementList({ container: '#custom-filters', attribute: 'genre' })
)
```

That is completely fine for this assignment. The point is to make the search UI fit your data.

## Using your own data

You are encouraged to upload data you actually care about.

The easiest path is a JSON file containing an array of objects, for example:

```json
[
  {
    "id": "book-1",
    "title": "Database Internals",
    "description": "A book about storage engines, indexing, and query execution.",
    "category": "book",
    "tags": ["databases", "systems"],
    "url": "https://example.com/database-internals"
  }
]
```

The upload scripts will try to:

- infer a primary key if you already have one;
- create synthetic `id` values if you do not; and
- infer a few filterable fields for the search UI.

You can override the inference if you want:

```bash
python3 scripts/upload.py my-data.json --reset \
  --primary-key slug \
  --filterable category \
  --filterable tags
```

## Files in this repo

- `search-app/index.html` — single-file static search app
- `scripts/upload.py` — Python upload helper using only the standard library
- `scripts/upload.mjs` — Node upload helper using built-in APIs
- `sample-data/yale-som-courses-spring-2026.json` — frozen Yale SOM course snapshot

## Important safety rule

Use the **write key** only in terminal scripts.

Use the **search-only key** in the browser.

Do **not** paste your write key into browser JavaScript.
