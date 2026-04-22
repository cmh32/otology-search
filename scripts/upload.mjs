#!/usr/bin/env node

import { readFile } from 'node:fs/promises';
import process from 'node:process';

function parseArgs(argv) {
  const args = {
    jsonFile: 'sample-data/yale-som-courses-spring-2026.json',
    url: process.env.MEILI_URL || '',
    index: process.env.MEILI_INDEX || '',
    key: process.env.MEILI_WRITE_KEY || '',
    primaryKey: '',
    filterable: [],
    reset: false,
  };

  const positionals = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--url') args.url = argv[++i] || '';
    else if (arg === '--index') args.index = argv[++i] || '';
    else if (arg === '--key') args.key = argv[++i] || '';
    else if (arg === '--primary-key') args.primaryKey = argv[++i] || '';
    else if (arg === '--filterable') args.filterable.push(argv[++i] || '');
    else if (arg === '--reset') args.reset = true;
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else if (arg.startsWith('--')) {
      throw new Error(`unknown option: ${arg}`);
    } else {
      positionals.push(arg);
    }
  }

  if (positionals[0]) {
    args.jsonFile = positionals[0];
  }

  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/upload.mjs [json-file] [options]

Options:
  --url <url>             Meilisearch base URL (or MEILI_URL)
  --index <uid>           Meilisearch index UID (or MEILI_INDEX)
  --key <key>             Meilisearch write key (or MEILI_WRITE_KEY)
  --primary-key <field>   Primary key field to use
  --filterable <field>    Repeat to force specific filterable fields
  --reset                 Delete and recreate the index first
`);
}

function ensure(value, label) {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error(`missing ${label}; pass --${label} or set the matching environment variable`);
  }
  return label === 'url' ? trimmed.replace(/\/$/, '') : trimmed;
}

async function loadDocuments(path) {
  const raw = await readFile(path, 'utf8');
  const payload = JSON.parse(raw);
  const documents = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.documents)
      ? payload.documents
      : null;
  if (!documents) {
    throw new Error("expected a JSON array or an object with a top-level 'documents' array");
  }
  if (documents.length === 0) {
    throw new Error('document list is empty');
  }
  if (!documents.every((doc) => doc && typeof doc === 'object' && !Array.isArray(doc))) {
    throw new Error('every document must be a JSON object');
  }
  return documents;
}

async function api(baseUrl, apiKey, path, { method = 'GET', body, expected = [200, 202] } = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!expected.includes(response.status)) {
    const text = await response.text();
    throw new Error(`${method} ${path} failed with HTTP ${response.status}: ${text}`);
  }

  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

async function maybeGet(baseUrl, apiKey, path) {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (response.status === 404) {
    return { status: 404, body: null };
  }
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null };
}

async function waitForTask(baseUrl, apiKey, taskUid) {
  for (let i = 0; i < 90; i += 1) {
    const task = await api(baseUrl, apiKey, `/tasks/${taskUid}`, { expected: [200] });
    if (task.status === 'succeeded') {
      return;
    }
    if (task.status === 'failed') {
      throw new Error(`task ${taskUid} failed: ${JSON.stringify(task, null, 2)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`timed out waiting for task ${taskUid}`);
}

function choosePrimaryKey(documents, preferred) {
  if (preferred) {
    return { primaryKey: preferred, documents };
  }

  const candidates = ['id', 'uid', '_id', 'slug', 'url', 'courseID', 'courseId', 'course_id'];
  for (const candidate of candidates) {
    const values = documents.map((doc) => doc[candidate]);
    if (values.some((value) => value === null || value === undefined || value === '')) {
      continue;
    }
    if (new Set(values.map(String)).size === documents.length) {
      return { primaryKey: candidate, documents };
    }
  }

  return {
    primaryKey: 'id',
    documents: documents.map((doc, index) => ({ ...doc, id: `doc-${String(index + 1).padStart(5, '0')}` })),
  };
}

function inferFilterableAttributes(documents, primaryKey) {
  const preferred = [
    'category',
    'tags',
    'type',
    'term',
    'courseCategory',
    'courseType',
    'courseSession',
    'section',
    'allowBid',
  ];
  const noisyTextFields = new Set([
    primaryKey,
    'title',
    'name',
    'headline',
    'description',
    'summary',
    'abstract',
    'body',
    'content',
    'text',
    'url',
    'link',
    'website',
    'homepage',
  ]);

  const keys = [...new Set(documents.flatMap((doc) => Object.keys(doc)))].sort();
  const selected = [];
  const orderedKeys = [...preferred, ...keys.filter((key) => !preferred.includes(key))];

  for (const key of orderedKeys) {
    if (selected.includes(key) || noisyTextFields.has(key)) {
      continue;
    }

    const values = documents
      .map((doc) => doc[key])
      .filter((value) => value !== null && value !== undefined && value !== '' && !(Array.isArray(value) && value.length === 0));
    if (values.length === 0) {
      continue;
    }

    let valid = true;
    const simpleValues = [];
    for (const value of values) {
      if (typeof value === 'boolean') {
        simpleValues.push(String(value));
      } else if (typeof value === 'number') {
        simpleValues.push(String(value));
      } else if (typeof value === 'string') {
        if (value.length > 40) {
          valid = false;
          break;
        }
        simpleValues.push(value);
      } else if (Array.isArray(value) && value.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) {
        simpleValues.push(...value.map(String));
      } else {
        valid = false;
        break;
      }
    }

    if (!valid || simpleValues.length === 0) {
      continue;
    }

    const uniqueValues = new Set(simpleValues.filter((value) => value.trim()));
    if (uniqueValues.size <= 1) {
      continue;
    }
    if (uniqueValues.size > Math.min(40, Math.max(8, Math.floor(documents.length / 2))) && !preferred.includes(key)) {
      continue;
    }

    selected.push(key);
    if (selected.length >= 10) {
      break;
    }
  }

  return selected;
}

async function recreateIndex(baseUrl, apiKey, indexUid, primaryKey, reset) {
  let { status } = await maybeGet(baseUrl, apiKey, `/indexes/${encodeURIComponent(indexUid)}`);
  if (reset && status === 200) {
    const task = await api(baseUrl, apiKey, `/indexes/${encodeURIComponent(indexUid)}`, {
      method: 'DELETE',
      expected: [202, 204],
    });
    if (task?.taskUid !== undefined) {
      await waitForTask(baseUrl, apiKey, task.taskUid);
    }
    status = 404;
  }

  if (status === 404) {
    const task = await api(baseUrl, apiKey, '/indexes', {
      method: 'POST',
      body: { uid: indexUid, primaryKey },
      expected: [202],
    });
    await waitForTask(baseUrl, apiKey, task.taskUid);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const baseUrl = ensure(args.url, 'url');
  const indexUid = ensure(args.index, 'index');
  const apiKey = ensure(args.key, 'key');

  const loadedDocuments = await loadDocuments(args.jsonFile);
  const { primaryKey, documents } = choosePrimaryKey(loadedDocuments, args.primaryKey);
  const filterableAttributes = args.filterable.length > 0 ? args.filterable.filter(Boolean) : inferFilterableAttributes(documents, primaryKey);

  await recreateIndex(baseUrl, apiKey, indexUid, primaryKey, args.reset);

  if (filterableAttributes.length > 0) {
    const task = await api(baseUrl, apiKey, `/indexes/${encodeURIComponent(indexUid)}/settings/filterable-attributes`, {
      method: 'PUT',
      body: filterableAttributes,
      expected: [202],
    });
    await waitForTask(baseUrl, apiKey, task.taskUid);
  }

  const task = await api(baseUrl, apiKey, `/indexes/${encodeURIComponent(indexUid)}/documents?primaryKey=${encodeURIComponent(primaryKey)}`, {
    method: 'POST',
    body: documents,
    expected: [202],
  });
  await waitForTask(baseUrl, apiKey, task.taskUid);

  const stats = await api(baseUrl, apiKey, `/indexes/${encodeURIComponent(indexUid)}/stats`, { expected: [200] });
  console.log(JSON.stringify({
    index: indexUid,
    primaryKey,
    numberOfDocuments: stats.numberOfDocuments,
    filterableAttributes,
  }, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
