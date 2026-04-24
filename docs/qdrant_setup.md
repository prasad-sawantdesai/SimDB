# Qdrant Vector Store — Setup & Usage

SimDB uses [Qdrant](https://qdrant.tech/) as an optional vector store to enable semantic search over IMAS simulation fields. When available, fields indexed via `GET /fields` can be queried with natural language via `GET /fields/search`.

---

## Prerequisites

Both packages are already installed in the SimDB virtual environment:

```
qdrant-client==1.17.1
sentence-transformers==5.4.1
```

The embedding model used is `all-MiniLM-L6-v2` (384-dimensional, ~90 MB, downloaded automatically on first use).

---

## Starting Qdrant

### Option A — Docker (recommended)

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify:
```bash
curl -s http://localhost:6333/collections | python3 -m json.tool
```

### Option B — Binary (no Docker)

```bash
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz | tar xz
./qdrant &
```
```bash
curl -s http://localhost:6333/collections | python3 -m json.tool
{
    "result": {
        "collections": []
    },
    "status": "ok",
    "time": 7.783e-06
}
```
---

## How Indexing Works

1. Call `GET /v1.2/simulation/<uuid>/fields` on any simulation.
2. SimDB scans the IMAS output file, returns the field list, and stores it in an **in-memory cache** (survives until server restart).
3. A **background daemon thread** immediately starts upserting field vectors into Qdrant — the HTTP response is not blocked.
4. Each field is embedded as: `"<ids_name>/<path>: <description>. Units: <units>."` using `all-MiniLM-L6-v2`.
5. Point IDs are **deterministic** (`uuid5` of `file_uuid:ids_name:occurrence:path`) — re-indexing the same file is idempotent.

### Qdrant collection: `simdb_fields`

| Payload field     | Description                                  |
|-------------------|----------------------------------------------|
| `simulation_uuid` | SimDB simulation UUID                        |
| `file_uuid`       | IMAS output file UUID                        |
| `ids_name`        | IDS name, e.g. `core_profiles`               |
| `occurrence`      | IDS occurrence index (integer)               |
| `path`            | Full field path, e.g. `core_profiles/profiles_1d/electrons/temperature` |
| `dtype`           | DD data type, e.g. `FLT_1D`                 |
| `units`           | Physical units, e.g. `eV`                   |
| `description`     | DD documentation string                      |
| `shape`           | Array shape if available, else `null`        |

---

## Confirm Fields Are Indexed

```bash
# 1. Check collection exists
curl -s http://localhost:6333/collections/simdb_fields | python3 -m json.tool

# 2. Count vectors for a specific file
curl -s -X POST http://localhost:6333/collections/simdb_fields/points/count \
  -H 'Content-Type: application/json' \
  -d '{
    "filter": {
      "must": [{"key": "file_uuid", "match": {"value": "<file-uuid>"}}]
    }
  }' | python3 -m json.tool
# Expected: {"result": {"count": <N>}, "status": "ok", ...}

# 3. Inspect a sample point
curl -s -X POST http://localhost:6333/collections/simdb_fields/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{
    "filter": {
      "must": [{"key": "file_uuid", "match": {"value": "<file-uuid>"}}]
    },
    "limit": 3,
    "with_payload": true,
    "with_vector": false
  }' | python3 -m json.tool
```

---

## Semantic Search (RAG)

```bash
SIM=bb47e1d63e5111f1a85548df37d69a88

# Step 1 — trigger indexing (first call only)
curl -s "http://localhost:5000/v1.2/simulation/$SIM/fields" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
total = sum(len(f) for ids in d['ids'].values() for f in ids.values())
print(f'IDSes: {list(d[\"ids\"].keys())}')
print(f'Total fields indexed: {total}')
print(f'Cached: {d.get(\"cached\", False)}')
"

# Step 2 — wait a few seconds for background indexing, then search

# Electron temperature
curl -s "http://localhost:5000/v1.2/simulation/$SIM/fields/search?q=electron+temperature+profile&limit=5" \
  | python3 -m json.tool

# Plasma boundary shape
curl -s "http://localhost:5000/v1.2/simulation/$SIM/fields/search?q=plasma+boundary+shape&limit=5" \
  | python3 -m json.tool

# Toroidal magnetic flux
curl -s "http://localhost:5000/v1.2/simulation/$SIM/fields/search?q=toroidal+magnetic+flux+surface&limit=5" \
  | python3 -m json.tool

# Ion density
curl -s "http://localhost:5000/v1.2/simulation/$SIM/fields/search?q=ion+density&limit=5" \
  | python3 -m json.tool
```

### Example response

```json
{
  "simulation": "bb47e1d6-3e51-11f1-a855-48df37d69a88",
  "file_uuid": "bd02a087-3e51-11f1-94f4-48df37d69a88",
  "query": "electron temperature profile",
  "results": [
    {
      "score": 0.921,
      "path": "core_profiles/profiles_1d/electrons/temperature",
      "ids_name": "core_profiles",
      "occurrence": 0,
      "dtype": "FLT_1D",
      "units": "eV",
      "description": "Temperature"
    },
    {
      "score": 0.874,
      "path": "core_profiles/profiles_1d/electrons/temperature_validity",
      "ids_name": "core_profiles",
      "occurrence": 0,
      "dtype": "INT_1D",
      "units": "",
      "description": "Indicator of the validity of the temperature data..."
    }
  ]
}
```

---

## Graceful Degradation

If Qdrant is not running:
- `GET /fields` still works normally (in-memory cache still used).
- `GET /fields/search` returns HTTP 503 with a clear error message.
- The SimDB server **never crashes** due to Qdrant being unavailable.

---

## Persistence

Qdrant stores its data in `./qdrant_storage/` (when started with Docker volume mount). Vectors survive server restarts. The SimDB in-memory cache does **not** survive restarts — call `/fields` again after restart to re-populate the cache, but Qdrant will already have the vectors so the search endpoint works immediately.

---

## Re-indexing

Re-calling `/fields` on the same file triggers a background upsert. Since point IDs are deterministic, duplicate fields are overwritten — not duplicated.
