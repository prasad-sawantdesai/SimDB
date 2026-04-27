"""Qdrant vector-store helpers for SimDB field search.

All Qdrant state (client, embedder, globals) lives here so that the rest of
the API layer only needs to import this module and call the public functions.

Model: intfloat/e5-small-v2 (384-dim, asymmetric retrieval).
  - Index text prefixed with "passage: "
  - Query text prefixed with "query: "
  This gives significantly better recall for short physics queries (Te, ne, q)
  against long field description passages.

  To switch models: change _EMBED_MODEL and bump _COLLECTION_VERSION.
  The version suffix forces a fresh collection so stale MiniLM vectors are
  not mixed with e5 vectors.
"""
import re
import threading
import uuid as _uuid

# ─── Qdrant vector store (optional, graceful degradation if unavailable) ──── #
_EMBED_MODEL = "intfloat/e5-small-v2"
_COLLECTION_VERSION = "e5v2"
_QDRANT_COLLECTION = f"simdb_fields_{_COLLECTION_VERSION}"
_VECTOR_DIM = 384  # e5-small-v2  384-dim
_ENCODE_BATCH_SIZE = 256  # sentence-transformers batch size for GPU/CPU efficiency

_qdrant_client = None
_embedder = None
_qdrant_available: bool | None = None  # None = not yet probed
_qdrant_unavailable_reason = "Vector search is not initialized."

# Lock prevents race between background preload thread and first request
_init_lock = threading.Lock()


def _get_qdrant_unavailable_reason() -> str:
    return _qdrant_unavailable_reason


def _init_qdrant() -> bool:
    """Init Qdrant client and sentence-transformer embedder.

    Thread-safe via double-checked locking — safe to call from the app startup
    thread and from request threads simultaneously.

    Returns True if Qdrant is available, False otherwise (sets reason string).
    """
    global _qdrant_client, _embedder, _qdrant_available, _qdrant_unavailable_reason
    # Fast path — already resolved (True or False)
    if _qdrant_available is not None:
        return _qdrant_available

    with _init_lock:
        # Second check inside lock — another thread may have finished first
        if _qdrant_available is not None:
            return _qdrant_available

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ModuleNotFoundError as exc:
            _qdrant_available = False
            _qdrant_unavailable_reason = (
                f"Vector search dependency '{exc.name}' is not installed in the "
                "SimDB server environment."
            )
            return False
        except Exception as exc:
            _qdrant_available = False
            _qdrant_unavailable_reason = (
                f"Failed to import Qdrant client dependencies: "
                f"{exc.__class__.__name__}: {exc}"
            )
            return False

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            _qdrant_available = False
            _qdrant_unavailable_reason = (
                f"Vector search dependency '{exc.name}' is not installed in the "
                "SimDB server environment."
            )
            return False
        except Exception as exc:
            _qdrant_available = False
            _qdrant_unavailable_reason = (
                f"Failed to import sentence-transformers: "
                f"{exc.__class__.__name__}: {exc}"
            )
            return False

        try:
            _qdrant_client = QdrantClient(host="localhost", port=6333)
            _embedder = SentenceTransformer(_EMBED_MODEL)
            existing = {c.name for c in _qdrant_client.get_collections().collections}
            if _QDRANT_COLLECTION not in existing:
                _qdrant_client.create_collection(
                    _QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
                )
            _qdrant_available = True
            _qdrant_unavailable_reason = ""
        except Exception as exc:
            _qdrant_available = False
            _qdrant_unavailable_reason = (
                f"Vector search backend initialization failed: "
                f"{exc.__class__.__name__}: {exc}"
            )
        return _qdrant_available


def _build_field_text(ids_name: str, path: str, desc: str, units: str, field: dict) -> str:
    """Build the document text for a single field.

    Prefixed with "passage: " for e5-small-v2 asymmetric retrieval.
    Path words are repeated once to give them extra weight in the embedding.
    """
    full_path = f"{ids_name}/{path}"
    path_words = re.sub(r"[/_\[\]]", " ", full_path).strip()
    leaf = path.split("/")[-1].replace("_", " ")

    stats_text = ""
    if "value" in field:
        stats_text = f"value {field['value']}."
    elif "mean" in field:
        mn, mx, mu = field.get("min"), field.get("max"), field["mean"]
        stats_text = (
            f"mean {mu:.4g} range {mn:.4g} to {mx:.4g}."
            if mn is not None and mx is not None
            else f"mean {mu:.4g}."
        )
        if field.get("is_monotonic"):
            stats_text += " monotonic."

    body = " ".join(filter(None, [
        path_words, path_words, leaf, desc, stats_text,
        f"Units: {units}." if units else "",
    ])).strip()

    return f"passage: {body}"


def encode_query(text: str) -> list:
    """Encode a user search query with the asymmetric query prefix.

    Returns a plain Python list (Qdrant-compatible).
    """
    return _embedder.encode(f"query: {text}", show_progress_bar=False).tolist()


def _index_in_qdrant(simulation_uuid: str, file_uuid: str, ids_result: dict) -> None:
    """Upsert all fields from *ids_result* into Qdrant.

    Intended to run in a background daemon thread so it never blocks the HTTP
    response.  Each field dict is expected to have the keys produced by
    ``_walk_nonempty`` (path, dtype, units, description, shape, and optional
    statistics: value/min/max/mean/std/n_elements/n_nan/is_monotonic/sparsity).

    All field texts are encoded in a single batched call for efficiency
    """
    if not _init_qdrant():
        return
    try:
        from qdrant_client.models import PointStruct

        _ns = _uuid.NAMESPACE_DNS

        # Collect texts and metadata in one pass
        texts: list[str] = []
        metas: list[dict] = []
        for ids_name, occ_map in ids_result.items():
            for occ, fields in occ_map.items():
                for field in fields:
                    path = field["path"]
                    desc = field.get("description", "") or ""
                    units = field.get("units", "") or ""

                    texts.append(_build_field_text(ids_name, path, desc, units, field))

                    payload: dict = {
                        "simulation_uuid": simulation_uuid,
                        "file_uuid": file_uuid,
                        "ids_name": ids_name,
                        "occurrence": occ,
                        "path": f"{ids_name}/{path}",
                        "dtype": field.get("dtype", ""),
                        "units": units,
                        "description": desc,
                        "shape": field.get("shape"),
                    }
                    for stat_key in (
                        "value", "min", "max", "mean", "std",
                        "n_elements", "n_nan", "is_monotonic", "sparsity",
                    ):
                        if stat_key in field:
                            payload[stat_key] = field[stat_key]

                    metas.append({
                        "point_id": str(_uuid.uuid5(_ns, f"{file_uuid}:{ids_name}:{occ}:{path}")),
                        "payload": payload,
                    })

        if not texts:
            return

        # Single batched encode call
        vecs = _embedder.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            show_progress_bar=False,
        )

        # Build points and upsert in chunks of 128
        points = [
            PointStruct(id=m["point_id"], vector=v.tolist(), payload=m["payload"])
            for v, m in zip(vecs, metas)
        ]
        for i in range(0, len(points), 128):
            _qdrant_client.upsert(_QDRANT_COLLECTION, points[i : i + 128])
    except Exception:
        pass  # Qdrant failure must never break the API response
