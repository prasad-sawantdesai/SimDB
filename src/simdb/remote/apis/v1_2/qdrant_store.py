"""Qdrant vector-store helpers for SimDB field search.

All Qdrant state (client, embedder, globals) lives here so that the rest of
the API layer only needs to import this module and call the public functions.
"""
import re
import uuid as _uuid

# ─── Qdrant vector store (optional, graceful degradation if unavailable) ──── #
_QDRANT_COLLECTION = "simdb_fields"
_VECTOR_DIM = 384  # all-MiniLM-L6-v2
_qdrant_client = None
_embedder = None
_qdrant_available: bool | None = None  # None = not yet probed
_qdrant_unavailable_reason = "Vector search is not initialized."


def _get_qdrant_unavailable_reason() -> str:
    return _qdrant_unavailable_reason


def _init_qdrant() -> bool:
    """Lazy-init Qdrant client and sentence-transformer embedder.

    Returns True if Qdrant is available, False otherwise (sets reason string).
    """
    global _qdrant_client, _embedder, _qdrant_available, _qdrant_unavailable_reason
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
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
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


def _index_in_qdrant(simulation_uuid: str, file_uuid: str, ids_result: dict) -> None:
    """Upsert all fields from *ids_result* into Qdrant.

    Intended to run in a background daemon thread so it never blocks the HTTP
    response.  Each field dict is expected to have the keys produced by
    ``_walk_nonempty`` (path, dtype, units, description, shape, and optional
    statistics: value/min/max/mean/std/n_elements/n_nan/is_monotonic/sparsity).
    """
    if not _init_qdrant():
        return
    try:
        from qdrant_client.models import PointStruct

        points = []
        _ns = _uuid.NAMESPACE_DNS
        for ids_name, occ_map in ids_result.items():
            for occ, fields in occ_map.items():
                for field in fields:
                    path = field["path"]
                    desc = field.get("description", "") or ""
                    units = field.get("units", "") or ""

                    # Build embedding text: path words repeated to dominate the
                    # embedding; description + statistics provide semantic context.
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

                    text = " ".join(filter(None, [
                        path_words, path_words, leaf, desc, stats_text,
                        f"Units: {units}." if units else "",
                    ])).strip()

                    vec = _embedder.encode(text, show_progress_bar=False).tolist()
                    point_id = str(_uuid.uuid5(_ns, f"{file_uuid}:{ids_name}:{occ}:{path}"))

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

                    points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        # Batch upsert in chunks of 128
        for i in range(0, len(points), 128):
            _qdrant_client.upsert(_QDRANT_COLLECTION, points[i : i + 128])
    except Exception:
        pass  # Qdrant failure must never break the API response
