import uuid as _uuid
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from flask import request
from flask_restx import Namespace, Resource
from imas.ids_primitive import IDSPrimitive
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure

from simdb.cli.manifest import DataObject
from simdb.database import DatabaseError
from simdb.imas.utils import ImasError, open_imas
from simdb.remote.core.auth import User, requires_auth
from simdb.remote.core.typing import current_app

api = Namespace("data", path="/")

# ─── In-memory fields cache ───────────────────────────────────────────────── #
# Key: (file_uuid_str, ids_filter, occurrence_filter) → ids result dict
# Files are immutable so entries never expire.
_fields_cache: dict[tuple, dict] = {}

# ─── Qdrant vector store (optional, graceful degradation if unavailable) ──── #
_QDRANT_COLLECTION = "simdb_fields"
_VECTOR_DIM = 384  # all-MiniLM-L6-v2
_qdrant_client = None
_embedder = None
_qdrant_available: bool | None = None  # None = not yet probed


def _init_qdrant() -> bool:
    """Lazy-init Qdrant client and sentence-transformer embedder.
    Returns True if Qdrant is available, False otherwise (silently)."""
    global _qdrant_client, _embedder, _qdrant_available
    if _qdrant_available is not None:
        return _qdrant_available
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from sentence_transformers import SentenceTransformer

        _qdrant_client = QdrantClient(host="localhost", port=6333)
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        existing = {c.name for c in _qdrant_client.get_collections().collections}
        if _QDRANT_COLLECTION not in existing:
            _qdrant_client.create_collection(
                _QDRANT_COLLECTION,
                vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
            )
        _qdrant_available = True
    except Exception:
        _qdrant_available = False
    return _qdrant_available


def _index_in_qdrant(simulation_uuid: str, file_uuid: str, ids_result: dict) -> None:
    """Upsert all fields from *ids_result* into Qdrant. Intended to run in a
    background daemon thread so it never blocks the HTTP response."""
    if not _init_qdrant():
        return
    try:
        from qdrant_client.models import PointStruct

        points = []
        _ns = _uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace
        for ids_name, occ_map in ids_result.items():
            for occ, fields in occ_map.items():
                for field in fields:
                    path = field["path"]
                    desc = field.get("description", "") or ""
                    units = field.get("units", "") or ""
                    # Build embedding text: path words are repeated to dominate
                    # the embedding so path-based queries rank highest, then
                    # description + statistics provide semantic context.
                    full_path = f"{ids_name}/{path}"
                    path_words = re.sub(r"[/_\[\]]", " ", full_path).strip()
                    leaf = path.split("/")[-1].replace("_", " ")

                    # Stats fragment for the embedding text
                    stats_text = ""
                    if "value" in field:
                        v = field["value"]
                        stats_text = f"value {v}."
                    elif "mean" in field:
                        mn, mx, mu = field.get("min"), field.get("max"), field["mean"]
                        stats_text = (
                            f"mean {mu:.4g} range {mn:.4g} to {mx:.4g}."
                            if mn is not None and mx is not None
                            else f"mean {mu:.4g}."
                        )
                        if field.get("is_monotonic"):
                            stats_text += " monotonic."

                    # Path repeated twice → ~2/3 of token weight on path
                    text = " ".join(filter(None, [
                        path_words, path_words, leaf, desc, stats_text,
                        f"Units: {units}." if units else "",
                    ])).strip()

                    vec = _embedder.encode(text, show_progress_bar=False).tolist()
                    point_id = str(_uuid.uuid5(_ns, f"{file_uuid}:{ids_name}:{occ}:{path}"))

                    # Build payload — include all statistics fields
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


def _to_python(value: Any) -> Any:
    """Convert a value returned by IDSPrimitive.value to a JSON-serialisable
    Python object.  numpy scalars (np.int32, np.float64, …) are not handled
    by Flask's CustomEncoder, so we convert them here."""
    if isinstance(value, np.ndarray):
        # Flatten to a plain Python list so the dashboard receives a regular
        # JSON array (not the internal base64 binary encoding).
        flat = value.tolist()
        # Replace NaN/Inf with None (not valid JSON)
        def _clean(v):
            if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                return None
            if isinstance(v, list):
                return [_clean(x) for x in v]
            return v
        return _clean(flat)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, np.complexfloating):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _traverse_path(entry, ids_name: str, field_segments: list, occurrence: int):
    """Open *ids_name* from *entry* and walk *field_segments*, returning a
    tuple of (value, coordinate_path) where coordinate_path is the first
    non-time coordinate of the leaf node in slash/index form, or None.

    Each segment is either:
    - a non-negative integer string → array-of-structures index
    - a plain name → attribute access (IDSStructure child node)
    """
    ids_obj = entry.get(ids_name, occurrence, lazy=True, autoconvert=False, ignore_unknown_dd_version=True)
    node = ids_obj
    for segment in field_segments:
        if segment.isdigit():
            node = node[int(segment)]
        else:
            try:
                node = getattr(node, segment)
            except AttributeError:
                raise ValueError(f"segment '{segment}' not found in IDS path")
    if not isinstance(node, IDSPrimitive):
        raise ValueError(
            f"path does not point to a scalar/array leaf "
            f"(reached {type(node).__name__}); add more path segments"
        )

    # Derive the first non-time coordinate path (e.g. profiles_1d(itime)/grid/rho_tor_norm)
    # and convert it to a fetchable slash form using the same AoS indices as the request.
    coordinate_path = None
    try:
        for coord in node.metadata.coordinates:
            if not coord.is_time_coordinate:
                # coord str e.g. 'profiles_1d(itime)/grid/rho_tor_norm'
                raw = str(coord)
                # Replace (label) placeholders with the actual integer index from field_segments
                def _replace_placeholder(m, _segs=field_segments, _counter=[0]):
                    idx = next((s for s in _segs if s.isdigit()), '0')
                    return '/' + idx + '/'
                clean = re.sub(r'\([^)]+\)/', _replace_placeholder, raw)
                # Prepend ids_name
                coordinate_path = ids_name + '/' + clean
                break
    except Exception:
        pass

    return _to_python(node.value), coordinate_path


@api.route("/simulation/<path:sim_id>/data")
class SimulationImasData(Resource):
    @requires_auth()
    def get(self, sim_id: str, user: User):
        """Return the value at a given IDS path for a simulation's IMAS output.

        Query parameters
        ----------------
        path      (required) IDS path, e.g. ``core_profiles/profiles_1d/0/electrons/density``
        file_uuid (optional) UUID of a specific IMAS output file; defaults to the first one
        occurrence (optional) IDS occurrence index (default 0)
        """
        path = request.args.get("path", "").strip()
        if not path:
            return {"error": "Query parameter 'path' is required"}, 400

        file_uuid_str = request.args.get("file_uuid", "").strip() or None

        try:
            occurrence = int(request.args.get("occurrence", "0"))
        except ValueError:
            return {"error": "'occurrence' must be a non-negative integer"}, 400

        if occurrence < 0:
            return {"error": "'occurrence' must be a non-negative integer"}, 400

        # ------------------------------------------------------------------ #
        # Resolve simulation                                                   #
        # ------------------------------------------------------------------ #
        try:
            simulation = current_app.db.get_simulation(sim_id)
        except DatabaseError as exc:
            return {"error": str(exc)}, 404

        imas_outputs = [
            f for f in simulation.outputs if f.type == DataObject.Type.IMAS
        ]
        if not imas_outputs:
            return {"error": f"Simulation {sim_id} has no IMAS output files"}, 404

        # ------------------------------------------------------------------ #
        # Select target file                                                   #
        # ------------------------------------------------------------------ #
        if file_uuid_str:
            try:
                target_uuid = _uuid.UUID(file_uuid_str)
            except ValueError:
                return {"error": f"Invalid file_uuid: {file_uuid_str!r}"}, 400

            imas_file = next(
                (f for f in imas_outputs if f.uuid == target_uuid), None
            )
            if imas_file is None:
                return (
                    {
                        "error": (
                            f"File {file_uuid_str} not found or is not an IMAS "
                            "output for this simulation"
                        )
                    },
                    404,
                )
        else:
            imas_file = imas_outputs[0]

        # ------------------------------------------------------------------ #
        # Parse IDS path                                                       #
        # ------------------------------------------------------------------ #
        segments = [s for s in path.split("/") if s]
        if not segments:
            return {"error": "'path' must not be empty"}, 400

        ids_name = segments[0]
        field_segments = segments[1:]

        # ------------------------------------------------------------------ #
        # Open IMAS entry, traverse path, close                               #
        # ------------------------------------------------------------------ #
        try:
            entry = open_imas(imas_file.uri)
        except (ImasError, ValueError) as exc:
            return {"error": f"Failed to open IMAS data: {exc}"}, 500

        try:
            value, coordinate_path = _traverse_path(entry, ids_name, field_segments, occurrence)
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            return {"error": f"Invalid IDS path '{path}': {exc}"}, 400
        except Exception as exc:
            # Catches imas.exception.DataEntryException (empty IDS, wrong occurrence, etc.)
            # and any other IMAS-level errors.
            msg = str(exc)
            status = 404 if "is empty" in msg or "not found" in msg.lower() else 500
            return {"error": msg}, status
        finally:
            try:
                entry.close()
            except Exception:
                pass

        shape = list(np.asarray(value).shape) if isinstance(value, list) else None
        return {
            "simulation": str(simulation.uuid),
            "file_uuid": str(imas_file.uuid),
            "path": path,
            "occurrence": occurrence,
            "value": value,
            "shape": shape,
            "coordinate": coordinate_path,
        }


def _walk_nonempty(node, prefix: str, out: list) -> None:
    """Recursively collect non-empty leaf paths using iter_nonempty_().

    ``prefix`` tracks the slash-separated path from the IDS root.  For
    IDSStructArray elements the index is embedded as ``name[i]``.
    """
    for child in node.iter_nonempty_(accept_lazy=True):
        # Use the last segment of _path (iter_nonempty_ may return an IDSPath
        # whose str representation is the full absolute path from IDS root at
        # L1 but a relative segment at deeper levels — taking split[-1] is
        # safe in both cases).
        segment = str(child._path).split("/")[-1]
        full = f"{prefix}/{segment}" if prefix else segment

        if isinstance(child, IDSPrimitive):
            units = getattr(child.metadata, "units", "") or ""
            if units in ("?", "-"):
                units = ""
            description = getattr(child.metadata, "documentation", "") or ""
            try:
                shape = list(child.shape)
            except Exception:
                shape = None

            # -------------------------------------------------------------- #
            # Compute statistics from the already-loaded value so that Qdrant #
            # has rich payload for LLM-ready retrieval.                       #
            # -------------------------------------------------------------- #
            stats: dict = {}
            try:
                val = child.value
                if isinstance(val, np.ndarray) and val.size > 0:
                    is_float = np.issubdtype(val.dtype, np.floating)
                    finite_mask = np.isfinite(val) if is_float else np.ones(val.shape, dtype=bool)
                    clean = val[finite_mask]
                    n_nan = int(val.size - clean.size) if is_float else 0
                    stats["n_elements"] = int(val.size)
                    stats["n_nan"] = n_nan
                    if clean.size:
                        stats["min"] = float(clean.min())
                        stats["max"] = float(clean.max())
                        stats["mean"] = float(clean.mean())
                        stats["std"] = float(clean.std())
                    if val.ndim == 1 and val.size > 1:
                        diff = np.diff(val)
                        stats["is_monotonic"] = bool(
                            np.all(diff >= 0) or np.all(diff <= 0)
                        )
                    if val.ndim >= 2:
                        stats["sparsity"] = float(np.sum(val == 0) / val.size)
                elif np.isscalar(val) and not isinstance(val, str):
                    # Single numeric scalar — store the value directly
                    if isinstance(val, (np.integer,)):
                        stats["value"] = int(val)
                    elif isinstance(val, (np.floating,)):
                        fv = float(val)
                        if np.isfinite(fv):
                            stats["value"] = fv
                    elif isinstance(val, (int, float)) and np.isfinite(val):
                        stats["value"] = val
                elif isinstance(val, str) and val:
                    stats["value"] = val
            except Exception:
                pass  # statistics are best-effort; never break the walk

            out.append(
                {
                    "path": full,
                    "dtype": str(child.data_type),
                    "units": units,
                    "shape": shape,
                    "description": description,
                    **stats,
                }
            )
        elif isinstance(child, IDSStructArray):
            # Use max index (len-1, 0-based) in the path so the caller knows
            # the array size; inspect element [0] for structure.
            if len(child) > 0:
                max_idx = len(child) - 1
                _walk_nonempty(child[0], f"{full}[{max_idx}]", out)
        elif isinstance(child, IDSStructure):
            _walk_nonempty(child, full, out)


@api.route("/simulation/<path:sim_id>/fields")
class SimulationImasFields(Resource):
    @requires_auth()
    def get(self, sim_id: str, user: User):
        """List all non-empty IDS fields actually present in a simulation's IMAS output.

        Unlike ``GET /catalogue`` (which lists every field defined in the Data
        Dictionary), this endpoint inspects the simulation file itself and
        returns only fields that contain data.

        Query parameters
        ----------------
        ids        (optional) Restrict to a single IDS name, e.g. ``equilibrium``.
                   If omitted, all non-empty IDSes are scanned.
        occurrence (optional) Restrict to a single occurrence index.
                   If omitted, all occurrences of each IDS are scanned.
        file_uuid  (optional) UUID of a specific IMAS output file.
        """
        file_uuid_str = request.args.get("file_uuid", "").strip() or None
        ids_filter = request.args.get("ids", "").strip() or None
        occurrence_str = request.args.get("occurrence", "").strip()
        occurrence_filter: int | None = None
        if occurrence_str:
            try:
                occ_val = int(occurrence_str)
            except ValueError:
                return {"error": "'occurrence' must be a non-negative integer"}, 400
            if occ_val < 0:
                return {"error": "'occurrence' must be a non-negative integer"}, 400
            occurrence_filter = occ_val

        try:
            simulation = current_app.db.get_simulation(sim_id)
        except DatabaseError as exc:
            return {"error": str(exc)}, 404

        imas_outputs = [
            f for f in simulation.outputs if f.type == DataObject.Type.IMAS
        ]
        if not imas_outputs:
            return {"error": f"Simulation {sim_id} has no IMAS output files"}, 404

        if file_uuid_str:
            try:
                target_uuid = _uuid.UUID(file_uuid_str)
            except ValueError:
                return {"error": f"Invalid file_uuid: {file_uuid_str!r}"}, 400
            imas_file = next(
                (f for f in imas_outputs if f.uuid == target_uuid), None
            )
            if imas_file is None:
                return {"error": f"File {file_uuid_str} not found"}, 404
        else:
            imas_file = imas_outputs[0]

        # ── Cache hit ──────────────────────────────────────────────────── #
        cache_key = (str(imas_file.uuid), ids_filter, occurrence_filter)
        if cache_key in _fields_cache:
            return {
                "simulation": str(simulation.uuid),
                "file_uuid": str(imas_file.uuid),
                "ids": _fields_cache[cache_key],
                "cached": True,
            }

        try:
            import imas as _imas
            factory = _imas.IDSFactory()
            known_ids = [ids_filter] if ids_filter else list(factory.ids_names())

            def _scan_ids(ids_name: str) -> tuple[str, dict]:
                """Serialised via _lock — HDF5 does not support concurrent
                access from multiple threads on the same file handle."""
                with _lock:
                    try:
                        occurrences = entry.list_all_occurrences(ids_name)
                    except Exception:
                        return ids_name, {}
                    if occurrence_filter is not None:
                        occurrences = [o for o in occurrences if o == occurrence_filter]
                    ids_entry: dict = {}
                    for occ in occurrences:
                        try:
                            ids_obj = entry.get(
                                ids_name, int(occ),
                                lazy=False,
                                autoconvert=False,
                                ignore_unknown_dd_version=True,
                            )
                            fields: list = []
                            _walk_nonempty(ids_obj, "", fields)
                            if fields:
                                ids_entry[int(occ)] = fields
                        except Exception:
                            continue
                    return ids_name, ids_entry

            entry = open_imas(imas_file.uri)
            _lock = threading.Lock()
            result = {}
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = {pool.submit(_scan_ids, n): n for n in known_ids}
                    for fut in as_completed(futures):
                        n, ie = fut.result()
                        if ie:
                            result[n] = ie
            finally:
                try:
                    entry.close()
                except Exception:
                    pass
        except ImportError:
            return {"error": "IMAS Python library is not installed"}, 503
        except Exception as exc:
            return {"error": f"IMAS scan failed: {exc}"}, 500

        # ── Store in cache ─────────────────────────────────────────────── #
        _fields_cache[cache_key] = result

        # ── Index in Qdrant (background, non-blocking) ─────────────────── #
        threading.Thread(
            target=_index_in_qdrant,
            args=(str(simulation.uuid), str(imas_file.uuid), result),
            daemon=True,
        ).start()

        return {
            "simulation": str(simulation.uuid),
            "file_uuid": str(imas_file.uuid),
            "ids": result,
        }


@api.route("/simulation/<path:sim_id>/fields/search")
class SimulationImasFieldsSearch(Resource):
    @requires_auth()
    def get(self, sim_id: str, user: User):
        """Semantic search over the fields actually present in a simulation.

        Requires the ``/fields`` endpoint to have been called at least once so
        the field vectors are indexed in Qdrant. Results are ranked by
        cosine similarity to the natural-language query.

        Query parameters
        ----------------
        q          (required) Natural-language query, e.g. ``electron temperature profile``.
        file_uuid  (optional) UUID of a specific IMAS output file.
        limit      (optional) Maximum number of results (default 10).
        """
        q = request.args.get("q", "").strip()
        if not q:
            return {"error": "Query parameter 'q' is required"}, 400

        file_uuid_str = request.args.get("file_uuid", "").strip() or None
        try:
            limit = max(1, int(request.args.get("limit", "10")))
        except ValueError:
            limit = 10

        try:
            simulation = current_app.db.get_simulation(sim_id)
        except DatabaseError as exc:
            return {"error": str(exc)}, 404

        imas_outputs = [
            f for f in simulation.outputs if f.type == DataObject.Type.IMAS
        ]
        if not imas_outputs:
            return {"error": f"Simulation {sim_id} has no IMAS output files"}, 404

        if file_uuid_str:
            try:
                target_uuid = _uuid.UUID(file_uuid_str)
            except ValueError:
                return {"error": f"Invalid file_uuid: {file_uuid_str!r}"}, 400
            imas_file = next((f for f in imas_outputs if f.uuid == target_uuid), None)
            if imas_file is None:
                return {"error": f"File {file_uuid_str} not found"}, 404
        else:
            imas_file = imas_outputs[0]

        if not _init_qdrant():
            return {
                "error": "Vector search not available — Qdrant is not running or "
                         "sentence-transformers is not installed. "
                         "Call GET /fields first to index, then retry."
            }, 503

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            flt = Filter(must=[
                FieldCondition(
                    key="file_uuid",
                    match=MatchValue(value=str(imas_file.uuid)),
                )
            ])

            # ── Auto-index if this file has never been indexed ─────────── #
            count = _qdrant_client.count(
                collection_name=_QDRANT_COLLECTION,
                count_filter=flt,
                exact=True,
            ).count
            if count == 0:
                # Run fields scan + indexing synchronously so vectors exist
                # before we search. Reuses the same logic as GET /fields.
                cache_key = (str(imas_file.uuid), None, None)
                if cache_key in _fields_cache:
                    ids_result = _fields_cache[cache_key]
                else:
                    try:
                        _entry = open_imas(imas_file.uri)
                    except (ImasError, ValueError) as exc:
                        return {"error": f"Auto-index: failed to open IMAS data: {exc}"}, 500
                    try:
                        import imas as _imas
                        factory = _imas.IDSFactory()
                        known_ids = list(factory.ids_names())
                        _use_fp = False
                        try:
                            _entry.list_filled_paths(known_ids[0], 0, autoconvert=False)
                            _use_fp = True
                        except Exception:
                            pass

                        def _meta(obj, p):
                            node = obj
                            for seg in p.split("/"):
                                try:
                                    node = getattr(node, seg)
                                except Exception:
                                    break
                            units = getattr(getattr(node, "metadata", None), "units", "") or ""
                            if units in ("?", "-"):
                                units = ""
                            desc = getattr(getattr(node, "metadata", None), "documentation", "") or ""
                            dtype = str(getattr(node, "data_type", "")) if hasattr(node, "data_type") else ""
                            return {"units": units, "description": desc, "dtype": dtype}

                        def _scan(n):
                            try:
                                occs = _entry.list_all_occurrences(n)
                            except Exception:
                                return n, {}
                            ie: dict = {}
                            for o in occs:
                                try:
                                    if _use_fp:
                                        rp = _entry.list_filled_paths(n, int(o), autoconvert=False)
                                        obj = _entry.get(n, int(o), lazy=True, autoconvert=False, ignore_unknown_dd_version=True)
                                        fs = [{"path": p, **_meta(obj, p)} for p in rp]
                                    else:
                                        obj = _entry.get(n, int(o), lazy=False, autoconvert=False, ignore_unknown_dd_version=True)
                                        fs = []
                                        _walk_nonempty(obj, "", fs)
                                    if fs:
                                        ie[int(o)] = fs
                                except Exception:
                                    continue
                            return n, ie

                        ids_result = {}
                        with ThreadPoolExecutor(max_workers=8) as pool:
                            for n, ie in pool.map(_scan, known_ids):
                                if ie:
                                    ids_result[n] = ie
                    finally:
                        try:
                            _entry.close()
                        except Exception:
                            pass
                    _fields_cache[cache_key] = ids_result

                # Index synchronously (blocking) so vectors are ready to search
                _index_in_qdrant(str(simulation.uuid), str(imas_file.uuid), ids_result)

            vec = _embedder.encode(q, show_progress_bar=False).tolist()
            response = _qdrant_client.query_points(
                collection_name=_QDRANT_COLLECTION,
                query=vec,
                query_filter=flt,
                limit=limit,
            )
            hits = response.points
        except Exception as exc:
            return {"error": f"Vector search failed: {exc}"}, 500

        return {
            "simulation": str(simulation.uuid),
            "file_uuid": str(imas_file.uuid),
            "query": q,
            "results": [
                {"score": round(h.score, 4), **h.payload}
                for h in hits
            ],
        }
