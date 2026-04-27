"""Simulation IMAS data endpoints: /data, /fields, /fields/search."""
import re
import threading
import uuid as _uuid
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

from . import qdrant_store

api = Namespace("data", path="/")

# In-memory fields cache
# Key: (file_uuid_str, ids_filter, occurrence_filter) → ids result dict.
# Files are immutable so entries never expire.
_fields_cache: dict[tuple, dict] = {}


# Helperss


def _to_python(value: Any) -> Any:
    """Convert a value returned by IDSPrimitive.value to a JSON-serialisable
    Python object.  numpy scalars (np.int32, np.float64, …) are not handled
    by Flask's CustomEncoder, so we convert them here."""
    if isinstance(value, np.ndarray):
        flat = value.tolist()

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

#TODO Replace this logic with slicing when supported by imas-python
#TODO Add support for [:], [:-1] [2:4:2] python slicing syntax
def _traverse_path(entry, ids_name: str, field_segments: list, occurrence: int):
    """Walk *field_segments* inside *ids_name* and return (value, coordinate_path).

    Each segment is either:
    - a non-negative integer string → array-of-structures index
    - a plain name → attribute access (IDSStructure child node)
    """
    ids_obj = entry.get(
        ids_name, occurrence,
        lazy=True, autoconvert=False, ignore_unknown_dd_version=True,
    )
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

    coordinate_path = None
    try:
        for coord in node.metadata.coordinates:
            if not coord.is_time_coordinate:
                raw = str(coord)

                def _replace_placeholder(m, _segs=field_segments):
                    idx = next((s for s in _segs if s.isdigit()), "0")
                    return "/" + idx + "/"

                clean = re.sub(r"\([^)]+\)/", _replace_placeholder, raw)
                coordinate_path = ids_name + "/" + clean
                break
    except Exception:
        pass

    return _to_python(node.value), coordinate_path


def _walk_nonempty(node, prefix: str, out: list) -> None:
    """Recursively collect non-empty leaf paths using iter_nonempty_().

    ``prefix`` tracks the slash-separated path from the IDS root.  For
    IDSStructArray elements the index is embedded as ``name[i]``.
    Each appended dict includes path, dtype, units, shape, description, and
    any computable statistics (value/min/max/mean/std/n_elements/n_nan/
    is_monotonic/sparsity).
    """
    for child in node.iter_nonempty_(accept_lazy=True):
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

            stats: dict = {}
            try:
                val = child.value
                if isinstance(val, np.ndarray) and val.size > 0:
                    is_float = np.issubdtype(val.dtype, np.floating)
                    finite_mask = (
                        np.isfinite(val) if is_float else np.ones(val.shape, dtype=bool)
                    )
                    clean = val[finite_mask]
                    n_nan = int(val.size - clean.size) if is_float else 0
                    stats["n_elements"] = int(val.size)
                    stats["n_nan"] = n_nan
                    if clean.size:
                        stats["min"] = float(clean.min())
                        stats["max"] = float(clean.max())
                        stats["mean"] = float(clean.mean())
                        stats["std"] = float(clean.std())
                    # TODO: Add more LLM-friendly field descriptors here, e.g.
                    # monotonic direction, fraction_zero, sign_pattern,
                    # constant_like, normalized_like, boolean_like, and a short
                    # semantic_summary string for retrieval/ranking.
                    if val.ndim == 1 and val.size > 1:
                        diff = np.diff(val)
                        stats["is_monotonic"] = bool(
                            np.all(diff >= 0) or np.all(diff <= 0)
                        )
                    if val.ndim >= 2:
                        stats["sparsity"] = float(np.sum(val == 0) / val.size)
                elif np.isscalar(val) and not isinstance(val, str):
                    if isinstance(val, np.integer):
                        stats["value"] = int(val)
                    elif isinstance(val, np.floating):
                        fv = float(val)
                        if np.isfinite(fv):
                            stats["value"] = fv
                    elif isinstance(val, (int, float)) and np.isfinite(val):
                        stats["value"] = val
                elif isinstance(val, str) and val:
                    stats["value"] = val
            except Exception:
                pass  # its ok if statistics are failed

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
            if len(child) > 0:
                max_idx = len(child) - 1
                _walk_nonempty(child[0], f"{full}[{max_idx}]", out)
        elif isinstance(child, IDSStructure):
            _walk_nonempty(child, full, out)


def _scan_imas_file(
    imas_file,
    ids_filter: str | None = None,
    occurrence_filter: int | None = None,
) -> dict:
    """Scan all non-empty IDS fields from *imas_file* and return an ids_result dict.

    Returns ``{ids_name: {occurrence_int: [field_dict, ...]}}``.  Each
    field_dict is produced by :func:`_walk_nonempty` and includes statistics.

    Raises :class:`ImportError` if the IMAS library is unavailable; propagates
    any other exception so callers can return the appropriate HTTP error.
    """
    import imas as _imas  # ImportError propagates to caller

    factory = _imas.IDSFactory()
    #TODO: IMAS data entries can have different version of data dictionary and here if take only latest
    # we might miss some IDSes
    known_ids = [ids_filter] if ids_filter else list(factory.ids_names())

    entry = open_imas(imas_file.uri)
    result: dict = {}
    try:
        for ids_name in known_ids:
            try:
                occurrences = entry.list_all_occurrences(ids_name)
            except Exception:
                continue
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
            if ids_entry:
                result[ids_name] = ids_entry
    finally:
        try:
            entry.close()
        except Exception:
            pass

    return result


def _get_simulation_and_imas_file(sim_id: str, file_uuid_str: str | None):
    try:
        simulation = current_app.db.get_simulation(sim_id)
    except DatabaseError as exc:
        return None, None, ({"error": str(exc)}, 404)

    imas_outputs = [f for f in simulation.outputs if f.type == DataObject.Type.IMAS]
    if not imas_outputs:
        return None, None, (
            {"error": f"Simulation {sim_id} has no IMAS output files"}, 404
        )

    if not file_uuid_str:
        return simulation, imas_outputs[0], None

    try:
        target_uuid = _uuid.UUID(file_uuid_str)
    except ValueError:
        return None, None, ({"error": f"Invalid file_uuid: {file_uuid_str!r}"}, 400)

    imas_file = next((f for f in imas_outputs if f.uuid == target_uuid), None)
    if imas_file is None:
        return None, None, ({"error": f"File {file_uuid_str} not found"}, 404)

    return simulation, imas_file, None


# Endpoints 

@api.route("/simulation/<path:sim_id>/data")
class SimulationImasData(Resource):
    @requires_auth()
    def get(self, sim_id: str, user: User):
        """Return the value at a given IDS path for a simulation's IMAS output.

        Query parameters
        ----------------
        path       (required) IDS path, e.g. ``core_profiles/profiles_1d/0/electrons/density``
        file_uuid  (optional) UUID of a IMAS output file
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

        simulation, imas_file, error = _get_simulation_and_imas_file(
            sim_id, file_uuid_str
        )
        if error:
            payload, status = error
            if file_uuid_str and status == 404 and "File " in payload["error"]:
                return (
                    {
                        "error": (
                            f"File {file_uuid_str} not found or is not an IMAS "
                            "output for this simulation"
                        )
                    },
                    404,
                )
            return payload, status

        segments = [s for s in path.split("/") if s]
        if not segments:
            return {"error": "'path' must not be empty"}, 400

        ids_name = segments[0]
        field_segments = segments[1:]

        try:
            entry = open_imas(imas_file.uri)
        except (ImasError, ValueError) as exc:
            return {"error": f"Failed to open IMAS data: {exc}"}, 500

        try:
            value, coordinate_path = _traverse_path(
                entry, ids_name, field_segments, occurrence
            )
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            return {"error": f"Invalid IDS path '{path}': {exc}"}, 400
        except Exception as exc:
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


@api.route("/simulation/<path:sim_id>/fields")
class SimulationImasFields(Resource):
    @requires_auth()
    def get(self, sim_id: str, user: User):
        """List all non-empty IDS fields actually present in a simulation's IMAS output.

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

        simulation, imas_file, error = _get_simulation_and_imas_file(
            sim_id, file_uuid_str
        )
        if error:
            return error

        # ── Cache hit ──────────────────────────────────────────────────────── #
        cache_key = (str(imas_file.uuid), ids_filter, occurrence_filter)
        if cache_key in _fields_cache:
            return {
                "simulation": str(simulation.uuid),
                "file_uuid": str(imas_file.uuid),
                "ids": _fields_cache[cache_key],
                "cached": True,
            }

        try:
            result = _scan_imas_file(imas_file, ids_filter, occurrence_filter)
        except ImportError:
            return {"error": "IMAS Python library is not installed"}, 503
        except Exception as exc:
            return {"error": f"IMAS scan failed: {exc}"}, 500

        # ── Store in cache ─────────────────────────────────────────────────── #
        _fields_cache[cache_key] = result

        # ── Index in Qdrant (background, non-blocking) ────────────────────── #
        threading.Thread(
            target=qdrant_store._index_in_qdrant,
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

        If the Qdrant index is empty for this file, a full scan is triggered
        automatically (same path as ``GET /fields``) so the first call is
        self-contained.

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

        simulation, imas_file, error = _get_simulation_and_imas_file(
            sim_id, file_uuid_str
        )
        if error:
            return error

        if not qdrant_store._init_qdrant():
            return {
                "error": (
                    f"Vector search not available: "
                    f"{qdrant_store._get_qdrant_unavailable_reason()}"
                )
            }, 503

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            flt = Filter(
                must=[
                    FieldCondition(
                        key="file_uuid",
                        match=MatchValue(value=str(imas_file.uuid)),
                    )
                ]
            )

            # ── Auto-index if this file has never been indexed ───────────── #
            count = qdrant_store._qdrant_client.count(
                collection_name=qdrant_store._QDRANT_COLLECTION,
                count_filter=flt,
                exact=True,
            ).count
            if count == 0:
                # Try to get cache entry first; if absent, run a full
                # scan using the same code path as GET /fields 
                full_cache_key = (str(imas_file.uuid), None, None)
                if full_cache_key in _fields_cache:
                    ids_result = _fields_cache[full_cache_key]
                else:
                    try:
                        ids_result = _scan_imas_file(imas_file)
                    except ImportError:
                        return {"error": "IMAS Python library is not installed"}, 503
                    except Exception as exc:
                        return {"error": f"Auto-index scan failed: {exc}"}, 500
                    _fields_cache[full_cache_key] = ids_result

                # Synchronous so vectors exist before the search below
                qdrant_store._index_in_qdrant(
                    str(simulation.uuid), str(imas_file.uuid), ids_result
                )

            vec = qdrant_store.encode_query(q)
            response = qdrant_store._qdrant_client.query_points(
                collection_name=qdrant_store._QDRANT_COLLECTION,
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
            "results": [{"score": round(h.score, 4), **h.payload} for h in hits],
        }
