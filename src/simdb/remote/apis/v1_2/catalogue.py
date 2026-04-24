"""GET /v1.2/catalogue — flat union of all IMAS DD leaf fields.

Returns every data-dictionary field that has ever existed across **all**
available DD versions, so users can look up fields regardless of which DD
version their simulation data was written with.

Returns a JSON array of objects::

    [
        {
            "path":          "core_profiles/profiles_1d[]/electrons/temperature",
            "ids":           "core_profiles",
            "dtype":         "FLT_1D",
            "units":         "eV",
            "description":   "Electron temperature (Te) 1D radial profile.",
            "introduced_in": "3.22.0",
            "domain":        "transport",
            "ids_rank":      1,
        },
        ...
    ]

Entries are sorted by ``ids_rank`` (most-used IDS first, from the IMAS MCP
catalog) then alphabetically by path within each IDS.

The full catalogue (~45 k paths) is built once on first request and cached
for the lifetime of the Flask process.  A background thread starts warming
the cache as soon as this module is imported.  No authentication is required.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from flask_restx import Namespace, Resource

log = logging.getLogger(__name__)

api = Namespace("catalogue", path="/")

# Data-type prefixes that identify leaf (non-container) IDS fields.
_LEAF_PREFIXES = frozenset({"STR", "INT", "FLT", "CPX"})


def _is_leaf(data_type: str) -> bool:
    return any(data_type.startswith(p) for p in _LEAF_PREFIXES)


# ── IDS usage ranking from IMAS MCP ──────────────────────────────────────────

_IMAS_MCP_URL = "https://imas-dd.iter.org/mcp"


async def _fetch_ids_catalog() -> List[Tuple[str, int, str]]:
    """Return list of (ids_name, path_count, domain) sorted by path_count desc."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    async with streamablehttp_client(_IMAS_MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            result = await s.call_tool("get_dd_catalog", {})
            text = result.content[0].text

    entries = []
    for m in re.finditer(
        r"^\s{2}(\w+)\s+\((\d+)\s+paths\)\s+\[([^\]]+)\]",
        text,
        re.MULTILINE,
    ):
        entries.append((m.group(1), int(m.group(2)), m.group(3)))
    entries.sort(key=lambda x: -x[1])
    return entries


@lru_cache(maxsize=1)
def _ids_ranking() -> Dict[str, Tuple[int, str]]:
    """Return {ids_name: (rank, domain)}, rank 1 = most paths.

    Falls back to an empty dict if the MCP server is unreachable.
    """
    try:
        entries = asyncio.run(_fetch_ids_catalog())
        return {name: (i + 1, domain) for i, (name, _, domain) in enumerate(entries)}
    except Exception:
        log.warning("Could not fetch IDS ranking from IMAS MCP — using unranked order")
        return {}


# ── DD XML parsing ────────────────────────────────────────────────────────────

def _parse_version(dd_version: str) -> Dict[str, Dict[str, Any]]:
    """Return {path: entry_dict} for a single DD version (no caching)."""
    import imas.dd_zip as ddz

    tree = ddz.dd_etree(dd_version)
    root = tree.getroot()
    result: Dict[str, Dict[str, Any]] = {}

    def _walk(element, path_prefix: str, ids_name: str) -> None:
        for child in element:
            if child.tag != "field":
                continue
            name = child.get("name", "")
            if not name:
                continue
            data_type = child.get("data_type", "")
            units = child.get("units") or ""
            if units in ("?", "-"):
                units = ""
            # Append [] to struct_array segments so the dashboard knows an
            # integer index is required (e.g. profiles_1d[] → profiles_1d[0]).
            segment = f"{name}[]" if data_type == "struct_array" else name
            full_path = f"{path_prefix}/{segment}"
            doc_el = child.find("documentation")
            description = (doc_el.text or "").strip() if doc_el is not None else ""

            if _is_leaf(data_type):
                result[full_path] = {
                    "path": full_path,
                    "ids": ids_name,
                    "dtype": data_type,
                    "units": units,
                    "description": description,
                }
            else:
                _walk(child, full_path, ids_name)

    for ids_el in root.findall("IDS"):
        ids_name = ids_el.get("name", "")
        if not ids_name:
            continue
        _walk(ids_el, ids_name, ids_name)

    return result


@lru_cache(maxsize=1)
def _build_full_catalogue() -> List[Dict[str, Any]]:
    """Build and cache the union catalogue across **all** available DD versions.

    Iterates versions chronologically so that ``introduced_in`` reflects the
    earliest version in which a field appeared.  Annotates each entry with
    ``domain`` and ``ids_rank`` from the IMAS MCP catalog, then sorts by rank.
    """
    import imas.dd_zip as ddz

    versions = sorted(ddz.dd_xml_versions())
    union: Dict[str, Dict[str, Any]] = {}

    for ver in versions:
        try:
            ver_paths = _parse_version(ver)
        except Exception:
            log.warning("Skipping DD version %s — failed to parse", ver)
            continue
        for path, entry in ver_paths.items():
            if path not in union:
                entry["introduced_in"] = ver
                union[path] = entry

    ranking = _ids_ranking()  # {ids_name: (rank, domain)}

    # Annotate and sort: ranked IDSes first (by rank asc), unranked last
    _MAX_RANK = len(ranking) + 1
    for entry in union.values():
        rank, domain = ranking.get(entry["ids"], (_MAX_RANK, ""))
        entry["ids_rank"] = rank
        entry["domain"] = domain

    return sorted(
        union.values(),
        key=lambda e: (e["ids_rank"], e["ids"], e["path"]),
    )


# ── Pre-warm cache in a background thread the moment this module loads ────────
def _prewarm() -> None:
    try:
        _ids_ranking()  # fetch MCP ranking first
        _build_full_catalogue()
        log.info(
            "IMAS DD catalogue pre-warm complete (%d paths)",
            len(_build_full_catalogue()),
        )
    except Exception:
        log.exception("IMAS DD catalogue pre-warm failed")


threading.Thread(target=_prewarm, daemon=True, name="catalogue-prewarm").start()


@api.route("/catalogue")
class Catalogue(Resource):
    @api.doc(security=[])  # public — no token required
    def get(self):
        """Return all IMAS DD leaf fields (union of all DD versions) as a flat JSON array.

        Each entry includes ``path``, ``ids``, ``dtype``, ``units``,
        ``description``, and ``introduced_in`` (earliest DD version that
        contains the field).
        """
        try:
            catalogue = _build_full_catalogue()
        except Exception as exc:
            log.exception("Failed to build full catalogue")
            return {"error": f"Failed to build catalogue: {exc}"}, 500

        return catalogue
