"""GET /v1.2/search — natural-language IMAS DD field search via the IMAS MCP server.

Calls ``search_dd_paths`` on the remote IMAS MCP HTTP server and returns a
ranked JSON list of matching IDS fields.

Query parameters
----------------
q           (required) Natural-language query, e.g. ``electron temperature``
ids         (optional) Restrict results to a single IDS, e.g. ``core_profiles``
domain      (optional) Physics domain filter, e.g. ``transport``
limit       (optional) Max results to return (default 20, max 50)
dd_version  (optional) DD version string passed as context hint to the MCP
            server (informational only — the MCP server resolves its own
            versioned index).  Defaults to the latest version.

Response
--------
::

    {
        "results": [
            {
                "path":        "core_profiles/profiles_1d/electrons/temperature",
                "ids":         "core_profiles",
                "dtype":       "FLT_1D",
                "units":       "eV",
                "description": "Electron temperature (Te) 1D radial profile.",
                "score":       0.92
            },
            ...
        ],
        "dd_version": "4.1.0",
        "query": "electron temperature"
    }

No authentication is required.  Results are cached in-process with
``lru_cache`` keyed on ``(query, ids, domain, limit)``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from flask import request
from flask_restx import Namespace, Resource

log = logging.getLogger(__name__)

api = Namespace("search", path="/")

_IMAS_MCP_URL = "https://imas-dd.iter.org/mcp"

# ---------------------------------------------------------------------------
# Response parser (MCP returns rendered Markdown text, not JSON)
# ---------------------------------------------------------------------------

def _parse_search_response(text: str) -> List[Dict[str, Any]]:
    """Parse the Markdown-formatted ``search_dd_paths`` response into a list
    of structured dicts."""
    results = []
    blocks = re.split(r"\n### ", text)
    for block in blocks[1:]:  # skip header line
        lines = block.strip().split("\n")
        header = lines[0]
        m = re.match(r"^(.+?)\s+\(score:\s*([\d.]+)\)", header)
        if not m:
            continue
        path = m.group(1).strip()
        score = float(m.group(2))
        ids_name = path.split("/")[0]
        description = ""
        dtype = ""
        units = ""
        for line in lines[1:]:
            line = line.strip()
            dm = re.match(r'^"(.+?)"', line)
            if dm:
                description = dm.group(1)
            tm = re.search(r"Type:\s*(\S+)", line)
            if tm:
                dtype = tm.group(1)
            um = re.search(r"Unit:\s*([^|]+)", line)
            if um:
                units = um.group(1).strip()
            im = re.search(r"IDS:\s*(\S+)", line)
            if im:
                ids_name = im.group(1).rstrip("|").strip()
        results.append(
            {
                "path": path,
                "ids": ids_name,
                "dtype": dtype,
                "units": units,
                "description": description,
                "score": score,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Async MCP caller
# ---------------------------------------------------------------------------

async def _call_mcp_search(
    query: str,
    ids_filter: Optional[str],
    domain: Optional[str],
    limit: int,
) -> str:
    """Open an MCP session, call ``search_dd_paths``, return the raw text."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    params: Dict[str, Any] = {"query": query, "k": limit}
    if ids_filter:
        params["ids_filter"] = ids_filter
    if domain:
        params["physics_domain"] = domain

    async with streamablehttp_client(_IMAS_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_dd_paths", params)
            return result.content[0].text


# ---------------------------------------------------------------------------
# Per-request cache (keyed on the logical query, not the HTTP request object)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_search(
    query: str,
    ids_filter: Optional[str],
    domain: Optional[str],
    limit: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run the MCP search synchronously (blocking) and parse the response.

    Returns ``(results, error_message)``.  ``error_message`` is ``""`` on
    success.
    """
    try:
        raw = asyncio.run(
            _call_mcp_search(query, ids_filter, domain, limit)
        )
        return _parse_search_response(raw), ""
    except Exception as exc:
        log.exception("IMAS MCP search failed for query=%r", query)
        return [], str(exc)


# ---------------------------------------------------------------------------
# Flask endpoint
# ---------------------------------------------------------------------------

@api.route("/search")
class Search(Resource):
    @api.doc(security=[])  # public — no token required
    def get(self):
        """Search IMAS DD fields by natural language via the IMAS MCP server."""
        query = request.args.get("q", "").strip()
        if not query:
            return {"error": "Query parameter 'q' is required"}, 400

        ids_filter = request.args.get("ids", "").strip() or None
        domain = request.args.get("domain", "").strip() or None

        try:
            limit = int(request.args.get("limit", "20"))
        except ValueError:
            return {"error": "'limit' must be a positive integer"}, 400
        limit = max(1, min(limit, 50))

        dd_version = request.args.get("dd_version", "").strip() or None

        results, err = _cached_search(query, ids_filter, domain, limit)

        if err:
            return {"error": f"IMAS MCP search failed: {err}"}, 502

        return {
            "results": results,
            "dd_version": dd_version,
            "query": query,
        }
