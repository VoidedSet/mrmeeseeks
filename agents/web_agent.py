"""
web_agent.py — Mr Meeseeks Web Agent
DuckDuckGo Instant Answers (primary) + Wikipedia snippets (fallback).
"""
import logging
import httpx
import urllib.parse
import re
import html

from core.ipc_bus import bus

log = logging.getLogger("web_agent")


async def _duckduckgo_search(query: str) -> str | None:
    """DuckDuckGo Instant Answers API — free, no key required."""
    try:
        url = (
            "https://api.duckduckgo.com/"
            f"?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        )
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, headers={"User-Agent": "MrMeeseeksBot/1.0"})
            data = resp.json()

        # AbstractText = direct answer paragraph
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            source = data.get("AbstractSource", "")
            return f"{abstract}" + (f" (via {source})" if source else "")

        # RelatedTopics = list of relevant snippets
        topics = data.get("RelatedTopics", [])
        snippets = []
        for t in topics[:3]:
            if isinstance(t, dict) and t.get("Text"):
                snippets.append(t["Text"].strip())
        if snippets:
            return "\n".join(snippets)

        return None
    except Exception as e:
        log.debug(f"DDG failed: {e}")
        return None


async def _wikipedia_search(query: str) -> str | None:
    """Wikipedia search API fallback."""
    try:
        url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(query)}"
            "&utf8=&format=json&srlimit=2"
        )
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, headers={"User-Agent": "MrMeeseeksBot/1.0"})
            data = resp.json()

        results = data.get("query", {}).get("search", [])
        if not results:
            return None

        snippets = []
        for item in results[:2]:
            title   = item.get("title", "")
            snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
            snippets.append(f"{title}: {snippet}")
        return "\n".join(snippets)
    except Exception as e:
        log.debug(f"Wikipedia failed: {e}")
        return None


async def handle_simple_scrape(args: dict) -> dict:
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing 'query' argument."}

    log.info(f"Searching: {query}")

    # Try DuckDuckGo first
    result = await _duckduckgo_search(query)
    source = "DuckDuckGo"

    # Fall back to Wikipedia
    if not result:
        result = await _wikipedia_search(query)
        source = "Wikipedia"

    if not result:
        return {"result": f"No results found for '{query}'."}

    # Store in memory for future recall
    try:
        await bus.dispatch("update_memory", {
            "key":  query.replace(" ", "_")[:40],
            "data": result,
        })
    except Exception:
        pass

    return {"result": result, "source": source}


async def handle_gui_research(args: dict) -> dict:
    return {"error": "gui_research (browser automation) not yet implemented."}


def register():
    bus.register("simple_scrape", handle_simple_scrape)
    bus.register("gui_research",  handle_gui_research)
    log.info("Web agent registered ✓")
