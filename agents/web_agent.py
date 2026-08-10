"""
web_agent.py — Mr Meeseeks Web Agent
Uses Ollama's native web_search and web_fetch APIs (ollama>=0.6.0).
Falls back to DuckDuckGo HTML scrape if OLLAMA_API_KEY not set.
"""
import logging
import os
import asyncio

from core.ipc_bus import bus

log = logging.getLogger("web_agent")


async def _ollama_web_search(query: str) -> str | None:
    """Use Ollama's native web_search API (requires OLLAMA_API_KEY)."""
    try:
        from ollama import web_search
        result = await asyncio.to_thread(web_search, query)
        if result and hasattr(result, 'results') and result.results:
            snippets = []
            for r in result.results[:5]:
                title = getattr(r, 'title', '')
                content = getattr(r, 'content', '')
                if title and content:
                    snippets.append(f"- {title}: {content[:300]}")
            return "\n".join(snippets) if snippets else None
        return None
    except Exception as e:
        log.debug(f"Ollama web_search failed: {e}")
        return None


async def _ollama_web_fetch(url: str) -> str | None:
    """Use Ollama's native web_fetch API to fetch a specific URL."""
    try:
        from ollama import web_fetch
        result = await asyncio.to_thread(web_fetch, url)
        if result:
            return getattr(result, 'content', str(result))[:3000]
        return None
    except Exception as e:
        log.debug(f"Ollama web_fetch failed: {e}")
        return None


async def _ddg_fallback(query: str) -> str | None:
    """DuckDuckGo HTML scrape fallback when OLLAMA_API_KEY not set."""
    import httpx
    import urllib.parse
    from bs4 import BeautifulSoup
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
        }
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("div", class_="result")
        snippets = []
        for res in results[:3]:
            title_elem = res.find("a", class_="result__url")
            snippet_elem = res.find("a", class_="result__snippet")
            if title_elem and snippet_elem:
                snippets.append(f"- {title_elem.get_text(strip=True)}: {snippet_elem.get_text(strip=True)}")
        return "\n".join(snippets) if snippets else None
    except Exception as e:
        log.debug(f"DDG fallback failed: {e}")
        return None


async def handle_simple_scrape(args: dict) -> dict:
    """Search the web. Uses Ollama web_search if API key available, else DDG."""
    query = args.get("query", "").strip()
    url = args.get("url", "").strip()  # optionally fetch a specific URL
    if not query and not url:
        return {"error": "Missing 'query' or 'url' argument."}

    if url:
        log.info(f"Fetching URL: {url}")
        result = await _ollama_web_fetch(url)
        source = "ollama.web_fetch"
        if not result:
            return {"result": f"Could not fetch URL: {url}"}
        return {"result": result, "source": source}

    log.info(f"Searching: {query}")

    # Check semantic web cache in ChromaDB first
    try:
        from core.chroma_store import chroma_store
        cached = await asyncio.to_thread(chroma_store.get_web_cache, query)
        if cached:
            return {"result": cached, "source": "chroma_web_cache"}
    except Exception as e:
        log.debug(f"Web cache lookup failed: {e}")

    result = None
    source = "unknown"

    # Try Ollama native web search first
    if os.environ.get("OLLAMA_API_KEY"):
        result = await _ollama_web_search(query)
        source = "ollama.web_search"

    # Fall back to DuckDuckGo
    if not result:
        result = await _ddg_fallback(query)
        source = "DuckDuckGo"

    if not result:
        return {"result": f"No results found for '{query}'."}

    # Save to semantic web cache asynchronously (fire-and-forget)
    try:
        from core.chroma_store import chroma_store
        asyncio.create_task(asyncio.to_thread(chroma_store.save_web_cache, query, result))
    except Exception:
        pass

    return {"result": result, "source": source}


async def handle_web_fetch(args: dict) -> dict:
    """Fetch content from a specific URL."""
    url = args.get("url", "").strip()
    if not url:
        return {"error": "Missing 'url' argument."}
    result = await _ollama_web_fetch(url)
    if not result:
        return {"result": f"Could not fetch URL: {url}"}
    return {"result": result, "source": "ollama.web_fetch"}


async def handle_gui_research(args: dict) -> dict:
    return {"error": "gui_research (browser automation) not yet implemented."}


def register():
    bus.register("simple_scrape", handle_simple_scrape)
    bus.register("web_fetch", handle_web_fetch)
    bus.register("gui_research", handle_gui_research)
    log.info("Web agent registered (Ollama web_search + DDG fallback) ✓")
