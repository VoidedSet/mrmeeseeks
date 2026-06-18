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


from bs4 import BeautifulSoup

async def _duckduckgo_search(query: str) -> str | None:
    """Scrape real DuckDuckGo HTML results."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
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
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                snippets.append(f"- {title}: {snippet}")
                
        if snippets:
            return "\n".join(snippets)
        return None
    except Exception as e:
        log.debug(f"DDG HTML failed: {e}")
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
