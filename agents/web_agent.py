"""
web_agent.py — Mr Meeseeks Web Agent
Implements simple_scrape and gui_research.
"""

import logging
import httpx
import urllib.parse
import re

from core.ipc_bus import bus

log = logging.getLogger("web_agent")


async def handle_simple_scrape(args: dict) -> dict:
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing 'query' argument."}
        
    log.info(f"Scraping simple info for: {query}")
    try:
        # Use Wikipedia API for simple factual scraping (simulates quick AI answers)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={"User-Agent": "MrMeeseeksBot/1.0"}, timeout=10)
            data = resp.json()
            
        search_results = data.get('query', {}).get('search', [])
        if not search_results:
            return {"result": f"No quick answers found for '{query}'."}
            
        # Get top 2 results and clean HTML tags
        snippets = []
        for item in search_results[:2]:
            title = item.get('title', '')
            raw_snippet = item.get('snippet', '')
            clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet)
            # Remove extra html entities like &quot;
            import html
            clean_snippet = html.unescape(clean_snippet)
            snippets.append(f"{title}: {clean_snippet}")
            
        summary = "\n".join(snippets)
        
        # Build local RAG base by storing this in memory
        await bus.dispatch("update_memory", {"key": f"{query.replace(' ', '_')}", "data": summary})
        
        return {"result": summary, "note": "Data stored in memory."}

    except Exception as e:
        log.error(f"simple_scrape failed: {e}")
        return {"error": f"Failed to fetch data: {str(e)}"}


async def handle_gui_research(args: dict) -> dict:
    return {"error": "gui_research (Eyes/Hands automation) is not implemented in Sprint 1."}


def register():
    bus.register("simple_scrape", handle_simple_scrape)
    bus.register("gui_research", handle_gui_research)
    log.info("Web agent registered ✓")
