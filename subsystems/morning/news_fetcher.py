"""
news_fetcher.py — Morning news headlines using Ollama web_search or DuckDuckGo fallback.
Fetches top headlines for user's configured interests:
- Football (Barca, Messi, Fabrizio Romano)
- Tech scene & advancements
- Global geopolitical situation
- Regional alerts / emergencies in Delhi / India
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

log = logging.getLogger("news_fetcher")

# Mapping from broad interest category to search query
NEWS_QUERIES = {
    "football": "FC Barcelona Barca Messi latest news Fabrizio Romano today",
    "messi": "Lionel Messi latest news match updates today",
    "tech": "latest artificial intelligence tech scene advancements news today",
    "geopolitics": "global geopolitical situation major world news today",
    "alerts": "Delhi India weather alerts emergency news today",
}


async def _web_search(query: str) -> list[dict]:
    """
    Execute web search using Ollama or DuckDuckGo fallback.
    Returns list of {title, content, url} dicts.
    """
    try:
        from agents.web_agent import handle_simple_scrape
        res = await handle_simple_scrape({"query": query})
        result_text = res.get("result", "")
        if not result_text or "No results found" in result_text or "error" in res:
            return []

        # Parse snippet lines
        items = []
        for line in result_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                parts = line[2:].split(":", 1)
                title = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else ""
                items.append({"title": title, "content": content, "url": ""})
            elif line:
                items.append({"title": line[:60], "content": line, "url": ""})
        return items[:3]
    except Exception as e:
        log.debug(f"News search failed for '{query}': {e}")
        return []


async def fetch_headlines(interests: Optional[list[str]] = None) -> list[dict]:
    """
    Fetch top news headlines for user's interests.
    Returns list of {category, title, summary} dicts.
    """
    if interests is None:
        raw = os.environ.get("USER_INTERESTS", "football,tech,geopolitics,alerts")
        interests = [i.strip().lower() for i in raw.split(",")]

    queries = []
    for interest in interests:
        if interest in NEWS_QUERIES:
            queries.append((interest, NEWS_QUERIES[interest]))
        else:
            queries.append((interest, f"{interest} latest news today"))

    seen = set()
    unique_queries = []
    for cat, q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append((cat, q))

    results = []
    tasks = [_web_search(q) for _, q in unique_queries[:4]]
    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, (cat, _) in enumerate(unique_queries[:4]):
        res = search_results[i]
        if isinstance(res, Exception) or not res:
            continue
        for r in res[:2]:
            if r.get("title"):
                results.append({
                    "category": cat,
                    "title": r["title"],
                    "summary": r["content"][:200],
                    "url": r.get("url", ""),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                })

    return results[:5]


def format_headlines_for_brief(headlines: list[dict]) -> str:
    """Format headlines into natural spoken text."""
    if not headlines:
        return "I couldn't pull the latest news headlines right now."
    lines = []
    for i, h in enumerate(headlines, 1):
        lines.append(f"{i}. {h['title']}.")
    return "Here are the top news updates. " + " ".join(lines)
