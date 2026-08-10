"""
briefing.py — Morning Brief Orchestrator.

On laptop open / session start:
- Greets user warmly
- Talks about daily targets & where we left off yesterday
- Summarizes received emails (listing sender names, avoiding spam)
- Shares top headlines (football, tech, geopolitics, regional alerts)
- Invites user to deep dive ("say 'tell me about Rahul's email' or 'more on Barca news'")
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

log = logging.getLogger("briefing")


async def generate_morning_brief(
    user_name: str = "Kshayik",
    run_email: bool = True,
    run_news: bool = True,
    run_weather: bool = True,
    run_goals: bool = True,
) -> str:
    """
    Generate the full morning brief text.
    Runs all fetches in parallel. Returns natural spoken string.
    """
    from subsystems.morning.weather import get_weather, format_weather_for_brief
    from subsystems.morning.news_fetcher import fetch_headlines, format_headlines_for_brief
    from subsystems.morning.goal_tracker import get_daily_context, format_goals_for_brief

    now = datetime.now()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %-d")
    hour = now.hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    brief_parts = [f"{greeting}, {user_name}! Happy {day_name}, {date_str}."]

    # Parallel fetch
    tasks = {}
    if run_weather:
        tasks["weather"] = asyncio.create_task(get_weather())
    if run_news:
        tasks["news"] = asyncio.create_task(fetch_headlines())
    if run_goals:
        tasks["goals"] = asyncio.create_task(get_daily_context())
    if run_email:
        from agents.email_agent import get_email_agent
        tasks["email"] = asyncio.create_task(get_email_agent().get_summary(n=5))

    task_names = list(tasks.keys())
    task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results = dict(zip(task_names, task_results))

    # Weather
    if "weather" in results and not isinstance(results["weather"], Exception):
        brief_parts.append(format_weather_for_brief(results["weather"]))

    # Goals / Unfinished work from yesterday
    if "goals" in results and not isinstance(results["goals"], Exception):
        goals_text = format_goals_for_brief(results["goals"])
        if goals_text:
            brief_parts.append(goals_text)

    # Emails (list senders, avoid promotional spam)
    if "email" in results and not isinstance(results["email"], Exception):
        emails = results["email"]
        if emails:
            senders = list(dict.fromkeys(e["sender_name"] for e in emails if e.get("sender_name")))
            if senders:
                if len(senders) == 1:
                    brief_parts.append(f"You received new emails from {senders[0]}.")
                elif len(senders) == 2:
                    brief_parts.append(f"You received new emails from {senders[0]} and {senders[1]}.")
                else:
                    brief_parts.append(f"You received new emails from {', '.join(senders[:-1])}, and {senders[-1]}.")
        else:
            brief_parts.append("No new emails in your inbox.")

    # News headlines
    if "news" in results and not isinstance(results["news"], Exception):
        brief_parts.append(format_headlines_for_brief(results["news"]))

    brief_parts.append("What would you like to explore first?")

    return " ".join(brief_parts)


async def run_morning_brief(speak_fn=None, user_name: str = "Kshayik", **kwargs):
    """
    Generate and speak the morning brief via Kokoro TTS / IPC bus.
    """
    try:
        log.info("[Morning Brief] Generating...")
        brief_text = await generate_morning_brief(user_name=user_name, **kwargs)

        if speak_fn:
            await speak_fn(brief_text)
        else:
            from core.ipc_bus import bus
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', brief_text) if s.strip()]
            for sentence in sentences:
                await bus.dispatch("speak", {"text": sentence})
                await asyncio.sleep(0.1)

        log.info("[Morning Brief] Done.")
        return brief_text
    except Exception as e:
        log.error(f"[Morning Brief] Failed: {e}")
        return ""
