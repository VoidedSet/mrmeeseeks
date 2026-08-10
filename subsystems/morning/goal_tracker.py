"""
goal_tracker.py — Daily goals and context persistence.
Stores goals and session notes in ~/.meeseeks/daily_context.json.
Tracks targets for the day and where we left off yesterday.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("goal_tracker")

MEESEEKS_DIR = Path.home() / ".meeseeks"
CONTEXT_FILE = MEESEEKS_DIR / "daily_context.json"


def _load_context() -> dict:
    """Load daily context from JSON file."""
    MEESEEKS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTEXT_FILE.exists():
        return {"date": datetime.now().strftime("%Y-%m-%d"), "goals": [], "completed": [], "pending": [], "session_notes": []}
    try:
        with open(CONTEXT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": datetime.now().strftime("%Y-%m-%d"), "goals": [], "completed": [], "pending": [], "session_notes": []}


def _save_context(ctx: dict):
    """Save daily context to JSON file."""
    MEESEEKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONTEXT_FILE, "w") as f:
        json.dump(ctx, f, indent=2)


async def get_daily_context() -> dict:
    """
    Get today's goals and yesterday's unfinished work.
    Returns dict with today, goals, completed, pending, session_notes, yesterday_unfinished.
    """
    ctx = await asyncio.to_thread(_load_context)
    today = datetime.now().strftime("%Y-%m-%d")

    # If context is from a previous day, roll over pending/goals into yesterday_unfinished
    if ctx.get("date") != today:
        yesterday_pending = ctx.get("goals", []) + ctx.get("pending", [])
        new_ctx = {
            "date": today,
            "goals": [],
            "completed": [],
            "pending": [],
            "session_notes": ctx.get("session_notes", [])[-3:],
            "yesterday_unfinished": yesterday_pending,
        }
        await asyncio.to_thread(_save_context, new_ctx)
        ctx = new_ctx

    return {
        "today": today,
        "goals": ctx.get("goals", []),
        "completed": ctx.get("completed", []),
        "pending": ctx.get("pending", []),
        "session_notes": ctx.get("session_notes", []),
        "yesterday_unfinished": ctx.get("yesterday_unfinished", []),
    }


async def add_goal(goal_text: str):
    """Add a goal / target for today."""
    ctx = await asyncio.to_thread(_load_context)
    ctx.setdefault("goals", []).append(goal_text)
    await asyncio.to_thread(_save_context, ctx)


async def mark_done(goal_text: str) -> bool:
    """Mark a goal as completed."""
    ctx = await asyncio.to_thread(_load_context)
    goals = ctx.get("goals", [])
    matched = next((g for g in goals if goal_text.lower() in g.lower() or g.lower() in goal_text.lower()), None)
    if matched:
        goals.remove(matched)
        ctx["goals"] = goals
        ctx.setdefault("completed", []).append(matched)
        await asyncio.to_thread(_save_context, ctx)
        return True
    return False


async def save_session_note(note: str):
    """Save a session note ('left off at X')."""
    ctx = await asyncio.to_thread(_load_context)
    note_entry = f"{datetime.now().strftime('%H:%M')} — {note}"
    ctx.setdefault("session_notes", []).append(note_entry)
    ctx["session_notes"] = ctx["session_notes"][-10:]
    await asyncio.to_thread(_save_context, ctx)


def format_goals_for_brief(daily_ctx: dict) -> str:
    """Format goals into natural spoken text."""
    parts = []
    if daily_ctx.get("yesterday_unfinished"):
        items = ", ".join(daily_ctx["yesterday_unfinished"][:3])
        parts.append(f"Where we left off yesterday: {items}.")
    if daily_ctx.get("session_notes"):
        last_note = daily_ctx["session_notes"][-1]
        parts.append(f"Last note: {last_note}.")
    if daily_ctx.get("goals"):
        items = ", ".join(daily_ctx["goals"][:3])
        parts.append(f"Today's targets: {items}.")
    if not parts:
        return "No specific targets logged for today yet."
    return " ".join(parts)
