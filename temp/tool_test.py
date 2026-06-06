# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ollama",
#     "rich",
# ]
# ///
"""
Tool-calling benchmark for local Ollama models.

Usage:
  python temp/tool_test.py                  # interactive menu
  python temp/tool_test.py benchmark        # run scored suite on one model
  python temp/tool_test.py compare          # run scored suite across ALL local models
  python temp/tool_test.py "your prompt"    # single prompt, quick test
"""

import sys
import os
import json
import glob
import subprocess
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.rule import Rule
from rich import box
from ollama import chat, Client

console = Console()

# ─── System prompt injected into every chat call ─────────────────────────────
# This is the primary guard against over-eager tool calling.
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a small set of OS tools.

CRITICAL RULES — read carefully before every response:

1. ONLY call a tool when the user is EXPLICITLY asking for live system data.
   - "what is my battery?" → call check_battery
   - "which window is active?" → call get_active_window
   - "run ls" / "list files" / "what's in this folder?" → call run_bg_cmd with the right shell command

2. DO NOT call any tool for:
   - Greetings or small talk ("hi", "hello", "how are you")
   - Questions about your own capabilities ("what can you do?", "what tools do you have?")
   - Questions that don't need live OS data
   → Just reply in plain text.

3. When calling run_bg_cmd, use STANDARD Linux shell commands (ls, pwd, grep, etc.).
   Do NOT invent file paths or guess directories — use sensible defaults.
   Example: "list files in Projects" → run_bg_cmd("ls ~/Projects") or run_bg_cmd("ls /home/<user>/Projects")

4. pick the CORRECT tool for the job:
   - Directory listings, file reads, system info → run_bg_cmd
   - Active application window → get_active_window
   - Battery info → check_battery

5. If unsure whether a tool is needed, err on the side of plain text.

Available tools and when to use them:
  check_battery      → ONLY when asked about battery level or charging status
  get_active_window  → ONLY when asked which app/window is currently focused
  run_bg_cmd(cmd)    → ONLY when asked to run a shell command or get filesystem/system info
"""

# ─── Tool Implementations ─────────────────────────────────────────────────────

def check_battery() -> str:
    """
    Check the laptop's battery percentage and charging status.
    Use ONLY when the user explicitly asks about battery level or charging state.
    Example triggers: 'battery status', 'is it charging', 'how much battery do I have'.
    Do NOT use for any other query.

    Returns:
        JSON string with keys: 'level' (e.g. '85%') and 'status' ('Charging' / 'Discharging' / 'Full').
    """
    try:
        bats = glob.glob("/sys/class/power_supply/BAT*")
        if not bats:
            return json.dumps({"error": "No battery found — this may be a desktop machine."})
        bat = bats[0]
        result = subprocess.run(
            f"cat {bat}/capacity && cat {bat}/status",
            shell=True, capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")
        level  = lines[0] if len(lines) > 0 else "unknown"
        status = lines[1] if len(lines) > 1 else "unknown"
        return json.dumps({"level": f"{level}%", "status": status})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_active_window() -> str:
    """
    Return the title of the application window that is currently in focus on screen.
    Use ONLY when the user asks which app/window is active or currently open.
    Example triggers: 'which window is active', 'what app is focused', 'what am I looking at'.
    Do NOT use to list files, directories, or answer shell questions.

    Returns:
        JSON string with key 'window' containing the window title string.
    """
    try:
        result = subprocess.run(
            "xdotool getactivewindow getwindowname",
            shell=True, capture_output=True, text=True, timeout=5
        )
        window = result.stdout.strip() or "unknown"
        return json.dumps({"window": window})
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_bg_cmd(cmd: str) -> str:
    """
    Execute a safe, read-only Linux shell command and return its output.
    Use when the user asks to run a command, list directory contents, check system info, read a file, etc.
    Example triggers: 'run ls', 'list files', 'show directory', 'what's in ~/Projects', 'show cpu info'.

    Allowed commands (whitelist): cat, ls, grep, pwd, echo, ps, df, free, uname, which, find, head, tail, wc, stat.
    Do NOT use for battery or active-window queries — use the dedicated tools for those.

    Args:
        cmd: A valid Linux shell command string using only the whitelisted commands above.
             Use real, sensible paths. If no path is given for ls, default to the current directory.

    Returns:
        JSON string with 'output' (stdout) and 'exit_code', or 'error' if blocked or failed.
    """
    cmd = cmd.strip()
    if not cmd:
        return json.dumps({"error": "Empty command."})

    first_word = cmd.split()[0]
    whitelisted = {
        "cat", "ls", "grep", "pwd", "echo", "ps", "df", "free",
        "uname", "which", "find", "head", "tail", "wc", "stat"
    }

    if first_word not in whitelisted:
        return json.dumps({
            "error": f"'{first_word}' is not allowed. Whitelisted: {sorted(whitelisted)}"
        })

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip() or result.stderr.strip() or "(no output)"
        if len(output) > 1500:
            output = output[:1500] + "\n...[truncated]"
        return json.dumps({"output": output, "exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Command timed out after 10 seconds."})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_MAPPING = {
    "check_battery":    check_battery,
    "get_active_window": get_active_window,
    "run_bg_cmd":       run_bg_cmd,
}
TOOLS_LIST = [check_battery, get_active_window, run_bg_cmd]

# ─── Benchmark test cases ─────────────────────────────────────────────────────
# Each case specifies what the correct behaviour should be:
#   expected_tool: None means no tool should be called (conversational)
BENCHMARK = [
    {
        "prompt":        "hi",
        "expected_tool": None,
        "desc":          "Simple greeting — no tool needed",
    },
    {
        "prompt":        "How are you doing today?",
        "expected_tool": None,
        "desc":          "Conversational — no tool needed",
    },
    {
        "prompt":        "What tools do you have access to?",
        "expected_tool": None,
        "desc":          "Meta question about capabilities — answer from context, no tool",
    },
    {
        "prompt":        "What is my battery status right now?",
        "expected_tool": "check_battery",
        "desc":          "Battery query → check_battery",
    },
    {
        "prompt":        "Is my laptop charging?",
        "expected_tool": "check_battery",
        "desc":          "Charging query → check_battery",
    },
    {
        "prompt":        "Which window do I currently have active?",
        "expected_tool": "get_active_window",
        "desc":          "Active window → get_active_window",
    },
    {
        "prompt":        "What app is currently focused on screen?",
        "expected_tool": "get_active_window",
        "desc":          "Focused app → get_active_window",
    },
    {
        "prompt":        "Run the ls command for me",
        "expected_tool": "run_bg_cmd",
        "desc":          "Explicit ls request → run_bg_cmd",
    },
    {
        "prompt":        "List all files in the current directory",
        "expected_tool": "run_bg_cmd",
        "desc":          "Directory listing → run_bg_cmd",
    },
    {
        "prompt":        "What is the current working directory?",
        "expected_tool": "run_bg_cmd",
        "desc":          "pwd query → run_bg_cmd",
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_local_models() -> list[str]:
    """Return all model names currently pulled in local Ollama."""
    try:
        client = Client()
        raw = client.list()
        names = []
        for m in raw.get("models", []):
            name = m.get("model") or m.get("name") or ""
            if name:
                names.append(name)
        return names
    except Exception as e:
        console.print(f"[red]Could not list models: {e}[/red]")
        return []


def pick_functiongemma(models: list[str]) -> str:
    for m in models:
        if "functiongemma" in m:
            return m
    return "functiongemma:270m"


def run_one(model: str, prompt: str, silent: bool = False) -> dict:
    """
    Run a single prompt through model. Returns a result dict with:
      tool_called, args, final_response, latency_ms, error
    """
    messages = []
    if not model.startswith("meeseeks-"):
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
    think_val = os.environ.get("OLLAMA_THINK", "false").lower().strip() == "true"

    t0 = time.monotonic()
    try:
        response = chat(
            model=model, 
            messages=messages, 
            tools=TOOLS_LIST,
            options={"num_ctx": num_ctx},
            think=think_val
        )
    except Exception as e:
        return {"tool_called": None, "args": {}, "final_response": "", "latency_ms": 0, "error": str(e)}

    latency_ms = int((time.monotonic() - t0) * 1000)
    tool_called = None
    args = {}
    final_response = ""
    error = None

    if response.message.tool_calls:
        tc = response.message.tool_calls[0]
        tool_called = tc.function.name
        args = tc.function.arguments or {}

        messages.append(response.message)

        if tool_called in TOOL_MAPPING:
            func = TOOL_MAPPING[tool_called]
            try:
                result = func(**args)
            except TypeError as e:
                result = json.dumps({"error": f"Bad args: {e}"})
        else:
            result = json.dumps({"error": f"Tool '{tool_called}' not found."})

        messages.append({"role": "tool", "name": tool_called, "content": result})

        t1 = time.monotonic()
        try:
            final = chat(
                model=model, 
                messages=messages,
                options={"num_ctx": num_ctx},
                think=think_val
            )
            final_response = final.message.content or ""
        except Exception as e:
            error = str(e)
        latency_ms = int((time.monotonic() - t0) * 1000)
    else:
        final_response = response.message.content or ""

    return {
        "tool_called":    tool_called,
        "args":           args,
        "final_response": final_response,
        "latency_ms":     latency_ms,
        "error":          error,
    }


def print_result(result: dict, prompt: str):
    """Pretty-print a single run result."""
    console.print(Panel(f"[bold cyan]{prompt}[/bold cyan]", title="Prompt"))

    if result["error"]:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        return

    if result["tool_called"]:
        console.print(f"  🔧 Tool called : [magenta]{result['tool_called']}[/magenta]")
        if result["args"]:
            console.print(Syntax(json.dumps(result["args"], indent=2), "json", theme="monokai"))
    else:
        console.print("  💬 No tool called (conversational)")

    if result["final_response"]:
        console.print(f"  [bold green]Response:[/bold green] {result['final_response']}")
    console.print(f"  ⏱  Latency: [dim]{result['latency_ms']}ms[/dim]")


def run_benchmark(model: str) -> list[dict]:
    """Run all benchmark cases and return scored results."""
    console.print()
    console.print(Rule(f"[bold]Benchmark: [cyan]{model}[/cyan]"))
    results = []

    # Ensure output directory exists
    os.makedirs("temp/output", exist_ok=True)
    model_sanitized = model.replace("/", "_").replace(":", "_")
    log_path = f"temp/output/{model_sanitized}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Run for Model: {model}\n")
        f.write("=" * 80 + "\n\n")

        for i, case in enumerate(BENCHMARK, 1):
            prompt        = case["prompt"]
            expected_tool = case["expected_tool"]
            desc          = case["desc"]

            console.print(f"\n[dim]Case {i}/{len(BENCHMARK)}:[/dim] {desc}")
            with console.status(f"[yellow]Running…"):
                r = run_one(model, prompt)

            tool_called = r["tool_called"]

            # Score: correct = expected matches reality
            correct = (tool_called == expected_tool)

            if correct:
                verdict = "[bold green]✓ PASS[/bold green]"
                file_verdict = "PASS"
            else:
                verdict = f"[bold red]✗ FAIL[/bold red] (got [magenta]{tool_called}[/magenta], expected [magenta]{expected_tool}[/magenta])"
                file_verdict = f"FAIL (got {tool_called}, expected {expected_tool})"

            console.print(f"  {verdict}  ⏱ {r['latency_ms']}ms")
            if r["final_response"]:
                console.print(f"  [dim]Response: {r['final_response']}[/dim]")

            # Write detailed output log
            f.write(f"Case {i}: {desc}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Expected Tool: {expected_tool or 'None'}\n")
            f.write(f"Got Tool: {tool_called or 'None'}\n")
            f.write(f"Verdict: {file_verdict}\n")
            f.write(f"Latency: {r['latency_ms']}ms\n")
            f.write(f"Response:\n{r['final_response']}\n")
            f.write("-" * 50 + "\n\n")

            results.append({
                "case":          case,
                "result":        r,
                "correct":       correct,
            })

    console.print(f"\n[bold green]Saved detailed logs to {log_path}[/bold green]\n")
    return results


def print_score_table(model: str, scored: list[dict]):
    """Print a summary table for one model's benchmark results."""
    passed = sum(1 for s in scored if s["correct"])
    total  = len(scored)
    pct    = int(100 * passed / total) if total else 0
    avg_ms = sum(s["result"]["latency_ms"] for s in scored) / total if total else 0

    table = Table(
        title=f"Results for [cyan]{model}[/cyan]  —  Score: {passed}/{total} ({pct}%)  —  Avg Wait: {avg_ms:.1f}ms",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#",        style="dim",     width=3)
    table.add_column("Prompt",   style="white",   max_width=45, overflow="fold")
    table.add_column("Expected", style="yellow",  width=20)
    table.add_column("Got",      style="magenta", width=20)
    table.add_column("Pass",     style="green",   width=6)
    table.add_column("ms",       style="dim",     width=6)

    for i, s in enumerate(scored, 1):
        expected = s["case"]["expected_tool"] or "none"
        got      = s["result"]["tool_called"] or "none"
        passmark = "✓" if s["correct"] else "✗"
        style    = "green" if s["correct"] else "red"
        table.add_row(
            str(i),
            s["case"]["prompt"],
            expected,
            got,
            f"[{style}]{passmark}[/{style}]",
            str(s["result"]["latency_ms"]),
        )

    console.print()
    console.print(table)
    return passed, total, pct


# ─── Interactive prompt loop ──────────────────────────────────────────────────

def interactive_mode(model: str):
    console.print(f"\n[bold green]Interactive mode — model: [cyan]{model}[/cyan][/bold green]")
    console.print("[dim]Type 'exit' to quit.[/dim]\n")

    while True:
        try:
            prompt = Prompt.ask("[bold cyan]You[/bold cyan]")
        except KeyboardInterrupt:
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "bye"}:
            console.print("[bold purple]Bye![/bold purple]")
            break

        with console.status("[yellow]Thinking…"):
            r = run_one(model, prompt)

        print_result(r, prompt)
        console.print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print("[bold purple]══════════════════════════════════════════════[/bold purple]")
    console.print("[bold purple]   OLLAMA TOOL-CALLING TEST SUITE             [/bold purple]")
    console.print("[bold purple]══════════════════════════════════════════════[/bold purple]")
    console.print()

    local_models = get_local_models()
    default_model = pick_functiongemma(local_models)

    # Print models table
    mt = Table(title="Models Available Locally", box=box.SIMPLE)
    mt.add_column("Model", style="cyan")
    mt.add_column("Is Default", style="yellow")
    for m in local_models:
        mt.add_row(m, "← default" if m == default_model else "")
    console.print(mt)
    console.print()

    # Print tools table
    tt = Table(title="Tools Registered", box=box.SIMPLE)
    tt.add_column("Tool",        style="magenta")
    tt.add_column("Trigger",     style="white")
    tt.add_column("Params",      style="cyan")
    tt.add_row("check_battery",    "Battery / charging questions",             "none")
    tt.add_row("get_active_window","Active window / focused app questions",    "none")
    tt.add_row("run_bg_cmd",       "Shell commands / file listings / system info", "cmd: str")
    console.print(tt)
    console.print()

    # CLI shortcut
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "benchmark":
            model = sys.argv[2] if len(sys.argv) > 2 else default_model
            scored = run_benchmark(model)
            print_score_table(model, scored)
            return

        if mode == "compare-models":
            if len(sys.argv) < 4:
                console.print("[bold red]Usage: python temp/tool_test.py compare-models <model_a> <model_b>[/bold red]")
                return
            model_a = sys.argv[2]
            model_b = sys.argv[3]
            console.print(f"[bold yellow]Comparing '{model_a}' vs '{model_b}' on benchmark suite…[/bold yellow]\n")
            
            scored_a = run_benchmark(model_a)
            passed_a, total_a, pct_a = print_score_table(model_a, scored_a)
            avg_a = sum(s["result"]["latency_ms"] for s in scored_a) / total_a if total_a else 0

            scored_b = run_benchmark(model_b)
            passed_b, total_b, pct_b = print_score_table(model_b, scored_b)
            avg_b = sum(s["result"]["latency_ms"] for s in scored_b) / total_b if total_b else 0

            console.print()
            console.print(Rule("[bold]Comparison Leaderboard[/bold]"))
            lb = Table(box=box.ROUNDED)
            lb.add_column("Model", style="cyan")
            lb.add_column("Score", style="green")
            lb.add_column("%",     style="yellow")
            lb.add_column("Avg Wait", style="magenta")
            
            comparison = [(model_a, passed_a, total_a, pct_a, avg_a), (model_b, passed_b, total_b, pct_b, avg_b)]
            comparison.sort(key=lambda x: -x[3])
            
            for m, passed, total, pct, avg in comparison:
                lb.add_row(m, f"{passed}/{total}", f"{pct}%", f"{avg:.1f}ms")
            console.print(lb)
            return

        if mode == "compare":
            console.print("[bold yellow]Comparing all local models on benchmark suite…[/bold yellow]\n")
            summary = []
            for m in local_models:
                scored = run_benchmark(m)
                passed, total, pct = print_score_table(m, scored)
                avg_ms = sum(s["result"]["latency_ms"] for s in scored) / total if total else 0
                summary.append((m, passed, total, pct, avg_ms))

            # Final leaderboard
            console.print()
            console.print(Rule("[bold]Leaderboard[/bold]"))
            lb = Table(box=box.ROUNDED)
            lb.add_column("Rank",  style="dim",    width=5)
            lb.add_column("Model", style="cyan")
            lb.add_column("Score", style="green")
            lb.add_column("%",     style="yellow")
            lb.add_column("Avg Wait", style="magenta")
            summary.sort(key=lambda x: -x[3])
            for rank, (m, passed, total, pct, avg) in enumerate(summary, 1):
                lb.add_row(str(rank), m, f"{passed}/{total}", f"{pct}%", f"{avg:.1f}ms")
            console.print(lb)
            return

        # Single prompt
        prompt = " ".join(sys.argv[1:])
        with console.status("[yellow]Thinking…"):
            r = run_one(default_model, prompt)
        print_result(r, prompt)
        return

    # Interactive menu
    console.print("[bold yellow]What do you want to do?[/bold yellow]")
    console.print("  1. [cyan]Benchmark[/cyan]   — scored test suite (one model)")
    console.print("  2. [cyan]Compare[/cyan]     — benchmark all local models & rank them")
    console.print("  3. [cyan]Interactive[/cyan] — free chat / test prompts manually")
    choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")

    if choice == "1":
        model = Prompt.ask("Model", default=default_model)
        scored = run_benchmark(model)
        print_score_table(model, scored)

    elif choice == "2":
        console.print("[bold yellow]Running compare across all local models…[/bold yellow]")
        summary = []
        for m in local_models:
            scored = run_benchmark(m)
            passed, total, pct = print_score_table(m, scored)
            avg_ms = sum(s["result"]["latency_ms"] for s in scored) / total if total else 0
            summary.append((m, passed, total, pct, avg_ms))

        console.print()
        console.print(Rule("[bold]🏆 Leaderboard[/bold]"))
        lb = Table(box=box.ROUNDED)
        lb.add_column("Rank",  style="dim",    width=5)
        lb.add_column("Model", style="cyan")
        lb.add_column("Score", style="green")
        lb.add_column("%",     style="yellow")
        lb.add_column("Avg Wait", style="magenta")
        summary.sort(key=lambda x: -x[3])
        for rank, (m, passed, total, pct, avg) in enumerate(summary, 1):
            lb.add_row(str(rank), m, f"{passed}/{total}", f"{pct}%", f"{avg:.1f}ms")
        console.print(lb)

    else:
        model = Prompt.ask("Model", default=default_model)
        interactive_mode(model)


if __name__ == "__main__":
    main()
