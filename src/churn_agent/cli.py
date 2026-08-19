"""Terminal REPL for the agent: `python -m churn_agent.cli` (or `... "one question"`).

Pure presentation — renders the same AgentEvents the Streamlit app consumes.
The live plan / tool-call / self-check / verification trace is the point:
you can watch the agent refuse to state numbers it didn't compute.
"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.panel import Panel

from .agent import Agent, AgentEvent
from .config import METRICS_PATH, AgentConfig
from .tools import warm_up

console = Console()


def render(event: AgentEvent) -> None:
    p = event.payload
    match event.kind:
        case "plan":
            console.print(f"[bold cyan]PLAN[/] {p['plan']}")
        case "tool_call":
            args = json.dumps(p["args"])
            args = args if len(args) <= 100 else args[:100] + "…"
            console.print(f"[dim]→ {p['tool']}({args})[/]")
        case "tool_result":
            colors = {"ok": "green", "empty": "yellow", "error": "red"}
            color = colors.get(p["status"], "yellow")
            console.print(f"  [{color}]{p['status']}[/] [dim]{p['summary']}[/]")
        case "self_check_retry":
            console.print(f"  [yellow]self-check:[/] [dim]{p['note']}[/]")
        case "revision":
            console.print(
                f"[magenta]verification rejected draft[/] "
                f"(unverified: {', '.join(p['unmatched'])}) — revising…"
            )
        case "verify_verdict":
            style = "green" if p["ok"] else "red"
            console.print(f"[{style}]verify:[/] {p['summary']}")
        case "error":
            console.print(f"[red]error:[/] {p['message']}")


def banner(config: AgentConfig) -> None:
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    console.print(
        Panel.fit(
            "[bold]Churn Analyst Agent[/] — ask about the 7,043-customer telco dataset\n"
            f"[dim]model: PR-AUC {metrics.get('PR-AUC', '?')} · ROC-AUC {metrics.get('ROC-AUC', '?')} · "
            f"provider: {config.base_url.split('/')[2]} · {config.model}[/]\n"
            "[dim]commands: /reset (clear memory) · /facts (last answer's checks) · /quit[/]",
            border_style="cyan",
        )
    )


def main() -> None:
    config = AgentConfig.from_env()
    agent = Agent(config)
    banner(config)
    with console.status("[dim]warming up (dataset, model, sandbox)…[/]"):
        warm_up()

    last_result = None
    one_shot = " ".join(sys.argv[1:]).strip()

    while True:
        try:
            question = one_shot or console.input("\n[bold]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/quit", "/q", "exit"):
            break
        if question == "/reset":
            agent.reset()
            console.print("[dim]memory cleared[/]")
            continue
        if question == "/facts":
            if last_result and last_result.verification:
                v = last_result.verification
                console.print(f"verified: {v.matched}\nunverified: {v.unmatched}\nredacted: {v.redactions}")
            else:
                console.print("[dim]nothing verified yet[/]")
            continue

        try:
            last_result = agent.ask(question, on_event=render)
        except Exception as exc:  # last resort: never show the user a traceback
            console.print(f"[red]unexpected error:[/] {exc} — nothing was fabricated; please try again")
            if one_shot:
                break
            continue
        verdict = ""
        if last_result.verification:
            verdict = f"  [dim]· {last_result.verification.summary()}[/]"
        console.print(
            Panel(last_result.answer, border_style="green", title="answer", title_align="left")
        )
        console.print(
            f"[dim]{last_result.llm_calls} LLM calls · {len(last_result.steps)} tool steps · "
            f"{last_result.elapsed_s}s[/]{verdict}"
        )
        if one_shot:
            break


if __name__ == "__main__":
    main()
