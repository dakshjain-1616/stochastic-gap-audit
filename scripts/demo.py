#!/usr/bin/env python3
"""
demo.py — Runnable demonstration of the Stochastic Gap Audit.

Works without any API keys (auto-detects and uses mock mode).
Always produces real output files in outputs/.

Usage:
    python demo.py
    python demo.py --seed 123
    python demo.py --model openai/gpt-5.4-mini
    python demo.py --html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, TaskProgressColumn
from rich import box

from stochastic_gap_audit import AuditReporter, HTMLReporter, StochasticGapSimulator, __version__
from stochastic_gap_audit.prompts import AUDIT_PROMPTS

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse demo command-line arguments."""
    p = argparse.ArgumentParser(description="Stochastic Gap Audit demo — no API key required")
    p.add_argument(
        "--model",
        default=os.getenv("AUDIT_MODEL", "mistralai/mistral-small-2603"),
        help="Model ID to audit (default: mistralai/mistral-small-2603)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("DEMO_SEED", "42")),
        help="Random seed for reproducible demo (default: 42)",
    )
    p.add_argument(
        "--output-dir",
        default=os.getenv("AUDIT_OUTPUT_DIR", "outputs"),
        help="Output directory (default: outputs/)",
    )
    p.add_argument(
        "--html",
        action="store_true",
        default=os.getenv("AUDIT_HTML", "").lower() in ("1", "true", "yes"),
        help="Also generate an HTML report with charts.",
    )
    return p.parse_args()


def _print_banner(model: str, dry_run: bool, seed: int) -> None:
    """Print the Rich startup banner."""
    mode_str = "[yellow]DRY-RUN (mock — no network calls)[/yellow]" if dry_run else "[green]LIVE (OpenRouter API)[/green]"
    content = (
        f"[bold cyan]Stochastic Gap Audit[/bold cyan]  [dim]v{__version__}[/dim]  "
        f"[dim]· Built by NEO (heyneo.so)[/dim]\n\n"
        f"  [bold]Model[/bold]   : [cyan]{model}[/cyan]\n"
        f"  [bold]Mode[/bold]    : {mode_str}\n"
        f"  [bold]Seed[/bold]    : {seed}\n"
        f"  [bold]Prompts[/bold] : {len(AUDIT_PROMPTS)}"
    )
    console.print(Panel(content, title="[bold white]DEMO — Pre-deployment Reliability Scorer[/bold white]", border_style="cyan"))


def _print_report(report) -> None:
    """Render the simulation report as Rich tables."""
    score = report.reliability_score

    # Score colour
    if score >= 80:
        score_style, risk_label, risk_style = "bold green", "LOW RISK — SAFE TO DEPLOY", "green"
    elif score >= 60:
        score_style, risk_label, risk_style = "bold yellow", "MEDIUM RISK — REVIEW BEFORE DEPLOY", "yellow"
    else:
        score_style, risk_label, risk_style = "bold red", "HIGH RISK — DO NOT DEPLOY", "red"

    bar_width = 38
    filled    = int(score / 100 * bar_width)
    bar_str   = "█" * filled + "░" * (bar_width - filled)

    score_content = (
        f"[{score_style}]{bar_str}[/]  [{score_style}]{score:.2f}/100[/]\n"
        f"  95% CI: [[dim]{report.score_ci_low:.2f}[/], [dim]{report.score_ci_high:.2f}[/]]\n"
        f"  Risk: [{risk_style}]{risk_label}[/]"
    )
    console.print(Panel(score_content, title="[bold]Reliability Score[/bold]", border_style=risk_style))

    # ── State distribution ───────────────────────────────────────────────────
    n = report.total_prompts
    dist_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    dist_table.add_column("State",   style="bold")
    dist_table.add_column("Count",   justify="right")
    dist_table.add_column("Pct",     justify="right")
    dist_table.add_row("[green]PASS[/]",       str(report.pass_count),      f"{report.pass_count/n*100:.1f}%")
    dist_table.add_row("[yellow]UNCERTAIN[/]", str(report.uncertain_count), f"{report.uncertain_count/n*100:.1f}%")
    dist_table.add_row("[red]FAIL[/]",         str(report.fail_count),      f"{report.fail_count/n*100:.1f}%")

    # ── Markov metrics ───────────────────────────────────────────────────────
    markov_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    markov_table.add_column("Metric")
    markov_table.add_column("Value",   justify="right")
    markov_table.add_column("Meaning", style="dim")
    markov_table.add_row("Oversight cost", f"{report.oversight_cost*100:.2f}%",           "fraction needing human review")
    markov_table.add_row("Stochastic gap", f"{report.stochastic_gap:.4f}",                "ideal − observed pass rate")
    markov_table.add_row("MFPT to FAIL",   f"{report.mean_first_passage_fail:.2f} steps", "avg steps PASS → first FAIL")
    markov_table.add_row("π[PASS]",        f"{report.steady_state[0]:.4f}",               "long-run PASS probability")
    markov_table.add_row("[red]π[FAIL][/]",f"[red]{report.steady_state[2]:.4f}[/]",       "long-run FAIL probability")

    # ── Per-tier scores ──────────────────────────────────────────────────────
    tier_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tier_table.add_column("Tier")
    tier_table.add_column("Pass rate", justify="right")
    tier_table.add_column("Grade")
    for tier, ts in sorted(report.tier_scores.items()):
        grade = "[green]✓ GOOD[/]" if ts >= 70 else "[yellow]△ REVIEW[/]" if ts >= 50 else "[red]✗ POOR[/]"
        tier_table.add_row(tier, f"{ts:.1f}%", grade)

    console.print("\n[bold]State Distribution[/bold]")
    console.print(dist_table)
    console.print("[bold]Markov Metrics[/bold]")
    console.print(markov_table)
    console.print("[bold]Per-Tier Scores[/bold]")
    console.print(tier_table)

    # ── Sample results ───────────────────────────────────────────────────────
    sample_table = Table(box=box.SIMPLE, show_header=True, header_style="bold", title="Sample Results (first 5 prompts)")
    sample_table.add_column("State",  style="bold", width=10)
    sample_table.add_column("Tier",   width=12)
    sample_table.add_column("Prompt", width=55)
    sample_table.add_column("Response (truncated)", width=60)
    icons = {"PASS": "[green]✓ PASS[/]", "UNCERTAIN": "[yellow]△ UNCERTAIN[/]", "FAIL": "[red]✗ FAIL[/]"}
    for r in report.results[:5]:
        sample_table.add_row(
            icons.get(r.state_label, r.state_label),
            r.tier,
            r.prompt[:55],
            r.response[:60],
        )
    console.print(sample_table)


def main() -> None:
    """Run the Stochastic Gap Audit demo, saving outputs to disk."""
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    dry_run = not api_key

    if not api_key:
        console.print(
            "[yellow]⚠  No OPENROUTER_API_KEY — running in mock mode (zero network calls).[/yellow]"
        )

    _print_banner(args.model, dry_run, args.seed)

    simulator = StochasticGapSimulator(
        model   = args.model,
        api_key = api_key or None,
        dry_run = dry_run,
        seed    = args.seed,
    )

    mode_str = "DRY-RUN (mock — no network calls)" if dry_run else "LIVE (OpenRouter API)"

    # ── Run with Rich progress bar ─────────────────────────────────────────
    from stochastic_gap_audit.simulator import (
        grade_response, SimulationReport, PromptResult,
        STATE_LABELS, STATE_PASS, STATE_UNCERTAIN, STATE_FAIL
    )
    from stochastic_gap_audit.prompts import TIER_WEIGHTS
    import numpy as np
    from datetime import datetime, timezone as _tz

    t0 = time.perf_counter()

    results: list[PromptResult] = []
    transition_counts = np.zeros((3, 3), dtype=float)
    prev_state = None
    tier_state_totals: dict = {}
    audit_ts = datetime.now(tz=_tz.utc).isoformat()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Markovian simulation…", total=len(AUDIT_PROMPTS))
        for i, p in enumerate(AUDIT_PROMPTS):
            prompt_ts         = datetime.now(tz=_tz.utc).isoformat()
            response, latency = simulator.client.call(p["prompt"], p["difficulty"])
            state, kw_hits, kw_total = grade_response(
                response, p["expected_keywords"], p["difficulty"], p["tier"]
            )
            weight     = TIER_WEIGHTS.get(p["tier"], 1.0)
            norm_score = (1.0 - state / 2.0) * weight

            result = PromptResult(
                prompt_id      = p["id"],
                tier           = p["tier"],
                prompt         = p["prompt"],
                response       = response,
                state          = state,
                state_label    = STATE_LABELS[state],
                latency_ms     = latency,
                keyword_hits   = kw_hits,
                keyword_total  = kw_total,
                difficulty     = p["difficulty"],
                weighted_score = norm_score,
                timestamp      = prompt_ts,
            )
            results.append(result)
            tier_state_totals.setdefault(p["tier"], []).append(state)
            if prev_state is not None:
                transition_counts[prev_state][state] += 1
            prev_state = state

            state_colour = {"PASS": "green", "UNCERTAIN": "yellow", "FAIL": "red"}.get(STATE_LABELS[state], "white")
            progress.update(
                task,
                advance=1,
                description=f"[{state_colour}]{STATE_LABELS[state]:<9}[/] prompt {i+1}/{len(AUDIT_PROMPTS)}",
            )

    elapsed   = time.perf_counter() - t0
    P         = simulator._normalise_transition_matrix(transition_counts)
    ss        = simulator._compute_steady_state(P)
    rel       = simulator._compute_reliability_score(results, ss, len(AUDIT_PROMPTS))
    ci_low, ci_high = simulator._bootstrap_confidence_interval(results, ss, seed=simulator.seed)

    uncertain_count = sum(1 for r in results if r.state == STATE_UNCERTAIN)
    pass_count      = sum(1 for r in results if r.state == STATE_PASS)
    fail_count      = sum(1 for r in results if r.state == STATE_FAIL)
    observed_pass   = pass_count / len(results)
    ideal_pass      = float(ss[STATE_PASS])
    stochastic_gap  = max(0.0, ideal_pass - observed_pass)
    mfpt            = simulator._mean_first_passage_time(P, target=STATE_FAIL)
    tier_scores     = {}
    for tier, states in tier_state_totals.items():
        tier_pass = sum(1 for s in states if s == STATE_PASS)
        tier_scores[tier] = round(tier_pass / len(states) * 100, 2)

    report = SimulationReport(
        model                   = simulator.model,
        dry_run                 = simulator.dry_run,
        results                 = results,
        transition_matrix       = P,
        steady_state            = ss,
        reliability_score       = rel,
        oversight_cost          = uncertain_count / len(results),
        stochastic_gap          = stochastic_gap,
        mean_first_passage_fail = mfpt,
        execution_time_s        = elapsed,
        tier_scores             = tier_scores,
        total_prompts           = len(results),
        pass_count              = pass_count,
        uncertain_count         = uncertain_count,
        fail_count              = fail_count,
        score_ci_low            = ci_low,
        score_ci_high           = ci_high,
        timestamp               = audit_ts,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    reporter = AuditReporter(output_dir=args.output_dir)
    paths    = reporter.save_all(report)
    csv_path = reporter.save_reliability_csv(report, filename="reliability_score.csv")

    if args.html:
        hr = HTMLReporter(output_dir=args.output_dir)
        paths["html"] = hr.save(report)

    # ── Save demo JSON report ──────────────────────────────────────────────
    demo_report_path = Path(args.output_dir) / "demo_report.json"
    demo_data = {
        "demo_run": {
            "model":            args.model,
            "mode":             mode_str,
            "seed":             args.seed,
            "execution_time_s": round(elapsed, 3),
        },
        "scores": {
            "reliability_score":       report.reliability_score,
            "score_ci_low":            report.score_ci_low,
            "score_ci_high":           report.score_ci_high,
            "oversight_cost":          round(report.oversight_cost, 4),
            "stochastic_gap":          round(report.stochastic_gap, 4),
            "mean_first_passage_fail": report.mean_first_passage_fail,
        },
        "state_distribution": {
            "PASS":      report.pass_count,
            "UNCERTAIN": report.uncertain_count,
            "FAIL":      report.fail_count,
        },
        "tier_scores": report.tier_scores,
        "steady_state": {
            "PASS":      round(float(report.steady_state[0]), 6),
            "UNCERTAIN": round(float(report.steady_state[1]), 6),
            "FAIL":      round(float(report.steady_state[2]), 6),
        },
        "transition_matrix": report.transition_matrix.tolist(),
        "top_5_results": [
            {
                "id":         r.prompt_id,
                "tier":       r.tier,
                "prompt":     r.prompt,
                "state":      r.state_label,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in report.results[:5]
        ],
    }
    demo_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(demo_report_path, "w") as f:
        json.dump(demo_data, f, indent=2)

    # ── Print Rich report ─────────────────────────────────────────────────
    _print_report(report)

    # ── File manifest ─────────────────────────────────────────────────────
    file_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    file_table.add_column("File")
    file_table.add_column("Description")
    file_table.add_row(str(csv_path),         "[green]main output[/] — reliability_score.csv")
    for fmt, p in paths.items():
        file_table.add_row(str(p), fmt)
    file_table.add_row(str(demo_report_path), "demo JSON report")
    console.print("\n[bold]Output Files[/bold]")
    console.print(file_table)

    # ── Final verdict ─────────────────────────────────────────────────────
    if report.reliability_score >= 80:
        console.print(f"\n[bold green]✓  Final verdict: LOW RISK — SAFE TO DEPLOY[/bold green]")
    elif report.reliability_score >= 60:
        console.print(f"\n[bold yellow]△  Final verdict: MEDIUM RISK — REVIEW BEFORE DEPLOY[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗  Final verdict: HIGH RISK — DO NOT DEPLOY[/bold red]")


if __name__ == "__main__":
    main()
