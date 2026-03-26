#!/usr/bin/env python3
"""
audit.py — Stochastic Gap Audit CLI  (v2.0)

Usage:
    python audit.py --model mistralai/mistral-small-2603
    python audit.py --model mistralai/mistral-small-2603 --dry-run
    python audit.py --model mistralai/mistral-small-2603 --output-dir results/ --seed 42
    python audit.py --compare mistralai/mistral-small-2603 openai/gpt-5.4-nano
    python audit.py --model mistralai/mistral-small-2603 --html --history
    python audit.py --version
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TaskProgressColumn,
)
from rich import box
from rich.text import Text

from stochastic_gap_audit import (
    AuditReporter,
    AuditHistory,
    HTMLReporter,
    ModelComparator,
    StochasticGapSimulator,
    __version__,
)
from stochastic_gap_audit.prompts import AUDIT_PROMPTS

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="audit.py",
        description="Stochastic Gap Audit v2.0 — pre-deployment reliability scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audit.py --model mistralai/mistral-small-2603
  python audit.py --model mistralai/mistral-small-2603 --dry-run --html
  python audit.py --compare mistralai/mistral-small-2603 openai/gpt-5.4-nano
  python audit.py --model mistralai/mistral-small-2603 --history --seed 42
        """,
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"Stochastic Gap Audit v{__version__} — built by NEO (heyneo.so)",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("AUDIT_MODEL", "mistralai/mistral-small-2603"),
        help="Model identifier (OpenRouter format). "
             "Default: mistralai/mistral-small-2603",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Compare multiple models side-by-side. "
             "Example: --compare openai/gpt-5.4-nano mistralai/mistral-small-2603",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        default=os.getenv("AUDIT_DRY_RUN", "").lower() in ("1", "true", "yes"),
        help="Run in mock mode (no API calls). Auto-enabled when no API key is set.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.getenv("AUDIT_OUTPUT_DIR", "outputs"),
        help="Directory to write output files. Default: outputs/",
    )
    parser.add_argument(
        "--output-csv",
        default=os.getenv("AUDIT_OUTPUT_CSV", "reliability_score.csv"),
        help="Filename for the compact reliability CSV. Default: reliability_score.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible mock runs.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional path to a JSON file with custom prompts (overrides built-in 100).",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=os.getenv("AUDIT_HTML", "").lower() in ("1", "true", "yes"),
        help="Generate a self-contained HTML report with charts.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=os.getenv("AUDIT_HISTORY", "").lower() in ("1", "true", "yes"),
        help="Append result to audit history and detect regressions.",
    )
    parser.add_argument(
        "--history-file",
        default=os.getenv("AUDIT_HISTORY_FILE", "outputs/audit_history.jsonl"),
        help="Path to the JSONL history file. Default: outputs/audit_history.jsonl",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=float(os.getenv("AUDIT_REGRESSION_THRESHOLD", "5.0")),
        help="Score-drop threshold (points) that triggers a regression warning. "
             "Default: 5.0",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """Configure logging level and format."""
    level   = logging.DEBUG if verbose else logging.WARNING
    fmt     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def load_custom_prompts(path: str) -> list[dict]:
    """Load and validate a JSON file of custom audit prompts."""
    import json
    with open(path) as f:
        prompts = json.load(f)
    required_keys = {"id", "tier", "prompt", "expected_keywords", "difficulty"}
    for i, p in enumerate(prompts):
        missing = required_keys - set(p.keys())
        if missing:
            raise ValueError(f"Prompt {i} missing keys: {missing}")
    return prompts


def _print_startup_banner(model: str, n_prompts: int, dry_run: bool, output_dir: str) -> None:
    """Print the Rich startup banner with project info and run parameters."""
    mode_str = "[yellow]DRY-RUN (mock — no network calls)[/yellow]" if dry_run else "[green]LIVE (OpenRouter API)[/green]"
    content = (
        f"[bold cyan]Stochastic Gap Audit[/bold cyan]  [dim]v{__version__}[/dim]  "
        f"[dim]· Built by NEO (heyneo.so)[/dim]\n\n"
        f"  [bold]Model[/bold]      : [cyan]{model}[/cyan]\n"
        f"  [bold]Mode[/bold]       : {mode_str}\n"
        f"  [bold]Prompts[/bold]    : {n_prompts}\n"
        f"  [bold]Output dir[/bold] : [dim]{output_dir}[/dim]"
    )
    console.print(Panel(content, title="[bold white]PRE-DEPLOYMENT RELIABILITY AUDIT[/bold white]", border_style="cyan"))


def _print_single_report(report, reporter) -> None:
    """Render the audit report as Rich tables and panels."""
    score = report.reliability_score

    # Colour-code by risk
    if score >= 80:
        score_style = "bold green"
        risk_label  = "LOW RISK"
        risk_style  = "green"
    elif score >= 60:
        score_style = "bold yellow"
        risk_label  = "MEDIUM RISK"
        risk_style  = "yellow"
    elif score >= 40:
        score_style = "bold red"
        risk_label  = "HIGH RISK"
        risk_style  = "red"
    else:
        score_style = "bold red"
        risk_label  = "CRITICAL RISK"
        risk_style  = "bold red"

    bar_width = 38
    filled    = int(score / 100 * bar_width)
    bar_green = "█" * filled
    bar_empty = "░" * (bar_width - filled)

    # ── Score panel ──────────────────────────────────────────────────────────
    score_content = (
        f"[{score_style}]{bar_green}[/][dim]{bar_empty}[/]  [{score_style}]{score:.2f}/100[/]\n"
        f"  95% CI: [[dim]{report.score_ci_low:.2f}[/], [dim]{report.score_ci_high:.2f}[/]]\n"
        f"  Risk level: [{risk_style}]{risk_label}[/]"
    )
    console.print(Panel(score_content, title="[bold]Reliability Score[/bold]", border_style=risk_style))

    # ── State distribution table ─────────────────────────────────────────────
    dist_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    dist_table.add_column("State",   style="bold")
    dist_table.add_column("Count",   justify="right")
    dist_table.add_column("Pct",     justify="right")
    n = report.total_prompts
    dist_table.add_row("[green]PASS[/]",      str(report.pass_count),      f"{report.pass_count/n*100:.1f}%")
    dist_table.add_row("[yellow]UNCERTAIN[/]", str(report.uncertain_count), f"{report.uncertain_count/n*100:.1f}%")
    dist_table.add_row("[red]FAIL[/]",        str(report.fail_count),      f"{report.fail_count/n*100:.1f}%")

    # ── Markov metrics table ─────────────────────────────────────────────────
    markov_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    markov_table.add_column("Metric")
    markov_table.add_column("Value", justify="right")
    markov_table.add_column("Meaning", style="dim")
    markov_table.add_row("Oversight cost",  f"{report.oversight_cost*100:.2f}%",              "fraction needing human review")
    markov_table.add_row("Stochastic gap",  f"{report.stochastic_gap:.4f}",                   "ideal − observed pass rate")
    markov_table.add_row("MFPT to FAIL",    f"{report.mean_first_passage_fail:.2f} steps",    "avg steps PASS → first FAIL")
    markov_table.add_row("π[PASS]",         f"{report.steady_state[0]:.4f}",                  "long-run PASS probability")
    markov_table.add_row("π[UNCERTAIN]",    f"{report.steady_state[1]:.4f}",                  "long-run UNCERTAIN probability")
    markov_table.add_row("[red]π[FAIL][/]", f"[red]{report.steady_state[2]:.4f}[/]",          "long-run FAIL probability")

    # ── Per-tier scores table ────────────────────────────────────────────────
    tier_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tier_table.add_column("Tier")
    tier_table.add_column("Pass rate", justify="right")
    tier_table.add_column("Grade")
    for tier, ts in sorted(report.tier_scores.items()):
        if ts >= 70:
            grade = "[green]✓ GOOD[/]"
        elif ts >= 50:
            grade = "[yellow]△ REVIEW[/]"
        else:
            grade = "[red]✗ POOR[/]"
        tier_table.add_row(tier, f"{ts:.1f}%", grade)

    console.print("\n[bold]State Distribution[/bold]")
    console.print(dist_table)
    console.print("[bold]Markov Metrics[/bold]")
    console.print(markov_table)
    console.print("[bold]Per-Tier Scores[/bold]")
    console.print(tier_table)


def _run_single(args, api_key: str, dry_run: bool, prompts: list, logger) -> int:
    """Run a single-model audit and return exit code."""
    _print_startup_banner(args.model, len(prompts), dry_run, args.output_dir)

    simulator = StochasticGapSimulator(
        model   = args.model,
        api_key = api_key or None,
        dry_run = dry_run,
        seed    = args.seed,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Markovian simulation…", total=len(prompts))

        _orig_run = simulator.run

        def _tracked_run(prompts=None):
            """Wrap simulator.run to tick the progress bar per prompt."""
            if prompts is None:
                from stochastic_gap_audit.prompts import AUDIT_PROMPTS as _AP
                prompts = _AP
            import time
            from stochastic_gap_audit.simulator import STATE_LABELS, PromptResult
            import numpy as np
            from datetime import datetime, timezone as _tz

            audit_ts  = datetime.now(tz=_tz.utc).isoformat()
            t_start   = time.perf_counter()
            results   = []
            transition_counts = np.zeros((3, 3), dtype=float)
            prev_state = None
            tier_state_totals: dict = {}
            from stochastic_gap_audit.simulator import grade_response
            from stochastic_gap_audit.prompts import TIER_WEIGHTS

            for i, p in enumerate(prompts):
                prompt_ts = datetime.now(tz=_tz.utc).isoformat()
                response, latency = simulator.client.call(p["prompt"], p["difficulty"])
                from stochastic_gap_audit.simulator import STATE_PASS, STATE_UNCERTAIN, STATE_FAIL
                state, kw_hits, kw_total = grade_response(
                    response, p["expected_keywords"], p["difficulty"], p["tier"]
                )
                weight     = TIER_WEIGHTS.get(p["tier"], 1.0)
                norm_score = (1.0 - state / 2.0) * weight

                result = PromptResult(
                    prompt_id     = p["id"],
                    tier          = p["tier"],
                    prompt        = p["prompt"],
                    response      = response,
                    state         = state,
                    state_label   = STATE_LABELS[state],
                    latency_ms    = latency,
                    keyword_hits  = kw_hits,
                    keyword_total = kw_total,
                    difficulty    = p["difficulty"],
                    weighted_score= norm_score,
                    timestamp     = prompt_ts,
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
                    description=f"[{state_colour}]{STATE_LABELS[state]:<9}[/] prompt {i+1}/{len(prompts)}",
                )

            execution_time = time.perf_counter() - t_start
            P   = simulator._normalise_transition_matrix(transition_counts)
            ss  = simulator._compute_steady_state(P)
            rel = simulator._compute_reliability_score(results, ss, len(prompts))
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

            from stochastic_gap_audit.simulator import SimulationReport
            return SimulationReport(
                model                   = simulator.model,
                dry_run                 = simulator.dry_run,
                results                 = results,
                transition_matrix       = P,
                steady_state            = ss,
                reliability_score       = rel,
                oversight_cost          = uncertain_count / len(results),
                stochastic_gap          = stochastic_gap,
                mean_first_passage_fail = mfpt,
                execution_time_s        = execution_time,
                tier_scores             = tier_scores,
                total_prompts           = len(results),
                pass_count              = pass_count,
                uncertain_count         = uncertain_count,
                fail_count              = fail_count,
                score_ci_low            = ci_low,
                score_ci_high           = ci_high,
                timestamp               = audit_ts,
            )

        report = _tracked_run(prompts)

    # ── Save standard outputs ──────────────────────────────────────────────
    reporter = AuditReporter(output_dir=args.output_dir)
    paths    = reporter.save_all(report)
    csv_path = reporter.save_reliability_csv(report, filename=args.output_csv)

    # ── Optional HTML report ───────────────────────────────────────────────
    if args.html:
        hr        = HTMLReporter(output_dir=args.output_dir)
        html_path = hr.save(report)
        paths["html"] = html_path

    # ── Optional history tracking ──────────────────────────────────────────
    if args.history:
        hist       = AuditHistory(history_file=args.history_file)
        hist.append(report)
        is_reg, delta = hist.detect_regression(report, threshold=args.regression_threshold)
        if is_reg:
            console.print(f"\n  [bold red]⚠  REGRESSION:[/bold red] score dropped {delta:+.2f} points vs. previous run!")
        elif delta is not None:
            console.print(f"\n  [dim]History: Δ={delta:+.2f} vs. previous run.[/dim]")
        for line in hist.trend_summary(model=report.model):
            console.print(f"  {line}")

    # ── Print Rich report ─────────────────────────────────────────────────
    _print_single_report(report, reporter)

    # ── File manifest table ───────────────────────────────────────────────
    file_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    file_table.add_column("File")
    file_table.add_column("Description")
    file_table.add_row(str(csv_path), "[green]main output[/] — reliability_score.csv")
    for fmt, p in paths.items():
        file_table.add_row(str(p), fmt)
    console.print("\n[bold]Output Files[/bold]")
    console.print(file_table)

    console.print(f"\n  [dim]Execution time: {report.execution_time_s:.2f} seconds[/dim]")

    exit_code = 0 if report.reliability_score >= 60 else 1
    if exit_code:
        console.print(f"\n[bold red]✗  Score {report.reliability_score:.2f} < 60 — model does NOT meet deployment threshold.[/bold red]")
    else:
        console.print(f"\n[bold green]✓  Score {report.reliability_score:.2f} ≥ 60 — model meets deployment threshold.[/bold green]")
    return exit_code


def _run_comparison(args, api_key: str, dry_run: bool, prompts: list) -> int:
    """Run multi-model comparison and return exit code (0 = best model ≥ 60)."""
    models = args.compare
    mode_str = "[yellow]DRY-RUN (mock)[/yellow]" if dry_run else "[green]LIVE (OpenRouter)[/green]"
    header = (
        f"[bold cyan]Stochastic Gap Audit[/bold cyan]  [dim]v{__version__}[/dim]  "
        f"[dim]· Built by NEO (heyneo.so)[/dim]\n\n"
        f"  [bold]Models[/bold]  : [cyan]{', '.join(models)}[/cyan]\n"
        f"  [bold]Prompts[/bold] : {len(prompts)}\n"
        f"  [bold]Mode[/bold]    : {mode_str}"
    )
    console.print(Panel(header, title="[bold white]MODEL COMPARISON[/bold white]", border_style="cyan"))

    comparator = ModelComparator(
        models     = models,
        api_key    = api_key or None,
        dry_run    = dry_run,
        seed       = args.seed,
        output_dir = args.output_dir,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running comparisons…", total=len(models))
        # Run each model sequentially, ticking the bar
        from stochastic_gap_audit.simulator import StochasticGapSimulator as _Sim
        from stochastic_gap_audit.comparator import ComparisonReport
        from datetime import datetime, timezone
        reports = {}
        for model in models:
            progress.update(task, description=f"[cyan]Auditing {model}…")
            sim = _Sim(model=model, api_key=api_key or None, dry_run=dry_run, seed=args.seed)
            reports[model] = sim.run(prompts=prompts)
            progress.advance(task)

    rankings = sorted(
        [(m, r.reliability_score) for m, r in reports.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    winner = rankings[0][0]
    ts     = datetime.now(timezone.utc).isoformat()
    comp   = ComparisonReport(
        models=models, reports=reports, timestamp=ts, winner=winner, rankings=rankings
    )
    csv_path = comparator.save_comparison(comp)

    if args.html:
        winner_report = comp.reports[comp.winner]
        hr = HTMLReporter(output_dir=args.output_dir)
        html_path = hr.save(winner_report)
        console.print(f"  HTML report (winner): [dim]{html_path}[/dim]")

    # ── Rich comparison table ─────────────────────────────────────────────
    cmp_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", title="Model Comparison")
    cmp_table.add_column("Rank",      justify="center")
    cmp_table.add_column("Model")
    cmp_table.add_column("Score",     justify="right")
    cmp_table.add_column("CI Low",    justify="right")
    cmp_table.add_column("CI High",   justify="right")
    cmp_table.add_column("Oversight", justify="right")
    cmp_table.add_column("MFPT",      justify="right")
    cmp_table.add_column("Verdict")
    for rank, (model, score) in enumerate(comp.rankings, 1):
        r      = comp.reports[model]
        crown  = "[bold green]★ WINNER[/bold green]" if model == comp.winner else ""
        style  = "green" if score >= 80 else "yellow" if score >= 60 else "red"
        cmp_table.add_row(
            str(rank),
            model,
            f"[{style}]{score:.2f}[/]",
            f"{r.score_ci_low:.2f}",
            f"{r.score_ci_high:.2f}",
            f"{r.oversight_cost*100:.1f}%",
            f"{r.mean_first_passage_fail:.2f}",
            crown,
        )
    console.print(cmp_table)
    console.print(f"\n  [dim]Comparison CSV: {csv_path}[/dim]")

    best_score = comp.rankings[0][1]
    return 0 if best_score >= 60 else 1


def main() -> int:
    """Entry point: parse args, run single audit or multi-model comparison."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("audit")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    dry_run = args.dry_run or not api_key

    if not api_key and not args.dry_run:
        console.print(
            "[yellow]⚠  OPENROUTER_API_KEY not set — running in DRY-RUN mode (mock responses).[/yellow]\n"
            "   [dim]Set the env var to use a live model.[/dim]"
        )

    prompts = AUDIT_PROMPTS
    if args.prompts_file:
        logger.info("Loading custom prompts from %s", args.prompts_file)
        prompts = load_custom_prompts(args.prompts_file)

    if args.compare:
        return _run_comparison(args, api_key, dry_run, prompts)
    return _run_single(args, api_key, dry_run, prompts, logger)


if __name__ == "__main__":
    sys.exit(main())
