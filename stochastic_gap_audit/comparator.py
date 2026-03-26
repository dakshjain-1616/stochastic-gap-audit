"""
Multi-model comparator: runs the same Stochastic Gap Audit against N models
and produces a ranked side-by-side comparison report.

Usage:
    from stochastic_gap_audit.comparator import ModelComparator
    comp = ModelComparator(["model-a", "model-b"], dry_run=True, seed=42)
    report = comp.run()
    comp.save_comparison(report)
    for line in comp.format_comparison_table(report):
        print(line)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .simulator import SimulationReport, StochasticGapSimulator
from .prompts import AUDIT_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Aggregated result of auditing multiple models on the same prompt set."""
    models: List[str]
    reports: Dict[str, SimulationReport]
    timestamp: str
    winner: str                  # model with highest reliability_score
    rankings: List[tuple]        # [(model, score), ...] sorted descending


class ModelComparator:
    """
    Runs the full Stochastic Gap Audit against several models and ranks them.

    All models use the same prompt set and (optionally) the same seed so
    results are directly comparable.
    """

    def __init__(
        self,
        models: List[str],
        api_key: Optional[str] = None,
        dry_run: bool = True,
        seed: Optional[int] = None,
        output_dir: str = "outputs",
    ):
        if not models:
            raise ValueError("ModelComparator requires at least one model.")
        self.models     = models
        self.api_key    = api_key
        self.dry_run    = dry_run
        self.seed       = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prompts=None) -> ComparisonReport:
        """Audit every model in self.models and return a ComparisonReport."""
        if prompts is None:
            prompts = AUDIT_PROMPTS

        reports: Dict[str, SimulationReport] = {}
        for model in self.models:
            logger.info("Comparing model: %s", model)
            sim = StochasticGapSimulator(
                model   = model,
                api_key = self.api_key,
                dry_run = self.dry_run,
                seed    = self.seed,
            )
            reports[model] = sim.run(prompts=prompts)

        rankings = sorted(
            [(m, r.reliability_score) for m, r in reports.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        winner = rankings[0][0]
        ts     = datetime.now(tz=timezone.utc).isoformat()

        return ComparisonReport(
            models   = self.models,
            reports  = reports,
            timestamp= ts,
            winner   = winner,
            rankings = rankings,
        )

    def save_comparison(
        self,
        comp: ComparisonReport,
        filename: str = "comparison.csv",
    ) -> Path:
        """Save side-by-side ranked comparison to CSV."""
        path = self.output_dir / filename
        rows = []
        for rank, (model, score) in enumerate(comp.rankings, 1):
            r = comp.reports[model]
            rows.append({
                "rank":              rank,
                "model":             model,
                "reliability_score": r.reliability_score,
                "score_ci_low":      r.score_ci_low,
                "score_ci_high":     r.score_ci_high,
                "oversight_cost":    round(r.oversight_cost, 4),
                "stochastic_gap":    round(r.stochastic_gap, 4),
                "mfpt_to_fail":      r.mean_first_passage_fail,
                "pass_count":        r.pass_count,
                "uncertain_count":   r.uncertain_count,
                "fail_count":        r.fail_count,
                "execution_time_s":  round(r.execution_time_s, 3),
                "pi_pass":           round(float(r.steady_state[0]), 4),
                "pi_uncertain":      round(float(r.steady_state[1]), 4),
                "pi_fail":           round(float(r.steady_state[2]), 4),
                "tier_math":         r.tier_scores.get("math", 0),
                "tier_code":         r.tier_scores.get("code", 0),
                "tier_factual":      r.tier_scores.get("factual", 0),
                "tier_instruction":  r.tier_scores.get("instruction", 0),
                "tier_safety":       r.tier_scores.get("safety", 0),
                "winner":            "YES" if model == comp.winner else "",
                "timestamp":         comp.timestamp,
            })

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Comparison CSV saved: %s", path)
        return path

    @staticmethod
    def format_comparison_table(comp: ComparisonReport) -> List[str]:
        """Return pretty-printed comparison table as a list of strings."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║         STOCHASTIC GAP AUDIT — MODEL COMPARISON          ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  {'Rank':<5} {'Model':<40} {'Score':>7}  {'CI-Low':>7}  "
            f"{'CI-High':>8}  {'Oversight':>10}  {'MFPT':>8}",
            "  " + "─" * 92,
        ]
        for rank, (model, score) in enumerate(comp.rankings, 1):
            r      = comp.reports[model]
            crown  = "  <-- WINNER" if model == comp.winner else ""
            lines.append(
                f"  {rank:<5} {model:<40} {score:>7.2f}  "
                f"{r.score_ci_low:>7.2f}  {r.score_ci_high:>8.2f}  "
                f"{r.oversight_cost*100:>9.1f}%  {r.mean_first_passage_fail:>8.2f}"
                f"{crown}"
            )
        lines += ["", f"  Timestamp : {comp.timestamp}", ""]
        return lines
