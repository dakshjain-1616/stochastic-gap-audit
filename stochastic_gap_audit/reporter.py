"""
Output reporter: saves SimulationReport to CSV, JSON, and a text summary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .simulator import SimulationReport, STATE_LABELS


class AuditReporter:
    """Serialises audit results to multiple formats."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_all(self, report: SimulationReport, prefix: str = "") -> dict[str, Path]:
        """Save CSV, JSON, and summary text. Returns dict of saved paths."""
        ts      = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug    = report.model.replace("/", "_").replace(":", "_")
        base    = f"{prefix}{slug}_{ts}" if prefix else f"{slug}_{ts}"

        paths = {}
        paths["csv"]     = self._save_csv(report, base)
        paths["json"]    = self._save_json(report, base)
        paths["summary"] = self._save_summary(report, base)
        return paths

    def save_reliability_csv(
        self, report: SimulationReport, filename: str = "reliability_score.csv"
    ) -> Path:
        """Save the compact reliability_score.csv (as required by the spec)."""
        path = self.output_dir / filename
        self._save_csv(report, path.stem, force_path=path)
        return path

    # ── Formatters ─────────────────────────────────────────────────────────────

    def _save_csv(
        self,
        report: SimulationReport,
        base: str,
        force_path: Optional[Path] = None,
    ) -> Path:
        rows = []
        for r in report.results:
            rows.append(
                {
                    "prompt_id":     r.prompt_id,
                    "tier":          r.tier,
                    "prompt":        r.prompt,
                    "state":         r.state,
                    "state_label":   r.state_label,
                    "latency_ms":    round(r.latency_ms, 2),
                    "keyword_hits":  r.keyword_hits,
                    "keyword_total": r.keyword_total,
                    "hit_rate":      round(r.keyword_hits / max(r.keyword_total, 1), 4),
                    "difficulty":    r.difficulty,
                    "weighted_score": round(r.weighted_score, 6),
                    "timestamp":     r.timestamp,
                }
            )

        df = pd.DataFrame(rows)

        # Append aggregate summary row
        summary_row = {
            "prompt_id":      "SUMMARY",
            "tier":           "ALL",
            "prompt":         f"model={report.model}",
            "state":          "",
            "state_label":    "",
            "latency_ms":     round(df["latency_ms"].mean(), 2),
            "keyword_hits":   df["keyword_hits"].sum(),
            "keyword_total":  df["keyword_total"].sum(),
            "hit_rate":       round(
                df["keyword_hits"].sum() / max(df["keyword_total"].sum(), 1), 4
            ),
            "difficulty":     round(df["difficulty"].mean(), 4),
            "weighted_score": round(df["weighted_score"].mean(), 6),
            "timestamp":      report.timestamp,
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        # Metadata columns in the summary row
        meta_cols = {
            "reliability_score": report.reliability_score,
            "score_ci_low":      report.score_ci_low,
            "score_ci_high":     report.score_ci_high,
            "oversight_cost":    round(report.oversight_cost, 4),
            "stochastic_gap":    round(report.stochastic_gap, 4),
            "mfpt_to_fail":      report.mean_first_passage_fail,
            "pass_count":        report.pass_count,
            "uncertain_count":   report.uncertain_count,
            "fail_count":        report.fail_count,
            "execution_time_s":  round(report.execution_time_s, 3),
            "dry_run":           report.dry_run,
        }
        for col, val in meta_cols.items():
            df[col] = ""
            df.loc[df.index[-1], col] = val

        path = force_path or (self.output_dir / f"{base}.csv")
        df.to_csv(path, index=False)
        return path

    def _save_json(self, report: SimulationReport, base: str) -> Path:
        data = {
            "meta": {
                "model":             report.model,
                "dry_run":           report.dry_run,
                "timestamp":         report.timestamp or datetime.now(tz=timezone.utc).isoformat(),
                "total_prompts":     report.total_prompts,
                "execution_time_s":  round(report.execution_time_s, 3),
            },
            "scores": {
                "reliability_score":       report.reliability_score,
                "score_ci_low":            report.score_ci_low,
                "score_ci_high":           report.score_ci_high,
                "oversight_cost":          round(report.oversight_cost, 4),
                "stochastic_gap":          round(report.stochastic_gap, 4),
                "mean_first_passage_fail": report.mean_first_passage_fail,
                "pass_count":              report.pass_count,
                "uncertain_count":         report.uncertain_count,
                "fail_count":              report.fail_count,
            },
            "tier_scores": report.tier_scores,
            "transition_matrix": {
                "states": ["PASS", "UNCERTAIN", "FAIL"],
                "matrix": report.transition_matrix.tolist(),
            },
            "steady_state": {
                "PASS":      round(float(report.steady_state[0]), 6),
                "UNCERTAIN": round(float(report.steady_state[1]), 6),
                "FAIL":      round(float(report.steady_state[2]), 6),
            },
            "results": [
                {
                    "prompt_id":     r.prompt_id,
                    "tier":          r.tier,
                    "prompt":        r.prompt,
                    "response":      r.response[:300],   # truncate for readability
                    "state":         r.state,
                    "state_label":   r.state_label,
                    "latency_ms":    round(r.latency_ms, 2),
                    "keyword_hits":  r.keyword_hits,
                    "keyword_total": r.keyword_total,
                    "difficulty":    r.difficulty,
                }
                for r in report.results
            ],
        }

        path = self.output_dir / f"{base}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _save_summary(self, report: SimulationReport, base: str) -> Path:
        lines = self.format_summary(report)
        path  = self.output_dir / f"{base}_summary.txt"
        path.write_text("\n".join(lines))
        return path

    # ── Pretty printer ─────────────────────────────────────────────────────────

    @staticmethod
    def format_summary(report: SimulationReport) -> list[str]:
        bar_width = 40
        score     = report.reliability_score
        filled    = int(score / 100 * bar_width)
        bar       = "█" * filled + "░" * (bar_width - filled)

        risk_label = (
            "LOW RISK"      if score >= 80 else
            "MEDIUM RISK"   if score >= 60 else
            "HIGH RISK"     if score >= 40 else
            "CRITICAL RISK"
        )

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║         STOCHASTIC GAP AUDIT — RELIABILITY REPORT        ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Model          : {report.model}",
            f"  Mode           : {'DRY-RUN (mock)' if report.dry_run else 'LIVE (OpenRouter)'}",
            f"  Prompts        : {report.total_prompts}",
            f"  Execution time : {report.execution_time_s:.2f}s",
            "",
            f"  ┌─ RELIABILITY SCORE ────────────────────────────────────┐",
            f"  │  [{bar}]  {score:.2f}/100  │",
            f"  │  95% CI: [{report.score_ci_low:.2f}, {report.score_ci_high:.2f}]"
            f"{'':>32}│",
            f"  │  Risk level: {risk_label:<45}│",
            f"  └────────────────────────────────────────────────────────┘",
            "",
            "  STATE DISTRIBUTION",
            f"    PASS      : {report.pass_count:>3} / {report.total_prompts}  "
            f"({report.pass_count/report.total_prompts*100:.1f}%)",
            f"    UNCERTAIN : {report.uncertain_count:>3} / {report.total_prompts}  "
            f"({report.uncertain_count/report.total_prompts*100:.1f}%)",
            f"    FAIL      : {report.fail_count:>3} / {report.total_prompts}  "
            f"({report.fail_count/report.total_prompts*100:.1f}%)",
            "",
            "  MARKOV METRICS",
            f"    Oversight cost       : {report.oversight_cost*100:.2f}%  "
            "(fraction needing human review)",
            f"    Stochastic gap       : {report.stochastic_gap:.4f}  "
            "(ideal − observed pass rate)",
            f"    MFPT to FAIL         : {report.mean_first_passage_fail:.2f} steps  "
            "(mean steps from PASS to first FAIL)",
            "",
            "  STEADY-STATE DISTRIBUTION (π)",
            f"    π[PASS]      = {report.steady_state[0]:.4f}",
            f"    π[UNCERTAIN] = {report.steady_state[1]:.4f}",
            f"    π[FAIL]      = {report.steady_state[2]:.4f}",
            "",
            "  TRANSITION MATRIX P  (rows = from, cols = to: PASS/UNC/FAIL)",
        ]
        for i, row in enumerate(report.transition_matrix):
            state_name = STATE_LABELS[i]
            lines.append(
                f"    {state_name:<9}: [{row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}]"
            )

        lines += [
            "",
            "  PER-TIER SCORES",
        ]
        for tier, score_t in sorted(report.tier_scores.items()):
            emoji = "✓" if score_t >= 70 else "△" if score_t >= 50 else "✗"
            lines.append(f"    {emoji} {tier:<12}: {score_t:.1f}%")

        lines += ["", "  Output files saved to outputs/"]
        return lines
