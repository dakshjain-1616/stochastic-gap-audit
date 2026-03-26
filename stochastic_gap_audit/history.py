"""
Audit history tracker: persists audit results to an append-only JSONL file.

Enables:
  - Regression detection (score drop > threshold vs. previous run)
  - Trend analysis across multiple audit runs
  - Long-term reliability monitoring

Usage:
    from stochastic_gap_audit.history import AuditHistory
    hist = AuditHistory()
    entry = hist.append(report)
    is_regression, delta = hist.detect_regression(report)
    for line in hist.trend_summary(model="my-model"):
        print(line)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .simulator import SimulationReport

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """One persisted audit result."""
    timestamp:         str
    model:             str
    reliability_score: float
    score_ci_low:      float
    score_ci_high:     float
    oversight_cost:    float
    stochastic_gap:    float
    mfpt:              float
    pass_count:        int
    uncertain_count:   int
    fail_count:        int
    dry_run:           bool
    tier_scores:       Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryEntry":
        # Forward-compatible: ignore unknown keys
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_report(cls, report: SimulationReport) -> "HistoryEntry":
        return cls(
            timestamp         = report.timestamp or datetime.now(tz=timezone.utc).isoformat(),
            model             = report.model,
            reliability_score = report.reliability_score,
            score_ci_low      = report.score_ci_low,
            score_ci_high     = report.score_ci_high,
            oversight_cost    = report.oversight_cost,
            stochastic_gap    = report.stochastic_gap,
            mfpt              = report.mean_first_passage_fail,
            pass_count        = report.pass_count,
            uncertain_count   = report.uncertain_count,
            fail_count        = report.fail_count,
            dry_run           = report.dry_run,
            tier_scores       = report.tier_scores,
        )


class AuditHistory:
    """
    Append-only JSONL store for Stochastic Gap Audit results.

    Each line in the file is a JSON-encoded HistoryEntry.  The file grows
    monotonically — entries are never deleted or mutated, making it safe
    to use as an audit trail.
    """

    # Default regression threshold (score drop in points that triggers a warning)
    DEFAULT_REGRESSION_THRESHOLD = float(
        os.getenv("AUDIT_REGRESSION_THRESHOLD", "5.0")
    )

    def __init__(
        self,
        history_file: str = "outputs/audit_history.jsonl",
    ):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def append(self, report: SimulationReport) -> HistoryEntry:
        """Persist a new audit result and return the stored entry."""
        entry = HistoryEntry.from_report(report)
        with open(self.history_file, "a") as fh:
            fh.write(json.dumps(entry.to_dict()) + "\n")
        logger.info(
            "History: appended score=%.2f for model=%s at %s",
            entry.reliability_score, entry.model, entry.timestamp,
        )
        return entry

    def load(self, model: Optional[str] = None) -> List[HistoryEntry]:
        """
        Load all stored entries.
        If `model` is given, only entries for that model are returned.
        Corrupt / incomplete lines are skipped with a warning.
        """
        if not self.history_file.exists():
            return []
        entries: List[HistoryEntry] = []
        with open(self.history_file) as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                    entries.append(HistoryEntry.from_dict(d))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning("Skipping corrupt history line %d: %s", lineno, exc)
        if model:
            entries = [e for e in entries if e.model == model]
        return entries

    def detect_regression(
        self,
        report: SimulationReport,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        Compare `report` against the most recent entry for the same model.

        Returns:
            (is_regression, delta)
            - is_regression: True if score dropped by more than `threshold` points
            - delta: current_score − previous_score (negative = worse)
            Returns (False, None) when no prior run exists.
        """
        if threshold is None:
            threshold = self.DEFAULT_REGRESSION_THRESHOLD
        history = self.load(model=report.model)
        if not history:
            return False, None
        prev_score = history[-1].reliability_score
        delta      = report.reliability_score - prev_score
        is_reg     = delta < -abs(threshold)
        if is_reg:
            logger.warning(
                "REGRESSION detected for %s: %.2f → %.2f (Δ=%.2f, threshold=%.1f)",
                report.model, prev_score, report.reliability_score, delta, threshold,
            )
        return is_reg, round(delta, 4)

    def trend_summary(self, model: Optional[str] = None) -> List[str]:
        """
        Return a human-readable trend table as a list of strings.
        Shows every run with the score delta vs. the preceding run.
        """
        entries = self.load(model=model)
        if not entries:
            return ["  No audit history found."]

        label = f"for model '{model}'" if model else "(all models)"
        lines = [
            f"  AUDIT HISTORY {label}",
            f"  {'Timestamp':<28} {'Model':<38} {'Score':>7}  "
            f"{'CI':>17}  {'Delta':>7}",
            "  " + "─" * 102,
        ]
        prev_by_model: Dict[str, float] = {}
        for e in entries:
            prev  = prev_by_model.get(e.model)
            if prev is not None:
                delta = e.reliability_score - prev
                delta_str = f"{delta:+.2f}"
                reg_flag  = " (!)" if delta < -5.0 else ("  +" if delta > 5.0 else "   ")
            else:
                delta_str = "  NEW"
                reg_flag  = "   "
            ci = f"[{e.score_ci_low:.1f}, {e.score_ci_high:.1f}]"
            lines.append(
                f"  {e.timestamp:<28} {e.model:<38} {e.reliability_score:>7.2f}  "
                f"{ci:>17}  {delta_str:>7}{reg_flag}"
            )
            prev_by_model[e.model] = e.reliability_score
        return lines

    def percentile_rank(
        self, score: float, model: Optional[str] = None
    ) -> Optional[float]:
        """
        Return the percentile rank (0–100) of `score` among stored history.
        Returns None if history is empty.
        """
        entries = self.load(model=model)
        if not entries:
            return None
        all_scores = [e.reliability_score for e in entries]
        n_below = sum(1 for s in all_scores if s < score)
        return round(n_below / len(all_scores) * 100, 1)
