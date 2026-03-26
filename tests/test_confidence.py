"""
Tests for bootstrap confidence interval and new SimulationReport fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from stochastic_gap_audit.simulator import (
    SimulationReport,
    StochasticGapSimulator,
    PromptResult,
    STATE_PASS,
)
from stochastic_gap_audit.prompts import AUDIT_PROMPTS


@pytest.fixture
def small_prompts() -> list[dict]:
    return AUDIT_PROMPTS[:15]


@pytest.fixture
def report(small_prompts) -> SimulationReport:
    sim = StochasticGapSimulator(model="ci-test-model", dry_run=True, seed=42)
    return sim.run(prompts=small_prompts)


class TestBootstrapCI:
    def test_ci_fields_present_on_report(self, report):
        assert hasattr(report, "score_ci_low")
        assert hasattr(report, "score_ci_high")

    def test_ci_within_valid_range(self, report):
        assert 0.0 <= report.score_ci_low <= 100.0
        assert 0.0 <= report.score_ci_high <= 100.0

    def test_ci_low_leq_score_leq_ci_high(self, report):
        # Score should be inside or near the CI
        # Allow slight float tolerance since bootstrap is random
        assert report.score_ci_low <= report.reliability_score + 1.0
        assert report.score_ci_high >= report.reliability_score - 1.0

    def test_ci_low_leq_ci_high(self, report):
        assert report.score_ci_low <= report.score_ci_high

    def test_bootstrap_is_reproducible_with_seed(self, small_prompts):
        sim1 = StochasticGapSimulator(model="m", dry_run=True, seed=77)
        sim2 = StochasticGapSimulator(model="m", dry_run=True, seed=77)
        r1   = sim1.run(prompts=small_prompts)
        r2   = sim2.run(prompts=small_prompts)
        assert r1.score_ci_low  == r2.score_ci_low
        assert r1.score_ci_high == r2.score_ci_high

    def test_bootstrap_static_method_directly(self, report):
        ci_low, ci_high = StochasticGapSimulator._bootstrap_confidence_interval(
            report.results, report.steady_state, n_bootstrap=50, seed=0
        )
        assert 0.0 <= ci_low <= 100.0
        assert 0.0 <= ci_high <= 100.0
        assert ci_low <= ci_high

    def test_narrow_ci_for_all_pass(self):
        """All-PASS results should yield a narrow, high CI."""
        results = []
        for i in range(20):
            results.append(PromptResult(
                prompt_id=i, tier="math", prompt="Q", response="A",
                state=STATE_PASS, state_label="PASS",
                latency_ms=100.0, keyword_hits=1, keyword_total=1,
                difficulty=0.1, weighted_score=1.0,
            ))
        pi = np.array([1.0, 0.0, 0.0])
        ci_low, ci_high = StochasticGapSimulator._bootstrap_confidence_interval(
            results, pi, n_bootstrap=100, seed=5
        )
        assert ci_low > 50.0   # should be high
        assert (ci_high - ci_low) < 30.0  # should be relatively narrow


class TestPromptResultTimestamp:
    def test_timestamp_field_present(self, report):
        for r in report.results:
            assert hasattr(r, "timestamp")

    def test_timestamp_is_non_empty(self, report):
        for r in report.results:
            assert isinstance(r.timestamp, str)
            assert len(r.timestamp) > 0

    def test_timestamps_look_like_iso8601(self, report):
        import re
        iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        for r in report.results:
            assert iso_pattern.match(r.timestamp), f"Bad timestamp: {r.timestamp}"


class TestSimulationReportTimestamp:
    def test_audit_timestamp_present(self, report):
        assert hasattr(report, "timestamp")
        assert isinstance(report.timestamp, str)
        assert len(report.timestamp) > 0

    def test_audit_timestamp_is_iso8601(self, report):
        import re
        iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        assert iso_pattern.match(report.timestamp), f"Bad timestamp: {report.timestamp}"
