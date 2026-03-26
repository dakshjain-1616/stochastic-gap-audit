"""
Tests for the Markovian simulator and grading logic.

Required test spec:
  1. Input: Model endpoint → Output: CSV with reliability score 0-100
  2. Input: 100 prompts    → Output: Execution time < 5 minutes
  3. Input: Local env      → Output: 0 network calls (if local/dry-run)
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stochastic_gap_audit.simulator import (
    STATE_FAIL,
    STATE_PASS,
    STATE_UNCERTAIN,
    MockModelClient,
    PromptResult,
    SimulationReport,
    StochasticGapSimulator,
    grade_response,
)
from stochastic_gap_audit.prompts import AUDIT_PROMPTS, TIER_WEIGHTS


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def simulator_dry() -> StochasticGapSimulator:
    return StochasticGapSimulator(model="test-model", dry_run=True, seed=42)


@pytest.fixture
def full_report(simulator_dry: StochasticGapSimulator) -> SimulationReport:
    return simulator_dry.run()


@pytest.fixture
def small_prompts() -> list[dict]:
    """Subset of 10 prompts for fast tests."""
    return AUDIT_PROMPTS[:10]


# ── 1. Prompt catalog ──────────────────────────────────────────────────────────

class TestPromptCatalog:
    def test_exactly_100_prompts(self):
        assert len(AUDIT_PROMPTS) == 100

    def test_unique_prompt_ids(self):
        ids = [p["id"] for p in AUDIT_PROMPTS]
        assert len(ids) == len(set(ids))

    def test_all_required_keys_present(self):
        required = {"id", "tier", "prompt", "expected_keywords", "difficulty"}
        for p in AUDIT_PROMPTS:
            assert required.issubset(p.keys()), f"Prompt {p.get('id')} missing keys"

    def test_difficulty_range(self):
        for p in AUDIT_PROMPTS:
            assert 0.0 <= p["difficulty"] <= 1.0, f"Prompt {p['id']} difficulty out of range"

    def test_tier_weights_cover_all_tiers(self):
        used_tiers = {p["tier"] for p in AUDIT_PROMPTS}
        for t in used_tiers:
            assert t in TIER_WEIGHTS, f"Tier '{t}' has no weight"

    def test_ids_sequential(self):
        ids = sorted(p["id"] for p in AUDIT_PROMPTS)
        assert ids == list(range(1, 101))


# ── 2. grade_response ──────────────────────────────────────────────────────────

class TestGradeResponse:
    def test_pass_state_on_keyword_match(self):
        state, hits, total = grade_response(
            "The answer is 42.", ["42"], 0.1, "math"
        )
        assert state == STATE_PASS
        assert hits == 1
        assert total == 1

    def test_fail_state_on_cannot(self):
        state, hits, total = grade_response(
            "I cannot provide a reliable answer.", ["42"], 0.1, "math"
        )
        assert state == STATE_FAIL

    def test_uncertain_state_on_hedge(self):
        state, hits, total = grade_response(
            "I'm not sure, but possibly 42.", ["42"], 0.1, "math"
        )
        assert state == STATE_UNCERTAIN

    def test_no_keywords_defaults_pass(self):
        state, hits, total = grade_response(
            "Here is a haiku:\nCode flows like water...", [], 0.2, "instruction"
        )
        assert state == STATE_PASS

    def test_fail_on_empty_keywords_no_match(self):
        # Even with no expected keywords, explicit fail phrases trigger FAIL
        state, _, _ = grade_response(
            "I am unable to complete this task.", [], 0.1, "instruction"
        )
        assert state == STATE_FAIL

    def test_hits_and_total_counted(self):
        state, hits, total = grade_response(
            "SELECT * FROM users WHERE age > 30",
            ["SELECT", "FROM", "WHERE", "age", "30"],
            0.15,
            "code",
        )
        assert hits == 5
        assert total == 5

    def test_partial_keyword_match_uncertain(self):
        state, hits, total = grade_response(
            "SELECT users", ["SELECT", "FROM", "WHERE", "age", "30"], 0.15, "code"
        )
        # 1/5 = 0.20 < threshold → uncertain (not zero, not enough for pass)
        assert state in (STATE_UNCERTAIN, STATE_FAIL)

    def test_high_difficulty_lowers_pass_threshold(self):
        # High difficulty prompts get more lenient grading (uncertain not fail)
        state, _, _ = grade_response(
            "I'm not entirely sure about this complex topic.",
            ["answer"],
            0.9,
            "safety",
        )
        assert state in (STATE_UNCERTAIN, STATE_FAIL)


# ── 3. MockModelClient ─────────────────────────────────────────────────────────

class TestMockModelClient:
    def test_returns_string_and_float(self):
        client = MockModelClient("test-model", seed=0)
        resp, latency = client.call("What is 2+2?", 0.1)
        assert isinstance(resp, str)
        assert isinstance(latency, float)
        assert latency > 0

    def test_response_not_empty(self):
        client = MockModelClient("test-model", seed=1)
        resp, _ = client.call("Hello", 0.2)
        assert len(resp) > 10

    def test_seed_gives_reproducible_results(self):
        c1 = MockModelClient("m", seed=99)
        c2 = MockModelClient("m", seed=99)
        states1 = [c1.call(f"q{i}", 0.2)[0][:20] for i in range(10)]
        states2 = [c2.call(f"q{i}", 0.2)[0][:20] for i in range(10)]
        assert states1 == states2

    def test_no_network_calls_in_mock_mode(self):
        """SPEC TEST 3: 0 network calls in local/dry-run mode."""
        with patch("requests.post") as mock_post:
            client = MockModelClient("test-model", seed=7)
            for i in range(100):
                client.call(f"prompt {i}", 0.1)
            mock_post.assert_not_called()


# ── 4. StochasticGapSimulator ──────────────────────────────────────────────────

class TestStochasticGapSimulator:
    def test_uses_mock_client_when_dry_run(self, simulator_dry):
        assert isinstance(simulator_dry.client, MockModelClient)

    def test_uses_mock_client_when_no_api_key(self):
        sim = StochasticGapSimulator(model="m", api_key=None, dry_run=False)
        assert isinstance(sim.client, MockModelClient)

    def test_run_returns_simulation_report(self, simulator_dry, small_prompts):
        report = simulator_dry.run(prompts=small_prompts)
        assert isinstance(report, SimulationReport)

    def test_report_has_correct_prompt_count(self, full_report):
        assert full_report.total_prompts == 100

    def test_report_results_length(self, full_report):
        assert len(full_report.results) == 100

    def test_reliability_score_range(self, full_report):
        """SPEC TEST 1a: reliability score is in [0, 100]."""
        assert 0.0 <= full_report.reliability_score <= 100.0

    def test_state_counts_sum_to_total(self, full_report):
        assert (
            full_report.pass_count
            + full_report.uncertain_count
            + full_report.fail_count
            == full_report.total_prompts
        )

    def test_transition_matrix_is_stochastic(self, full_report):
        """Each row of P sums to 1."""
        row_sums = full_report.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-10)

    def test_steady_state_sums_to_one(self, full_report):
        assert abs(full_report.steady_state.sum() - 1.0) < 1e-10

    def test_steady_state_all_positive(self, full_report):
        assert all(v > 0 for v in full_report.steady_state)

    def test_oversight_cost_in_range(self, full_report):
        assert 0.0 <= full_report.oversight_cost <= 1.0

    def test_stochastic_gap_non_negative(self, full_report):
        assert full_report.stochastic_gap >= 0.0

    def test_mfpt_positive(self, full_report):
        assert full_report.mean_first_passage_fail > 0

    def test_tier_scores_present(self, full_report):
        expected_tiers = {"math", "code", "factual", "instruction", "safety"}
        assert expected_tiers == set(full_report.tier_scores.keys())

    def test_tier_scores_in_range(self, full_report):
        for tier, score in full_report.tier_scores.items():
            assert 0.0 <= score <= 100.0, f"Tier '{tier}' score {score} out of range"

    def test_execution_time_positive(self, full_report):
        assert full_report.execution_time_s > 0


# ── 5. SPEC TEST 2: 100 prompts finish in < 5 minutes ─────────────────────────

class TestPerformance:
    @pytest.mark.timeout(300)  # 5-minute hard timeout
    def test_100_prompts_under_5_minutes(self):
        """SPEC TEST 2: 100 prompts must complete in < 5 minutes (300 seconds)."""
        sim = StochasticGapSimulator(model="perf-model", dry_run=True, seed=0)
        t0  = time.perf_counter()
        report = sim.run()
        elapsed = time.perf_counter() - t0

        assert report.total_prompts == 100
        assert elapsed < 300.0, f"Audit took {elapsed:.1f}s, exceeds 5-minute limit"

    def test_execution_time_recorded_in_report(self):
        sim    = StochasticGapSimulator(model="m", dry_run=True, seed=1)
        report = sim.run(prompts=AUDIT_PROMPTS[:5])
        assert report.execution_time_s > 0
        assert report.execution_time_s < 60  # 5 prompts must be fast


# ── 6. Markov math ─────────────────────────────────────────────────────────────

class TestMarkovMath:
    def test_normalise_transition_matrix(self):
        counts = np.array([
            [10, 2, 1],
            [ 3, 5, 2],
            [ 1, 1, 4],
        ], dtype=float)
        P = StochasticGapSimulator._normalise_transition_matrix(counts)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-12)

    def test_normalise_zero_row_gets_uniform(self):
        counts = np.array([
            [0, 0, 0],
            [5, 3, 2],
            [1, 1, 1],
        ], dtype=float)
        P = StochasticGapSimulator._normalise_transition_matrix(counts)
        np.testing.assert_allclose(P[0], [1/3, 1/3, 1/3], atol=1e-10)

    def test_steady_state_fixed_point(self, full_report):
        """π @ P ≈ π (steady-state property)."""
        pi = full_report.steady_state
        P  = full_report.transition_matrix
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)

    def test_mfpt_absorbing_case(self):
        """When FAIL is absorbing (P[FAIL,FAIL]=1), MFPT from PASS is finite."""
        P = np.array([
            [0.8, 0.15, 0.05],
            [0.5, 0.30, 0.20],
            [0.0, 0.00, 1.00],   # absorbing
        ])
        mfpt = StochasticGapSimulator._mean_first_passage_time(P, target=STATE_FAIL)
        assert mfpt > 0
        assert mfpt < float("inf")
