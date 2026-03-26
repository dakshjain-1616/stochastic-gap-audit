"""
Tests for AuditHistory (trend tracking + regression detection feature).
"""

from __future__ import annotations

import json

import pytest

from stochastic_gap_audit.history import AuditHistory, HistoryEntry
from stochastic_gap_audit.simulator import SimulationReport, StochasticGapSimulator
from stochastic_gap_audit.prompts import AUDIT_PROMPTS


@pytest.fixture
def small_prompts() -> list[dict]:
    return AUDIT_PROMPTS[:8]


@pytest.fixture
def sample_report(small_prompts) -> SimulationReport:
    sim = StochasticGapSimulator(model="hist-test-model", dry_run=True, seed=1)
    return sim.run(prompts=small_prompts)


@pytest.fixture
def history(tmp_path) -> AuditHistory:
    return AuditHistory(history_file=str(tmp_path / "test_history.jsonl"))


class TestHistoryEntryFromReport:
    def test_round_trips_to_dict(self, sample_report):
        entry = HistoryEntry.from_report(sample_report)
        d     = entry.to_dict()
        entry2 = HistoryEntry.from_dict(d)
        assert entry2.model == sample_report.model
        assert entry2.reliability_score == sample_report.reliability_score

    def test_has_ci_fields(self, sample_report):
        entry = HistoryEntry.from_report(sample_report)
        assert hasattr(entry, "score_ci_low")
        assert hasattr(entry, "score_ci_high")
        assert 0.0 <= entry.score_ci_low <= entry.score_ci_high <= 100.0

    def test_has_tier_scores(self, sample_report):
        entry = HistoryEntry.from_report(sample_report)
        assert isinstance(entry.tier_scores, dict)
        assert len(entry.tier_scores) > 0


class TestAuditHistoryAppend:
    def test_file_created_on_append(self, history, sample_report):
        assert not history.history_file.exists()
        history.append(sample_report)
        assert history.history_file.exists()

    def test_file_is_valid_jsonl(self, history, sample_report):
        history.append(sample_report)
        with open(history.history_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "reliability_score" in parsed

    def test_multiple_appends_give_multiple_lines(self, history, sample_report):
        history.append(sample_report)
        history.append(sample_report)
        entries = history.load()
        assert len(entries) == 2


class TestAuditHistoryLoad:
    def test_empty_file_returns_empty_list(self, history):
        assert history.load() == []

    def test_load_filters_by_model(self, tmp_path, small_prompts):
        hist = AuditHistory(history_file=str(tmp_path / "h.jsonl"))
        sim_a = StochasticGapSimulator(model="model-a", dry_run=True, seed=10)
        sim_b = StochasticGapSimulator(model="model-b", dry_run=True, seed=20)
        hist.append(sim_a.run(prompts=small_prompts))
        hist.append(sim_b.run(prompts=small_prompts))

        a_entries = hist.load(model="model-a")
        b_entries = hist.load(model="model-b")
        assert len(a_entries) == 1
        assert a_entries[0].model == "model-a"
        assert len(b_entries) == 1

    def test_load_all_without_model_filter(self, tmp_path, small_prompts):
        hist = AuditHistory(history_file=str(tmp_path / "h.jsonl"))
        for i in range(3):
            sim = StochasticGapSimulator(model=f"m-{i}", dry_run=True, seed=i)
            hist.append(sim.run(prompts=small_prompts))
        assert len(hist.load()) == 3

    def test_corrupt_line_is_skipped(self, history, sample_report):
        history.append(sample_report)
        with open(history.history_file, "a") as f:
            f.write("NOT VALID JSON\n")
        entries = history.load()
        assert len(entries) == 1  # corrupt line skipped


class TestRegressionDetection:
    def test_no_history_returns_false_none(self, history, sample_report):
        is_reg, delta = history.detect_regression(sample_report)
        assert is_reg is False
        assert delta is None

    def test_identical_score_is_not_regression(self, history, sample_report):
        history.append(sample_report)
        is_reg, delta = history.detect_regression(sample_report)
        assert is_reg is False
        assert delta == 0.0

    def test_large_drop_is_regression(self, tmp_path, small_prompts):
        hist = AuditHistory(history_file=str(tmp_path / "r.jsonl"))
        # Build a fake high-score report, then a low one
        sim_high = StochasticGapSimulator(model="track-model", dry_run=True, seed=0)
        report_high = sim_high.run(prompts=small_prompts)

        # Manually set a very high score for the stored entry
        import dataclasses
        high_entry = HistoryEntry.from_report(report_high)
        high_entry = dataclasses.replace(high_entry, reliability_score=90.0)
        with open(hist.history_file, "a") as f:
            f.write(json.dumps(high_entry.to_dict()) + "\n")

        # Run a low-score report
        sim_low = StochasticGapSimulator(model="track-model", dry_run=True, seed=99)
        report_low = sim_low.run(prompts=small_prompts)
        # Force a low score
        report_low = dataclasses.replace(report_low, reliability_score=50.0)

        is_reg, delta = hist.detect_regression(report_low, threshold=5.0)
        assert is_reg is True
        assert delta == pytest.approx(-40.0)

    def test_improvement_is_not_regression(self, history, sample_report, small_prompts):
        history.append(sample_report)
        import dataclasses
        better = dataclasses.replace(
            sample_report, reliability_score=sample_report.reliability_score + 10
        )
        is_reg, delta = history.detect_regression(better)
        assert is_reg is False
        assert delta > 0


class TestTrendSummary:
    def test_empty_history_message(self, history):
        lines = history.trend_summary()
        assert any("No audit history" in l for l in lines)

    def test_summary_contains_model_name(self, history, sample_report):
        history.append(sample_report)
        lines = history.trend_summary()
        text  = "\n".join(lines)
        assert "hist-test-model" in text

    def test_summary_shows_delta_for_second_run(self, history, sample_report):
        history.append(sample_report)
        history.append(sample_report)
        lines = history.trend_summary(model=sample_report.model)
        # Second line after header should show a delta
        text = "\n".join(lines)
        assert "+" in text or "0.00" in text


class TestPercentileRank:
    def test_returns_none_with_no_history(self, history):
        assert history.percentile_rank(80.0) is None

    def test_highest_score_ranks_at_100(self, history, sample_report, small_prompts):
        import dataclasses
        for score in [60.0, 70.0, 80.0]:
            entry = HistoryEntry.from_report(sample_report)
            entry = dataclasses.replace(entry, reliability_score=score)
            with open(history.history_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        rank = history.percentile_rank(85.0)
        assert rank == 100.0

    def test_lowest_score_ranks_at_0(self, history, sample_report):
        import dataclasses
        for score in [70.0, 80.0, 90.0]:
            entry = HistoryEntry.from_report(sample_report)
            entry = dataclasses.replace(entry, reliability_score=score)
            with open(history.history_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        rank = history.percentile_rank(50.0)
        assert rank == 0.0
