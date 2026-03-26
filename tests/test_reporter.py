"""Tests for the AuditReporter output module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochastic_gap_audit import AuditReporter, StochasticGapSimulator
from stochastic_gap_audit.simulator import SimulationReport


@pytest.fixture
def tmp_report_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audit_outputs"
    d.mkdir()
    return d


@pytest.fixture
def report() -> SimulationReport:
    sim = StochasticGapSimulator(model="test/model", dry_run=True, seed=7)
    return sim.run()


@pytest.fixture
def reporter(tmp_report_dir: Path) -> AuditReporter:
    return AuditReporter(output_dir=str(tmp_report_dir))


# ── save_all ───────────────────────────────────────────────────────────────────

class TestSaveAll:
    def test_returns_three_keys(self, reporter, report):
        paths = reporter.save_all(report)
        assert set(paths.keys()) == {"csv", "json", "summary"}

    def test_all_files_exist(self, reporter, report):
        paths = reporter.save_all(report)
        for key, path in paths.items():
            assert path.exists(), f"{key} file not created: {path}"

    def test_csv_file_non_empty(self, reporter, report):
        paths = reporter.save_all(report)
        assert paths["csv"].stat().st_size > 0

    def test_json_file_valid(self, reporter, report):
        paths = reporter.save_all(report)
        data = json.loads(paths["json"].read_text())
        assert "meta" in data
        assert "scores" in data
        assert "results" in data

    def test_summary_file_contains_score(self, reporter, report):
        paths = reporter.save_all(report)
        content = paths["summary"].read_text()
        assert "RELIABILITY SCORE" in content
        assert str(int(report.reliability_score)) in content

    def test_json_has_100_results(self, reporter, report):
        paths = reporter.save_all(report)
        data = json.loads(paths["json"].read_text())
        assert len(data["results"]) == 100

    def test_json_steady_state_sums_to_one(self, reporter, report):
        paths = reporter.save_all(report)
        data  = json.loads(paths["json"].read_text())
        ss    = data["steady_state"]
        total = ss["PASS"] + ss["UNCERTAIN"] + ss["FAIL"]
        assert abs(total - 1.0) < 1e-5

    def test_json_transition_matrix_shape(self, reporter, report):
        paths = reporter.save_all(report)
        data  = json.loads(paths["json"].read_text())
        matrix = data["transition_matrix"]["matrix"]
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)


# ── save_reliability_csv ───────────────────────────────────────────────────────

class TestReliabilityCSV:
    def test_creates_csv_file(self, reporter, report):
        path = reporter.save_reliability_csv(report, "reliability_score.csv")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_csv_has_expected_columns(self, reporter, report):
        path = reporter.save_reliability_csv(report)
        df   = pd.read_csv(path)
        required_cols = {
            "prompt_id", "tier", "state_label", "latency_ms",
            "keyword_hits", "keyword_total", "difficulty", "weighted_score",
        }
        assert required_cols.issubset(set(df.columns))

    def test_csv_has_101_rows(self, reporter, report):
        """100 prompt rows + 1 summary row."""
        path = reporter.save_reliability_csv(report)
        df   = pd.read_csv(path)
        assert len(df) == 101

    def test_csv_summary_row_has_reliability_score(self, reporter, report):
        path = reporter.save_reliability_csv(report)
        df   = pd.read_csv(path)
        summary_row = df[df["prompt_id"] == "SUMMARY"].iloc[0]
        assert float(summary_row["reliability_score"]) == report.reliability_score

    def test_state_labels_valid(self, reporter, report):
        path  = reporter.save_reliability_csv(report)
        df    = pd.read_csv(path)
        valid = {"PASS", "UNCERTAIN", "FAIL", ""}
        labels = set(df["state_label"].dropna().unique())
        assert labels.issubset(valid)


# ── format_summary ─────────────────────────────────────────────────────────────

class TestFormatSummary:
    def test_returns_list_of_strings(self, report):
        lines = AuditReporter.format_summary(report)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_contains_model_name(self, report):
        lines = AuditReporter.format_summary(report)
        text  = "\n".join(lines)
        assert report.model in text

    def test_contains_all_tiers(self, report):
        lines = AuditReporter.format_summary(report)
        text  = "\n".join(lines)
        for tier in report.tier_scores:
            assert tier in text

    def test_score_in_summary(self, report):
        lines = AuditReporter.format_summary(report)
        text  = "\n".join(lines)
        # Score should appear rounded to 2 dp
        assert str(round(report.reliability_score, 2)) in text

    def test_risk_label_present(self, report):
        lines  = AuditReporter.format_summary(report)
        text   = "\n".join(lines)
        labels = ["LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK"]
        assert any(label in text for label in labels)

    def test_transition_matrix_printed(self, report):
        lines = AuditReporter.format_summary(report)
        text  = "\n".join(lines)
        assert "PASS" in text
        assert "UNCERTAIN" in text
        assert "FAIL" in text


# ── Directory creation ─────────────────────────────────────────────────────────

class TestDirectoryHandling:
    def test_creates_output_dir_if_missing(self, tmp_path):
        new_dir  = tmp_path / "nested" / "output"
        reporter = AuditReporter(output_dir=str(new_dir))
        sim      = StochasticGapSimulator(model="m", dry_run=True, seed=0)
        single_prompt = [
            {"id": 1, "tier": "math", "prompt": "1+1=?",
             "expected_keywords": ["2"], "difficulty": 0.1}
        ]
        report = sim.run(prompts=single_prompt)
        reporter.save_all(report)
        assert new_dir.exists()
