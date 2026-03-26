"""
Integration tests for the audit.py CLI entry-point.

Tests cover the three required scenarios from the spec:
  1. CSV output with reliability score 0-100
  2. 100 prompts finish in < 5 minutes
  3. Zero network calls in local/dry-run mode
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest


AUDIT_SCRIPT = str(Path(__file__).parent.parent / "audit.py")


# ── SPEC TEST 1: CSV output with reliability score 0-100 ─────────────────────

class TestCSVOutput:
    def test_creates_reliability_csv(self, tmp_path):
        result = subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--output-csv", "reliability_score.csv",
             "--seed", "42"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
        csv_path = tmp_path / "reliability_score.csv"
        assert csv_path.exists(), "reliability_score.csv was not created"

    def test_csv_reliability_score_in_range(self, tmp_path):
        subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--output-csv", "reliability_score.csv",
             "--seed", "42"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        csv_path = tmp_path / "reliability_score.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        summary = next(r for r in rows if r.get("prompt_id") == "SUMMARY")
        score = float(summary["reliability_score"])
        assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100]"

    def test_csv_has_100_data_rows(self, tmp_path):
        subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--seed", "0"],
            capture_output=True, text=True, timeout=120,
        )
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) >= 1
        # The main reliability_score.csv has 100 data rows + 1 summary
        reliability_csv = tmp_path / "reliability_score.csv"
        with open(reliability_csv) as f:
            rows = list(csv.DictReader(f))
        data_rows = [r for r in rows if r.get("prompt_id") != "SUMMARY"]
        assert len(data_rows) == 100


# ── SPEC TEST 2: < 5 minutes for 100 prompts ──────────────────────────────────

class TestExecutionTime:
    @pytest.mark.timeout(300)
    def test_100_prompts_under_5_minutes_via_cli(self, tmp_path):
        """Full CLI run over 100 prompts must complete in < 300 seconds."""
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model-perf",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--seed", "1"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - t0
        assert result.returncode in (0, 1)
        assert elapsed < 300.0, f"CLI took {elapsed:.1f}s, exceeds 5-minute limit"

    def test_execution_time_in_stdout(self, tmp_path):
        result = subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--seed", "2"],
            capture_output=True, text=True, timeout=120,
        )
        combined = result.stdout + result.stderr
        assert any(
            keyword in combined
            for keyword in ["Execution time", "execution time", "seconds", "SUMMARY"]
        )


# ── SPEC TEST 3: Zero network calls in local/dry-run mode ────────────────────

class TestNoNetworkCalls:
    def test_no_requests_post_in_dry_run(self):
        """SPEC TEST 3: dry-run must make 0 outbound HTTP calls."""
        with patch("requests.post") as mock_post, \
             patch("requests.get")  as mock_get:
            # Import after patching so the mock takes effect
            from stochastic_gap_audit import StochasticGapSimulator
            sim = StochasticGapSimulator(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                dry_run=True,
                seed=0,
            )
            sim.run()
            mock_post.assert_not_called()
            mock_get.assert_not_called()

    def test_dry_run_auto_enabled_without_api_key(self):
        """Without API key, client must be MockModelClient."""
        import os
        saved = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            from stochastic_gap_audit.simulator import MockModelClient, StochasticGapSimulator
            sim = StochasticGapSimulator(model="m", api_key=None, dry_run=False)
            assert isinstance(sim.client, MockModelClient)
        finally:
            if saved is not None:
                os.environ["OPENROUTER_API_KEY"] = saved


# ── Exit code tests ────────────────────────────────────────────────────────────

class TestExitCodes:
    def test_cli_exits_cleanly(self, tmp_path):
        result = subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--seed", "42"],
            capture_output=True, text=True, timeout=120,
        )
        # Exit 0 (pass) or 1 (risky model) — never 2+ (error)
        assert result.returncode in (0, 1), (
            f"Unexpected exit code {result.returncode}\n"
            f"STDERR: {result.stderr[:500]}"
        )

    def test_cli_prints_reliability_score(self, tmp_path):
        result = subprocess.run(
            [sys.executable, AUDIT_SCRIPT,
             "--model", "test/model",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--seed", "42"],
            capture_output=True, text=True, timeout=120,
        )
        assert "RELIABILITY SCORE" in result.stdout or "reliability" in result.stdout.lower()
