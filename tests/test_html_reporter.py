"""
Tests for HTMLReporter (self-contained HTML report generation feature).
"""

from __future__ import annotations

import pytest

from stochastic_gap_audit.html_reporter import HTMLReporter
from stochastic_gap_audit.simulator import SimulationReport, StochasticGapSimulator
from stochastic_gap_audit.prompts import AUDIT_PROMPTS


@pytest.fixture
def small_prompts() -> list[dict]:
    return AUDIT_PROMPTS[:10]


@pytest.fixture
def sample_report(small_prompts) -> SimulationReport:
    sim = StochasticGapSimulator(model="html-test-model", dry_run=True, seed=3)
    return sim.run(prompts=small_prompts)


@pytest.fixture
def reporter(tmp_path) -> HTMLReporter:
    return HTMLReporter(output_dir=str(tmp_path))


class TestHTMLReporterSave:
    def test_creates_html_file(self, reporter, sample_report):
        path = reporter.save(sample_report)
        assert path.exists()
        assert path.suffix == ".html"

    def test_file_is_non_empty(self, reporter, sample_report):
        path = reporter.save(sample_report)
        assert path.stat().st_size > 1000

    def test_custom_filename_respected(self, reporter, sample_report):
        path = reporter.save(sample_report, filename="custom_report.html")
        assert path.name == "custom_report.html"

    def test_output_dir_created_automatically(self, tmp_path, sample_report):
        out = tmp_path / "nested" / "reports"
        hr  = HTMLReporter(output_dir=str(out))
        path = hr.save(sample_report)
        assert path.exists()


class TestHTMLContent:
    def test_is_valid_html(self, reporter, sample_report):
        path    = reporter.save(sample_report)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_contains_model_name(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "html-test-model" in content

    def test_contains_reliability_score(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        score_str = f"{sample_report.reliability_score:.2f}"
        assert score_str in content

    def test_contains_ci_values(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "95% CI" in content

    def test_contains_state_labels(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        for label in ("PASS", "UNCERTAIN", "FAIL"):
            assert label in content

    def test_contains_tier_names(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        for tier in ("math", "code", "factual", "instruction", "safety"):
            assert tier in content

    def test_contains_markov_metrics(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "Oversight cost" in content
        assert "Stochastic gap" in content
        assert "MFPT" in content

    def test_contains_transition_matrix(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "Transition Matrix" in content

    def test_no_external_links(self, reporter, sample_report):
        """HTML should be fully self-contained — no CDN or external JS/CSS."""
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        # No script src pointing to external URLs
        assert 'src="http' not in content
        # No link rel=stylesheet pointing to external URLs
        assert 'href="http' not in content or "github.com" in content

    def test_per_prompt_table_present(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "<table>" in content
        assert "<th>" in content

    def test_contains_all_prompt_ids(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        for r in sample_report.results:
            assert f"<td>{r.prompt_id}</td>" in content

    def test_risk_label_present(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        risk_keywords = ("LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK")
        assert any(k in content for k in risk_keywords)


class TestHTMLReporterSVGCharts:
    def test_state_distribution_svg_present(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "<svg" in content

    def test_svg_contains_rect_elements(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        assert "<rect " in content

    def test_dry_run_label_present(self, reporter, sample_report):
        content = reporter.save(sample_report).read_text(encoding="utf-8")
        # dry_run=True in fixture
        assert "DRY-RUN" in content
