"""
Tests for ModelComparator (multi-model comparison feature).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stochastic_gap_audit.comparator import ComparisonReport, ModelComparator
from stochastic_gap_audit.prompts import AUDIT_PROMPTS


@pytest.fixture
def small_prompts() -> list[dict]:
    return AUDIT_PROMPTS[:10]


@pytest.fixture
def two_model_comp(tmp_path, small_prompts) -> ComparisonReport:
    comp = ModelComparator(
        models=["model-alpha", "model-beta"],
        dry_run=True,
        seed=7,
        output_dir=str(tmp_path),
    )
    return comp.run(prompts=small_prompts)


class TestModelComparatorInit:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one model"):
            ModelComparator(models=[], dry_run=True)

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        ModelComparator(["m"], dry_run=True, output_dir=str(out))
        assert out.exists()


class TestComparisonRun:
    def test_returns_comparison_report(self, two_model_comp):
        assert isinstance(two_model_comp, ComparisonReport)

    def test_both_models_present(self, two_model_comp):
        assert "model-alpha" in two_model_comp.reports
        assert "model-beta" in two_model_comp.reports

    def test_winner_is_highest_scorer(self, two_model_comp):
        winner_score = two_model_comp.reports[two_model_comp.winner].reliability_score
        for model, score in two_model_comp.rankings:
            assert winner_score >= score

    def test_rankings_sorted_descending(self, two_model_comp):
        scores = [s for _, s in two_model_comp.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_timestamp_present(self, two_model_comp):
        assert len(two_model_comp.timestamp) > 10

    def test_rankings_cover_all_models(self, two_model_comp):
        assert len(two_model_comp.rankings) == len(two_model_comp.models)

    def test_all_reports_have_valid_scores(self, two_model_comp):
        for model, report in two_model_comp.reports.items():
            assert 0.0 <= report.reliability_score <= 100.0, model

    def test_single_model_comparison(self, tmp_path, small_prompts):
        comp = ModelComparator(["solo-model"], dry_run=True, output_dir=str(tmp_path))
        result = comp.run(prompts=small_prompts)
        assert result.winner == "solo-model"
        assert len(result.rankings) == 1


class TestSaveComparison:
    def test_creates_csv_file(self, tmp_path, two_model_comp):
        comp = ModelComparator(["m1", "m2"], dry_run=True, output_dir=str(tmp_path))
        path = comp.save_comparison(two_model_comp)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_csv_has_correct_columns(self, tmp_path, two_model_comp):
        import csv
        comp = ModelComparator(["m1", "m2"], dry_run=True, output_dir=str(tmp_path))
        path = comp.save_comparison(two_model_comp)
        with open(path) as f:
            reader = csv.DictReader(f)
            cols   = reader.fieldnames or []
        for col in ("rank", "model", "reliability_score", "winner", "timestamp"):
            assert col in cols, f"Missing column: {col}"

    def test_csv_has_correct_row_count(self, tmp_path, two_model_comp):
        import csv
        comp = ModelComparator(["m1", "m2"], dry_run=True, output_dir=str(tmp_path))
        path = comp.save_comparison(two_model_comp)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # one row per model

    def test_rank_1_is_winner(self, tmp_path, two_model_comp):
        import csv
        comp = ModelComparator(["m1", "m2"], dry_run=True, output_dir=str(tmp_path))
        path = comp.save_comparison(two_model_comp)
        with open(path) as f:
            first_row = next(csv.DictReader(f))
        assert first_row["rank"] == "1"
        assert first_row["winner"] == "YES"


class TestFormatComparisonTable:
    def test_returns_list_of_strings(self, two_model_comp):
        lines = ModelComparator.format_comparison_table(two_model_comp)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_winner_marked(self, two_model_comp):
        lines = ModelComparator.format_comparison_table(two_model_comp)
        text  = "\n".join(lines)
        assert "WINNER" in text

    def test_both_models_in_table(self, two_model_comp):
        text = "\n".join(ModelComparator.format_comparison_table(two_model_comp))
        assert "model-alpha" in text
        assert "model-beta" in text
