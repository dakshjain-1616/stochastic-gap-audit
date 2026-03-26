"""
04_full_pipeline.py — End-to-end workflow showing all project capabilities.

Demonstrates:
  1. Single-model audit with CSV/JSON/HTML output
  2. Multi-model comparison
  3. History tracking and regression detection
  4. Trend summary across multiple runs
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stochastic_gap_audit import (
    StochasticGapSimulator,
    AuditReporter,
    HTMLReporter,
    ModelComparator,
    AuditHistory,
)
from stochastic_gap_audit.prompts import AUDIT_PROMPTS

OUTPUT_DIR   = "outputs/pipeline_demo"
SMALL_PROMPTS = AUDIT_PROMPTS[:15]  # use 15 prompts so the demo runs fast

# ── Step 1: Single-model audit ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Single-model audit")
print("=" * 60)

sim    = StochasticGapSimulator(model="model-alpha", dry_run=True, seed=1)
report = sim.run(prompts=SMALL_PROMPTS)

reporter = AuditReporter(output_dir=OUTPUT_DIR)
paths    = reporter.save_all(report)
csv_path = reporter.save_reliability_csv(report, filename="reliability_score.csv")

print(f"  Score   : {report.reliability_score:.2f}/100")
print(f"  CSV     : {csv_path}")
for fmt, p in paths.items():
    print(f"  {fmt:8}: {p}")

# ── Step 2: HTML report ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Self-contained HTML report")
print("=" * 60)

html_reporter = HTMLReporter(output_dir=OUTPUT_DIR)
html_path     = html_reporter.save(report, filename="audit_report.html")
print(f"  HTML report: {html_path}")

# ── Step 3: Multi-model comparison ────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Multi-model comparison")
print("=" * 60)

comparator   = ModelComparator(
    models     = ["model-alpha", "model-beta", "model-gamma"],
    dry_run    = True,
    seed       = 42,
    output_dir = OUTPUT_DIR,
)
comparison   = comparator.run(prompts=SMALL_PROMPTS)
comp_csv     = comparator.save_comparison(comparison, filename="comparison.csv")

for line in comparator.format_comparison_table(comparison):
    print(line)
print(f"  Comparison CSV: {comp_csv}")

# ── Step 4: History tracking & regression detection ───────────────────────────
print()
print("=" * 60)
print("STEP 4: History tracking & regression detection")
print("=" * 60)

history = AuditHistory(history_file=f"{OUTPUT_DIR}/audit_history.jsonl")

# Simulate two runs for the same model to show regression detection
for run_seed in [10, 20]:
    r = StochasticGapSimulator(model="model-alpha", dry_run=True, seed=run_seed)
    run_report = r.run(prompts=SMALL_PROMPTS)
    entry = history.append(run_report)
    is_reg, delta = history.detect_regression(run_report)
    print(f"  Run seed={run_seed}: score={run_report.reliability_score:.2f}  "
          f"regression={is_reg}  delta={delta}")

# Show trend summary
print()
for line in history.trend_summary(model="model-alpha"):
    print(line)

print()
print("Pipeline complete. All outputs saved to:", OUTPUT_DIR)
