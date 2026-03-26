"""
02_advanced_usage.py — Advanced features: tier scores, Markov metrics,
                        CSV/JSON/text output, and HTML report.

Runs the full 100-prompt audit in dry-run mode and saves all output formats.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stochastic_gap_audit import StochasticGapSimulator, AuditReporter, HTMLReporter
from stochastic_gap_audit.reporter import AuditReporter

# Run full 100-prompt audit with a fixed seed for reproducibility
sim = StochasticGapSimulator(model="mistralai/mistral-small-2603", dry_run=True, seed=42)
report = sim.run()

print(f"=== RELIABILITY SCORE: {report.reliability_score:.2f}/100 ===")
print(f"95% CI: [{report.score_ci_low:.2f}, {report.score_ci_high:.2f}]")
print()

# Markov chain metrics
print("--- MARKOV METRICS ---")
print(f"  Oversight cost  : {report.oversight_cost*100:.1f}%")
print(f"  Stochastic gap  : {report.stochastic_gap:.4f}")
print(f"  MFPT to FAIL    : {report.mean_first_passage_fail:.2f} steps")
print(f"  π[PASS]         : {report.steady_state[0]:.4f}")
print(f"  π[UNCERTAIN]    : {report.steady_state[1]:.4f}")
print(f"  π[FAIL]         : {report.steady_state[2]:.4f}")
print()

# Per-tier breakdown
print("--- PER-TIER SCORES ---")
for tier, score in sorted(report.tier_scores.items()):
    bar = "█" * int(score / 5)
    print(f"  {tier:<12}: {score:5.1f}%  {bar}")
print()

# Save all output formats
reporter = AuditReporter(output_dir="outputs")
paths = reporter.save_all(report)
print("--- SAVED FILES ---")
for fmt, path in paths.items():
    print(f"  {fmt}: {path}")

# Also save the canonical reliability_score.csv
csv_path = reporter.save_reliability_csv(report)
print(f"  reliability_score.csv: {csv_path}")

# Generate self-contained HTML report
html_reporter = HTMLReporter(output_dir="outputs")
html_path = html_reporter.save(report)
print(f"  html: {html_path}")
