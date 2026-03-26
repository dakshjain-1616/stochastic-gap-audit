"""
01_quick_start.py — Minimal working example of Stochastic-Gap-Audit.

Runs 10 prompts in dry-run mode (no API key required) and prints the score.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stochastic_gap_audit import StochasticGapSimulator, AuditReporter
from stochastic_gap_audit.prompts import AUDIT_PROMPTS

# Use the first 10 prompts so this finishes in seconds
prompts = AUDIT_PROMPTS[:10]

# dry_run=True uses a seeded mock client — no API key needed
sim = StochasticGapSimulator(model="my-model", dry_run=True, seed=42)
report = sim.run(prompts=prompts)

print(f"Model         : {report.model}")
print(f"Prompts run   : {report.total_prompts}")
print(f"Reliability   : {report.reliability_score:.2f} / 100")
print(f"95% CI        : [{report.score_ci_low:.2f}, {report.score_ci_high:.2f}]")
print(f"PASS / UNC / FAIL: {report.pass_count} / {report.uncertain_count} / {report.fail_count}")
