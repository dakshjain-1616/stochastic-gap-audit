"""
03_custom_config.py — Configure audit behaviour via environment variables.

Shows how to customise the model, output directory, seed, and thresholds
using either os.environ or a .env file loaded with python-dotenv.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# --- Option A: set env vars programmatically before importing ---
os.environ.setdefault("AUDIT_MODEL",        "openai/gpt-4o-mini")
os.environ.setdefault("AUDIT_OUTPUT_DIR",   "outputs/custom_run")
os.environ.setdefault("DEMO_SEED",          "99")
os.environ.setdefault("AUDIT_REGRESSION_THRESHOLD", "3.0")  # stricter regression guard

# --- Option B: load from a .env file (uncomment if you have one) ---
# from dotenv import load_dotenv
# load_dotenv()  # reads OPENROUTER_API_KEY, AUDIT_MODEL, etc. from .env

from stochastic_gap_audit import StochasticGapSimulator, AuditReporter
from stochastic_gap_audit.prompts import AUDIT_PROMPTS

# Read config from env (with sensible defaults)
model      = os.getenv("AUDIT_MODEL",      "mistralai/mistral-small-2603")
output_dir = os.getenv("AUDIT_OUTPUT_DIR", "outputs")
seed       = int(os.getenv("DEMO_SEED",    "42"))
api_key    = os.getenv("OPENROUTER_API_KEY")  # None → auto-enable dry-run

print(f"Config from environment:")
print(f"  model      = {model}")
print(f"  output_dir = {output_dir}")
print(f"  seed       = {seed}")
print(f"  api_key    = {'<set>' if api_key else '<not set — dry-run mode>'}")
print()

# Use only the first 20 prompts so this example runs quickly
prompts = AUDIT_PROMPTS[:20]

sim = StochasticGapSimulator(
    model   = model,
    api_key = api_key,      # None triggers dry-run automatically
    seed    = seed,
)
report = sim.run(prompts=prompts)

print(f"Reliability score: {report.reliability_score:.2f}/100  "
      f"({'DRY-RUN' if report.dry_run else 'LIVE'})")
print(f"Execution time   : {report.execution_time_s:.2f}s")

reporter = AuditReporter(output_dir=output_dir)
csv_path = reporter.save_reliability_csv(report)
print(f"CSV saved to     : {csv_path}")
