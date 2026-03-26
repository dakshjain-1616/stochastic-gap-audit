# Examples

Runnable examples for Stochastic-Gap-Audit. Every script works offline — no API key required.

```bash
# From the project root:
python examples/01_quick_start.py
python examples/02_advanced_usage.py
python examples/03_custom_config.py
python examples/04_full_pipeline.py
```

| Script | What it demonstrates |
|--------|----------------------|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example — run 10 prompts, print the reliability score (10–15 lines) |
| [`02_advanced_usage.py`](02_advanced_usage.py) | Full 100-prompt audit with per-tier scores, Markov metrics, and CSV/JSON/HTML output |
| [`03_custom_config.py`](03_custom_config.py) | Configure model, output directory, seed, and regression threshold via environment variables or `.env` |
| [`04_full_pipeline.py`](04_full_pipeline.py) | End-to-end workflow: single-model audit → HTML report → multi-model comparison → history tracking + regression detection |

## Notes

- All scripts use `sys.path.insert(0, ...)` so they work from any directory without installing the package.
- Set `OPENROUTER_API_KEY` in your `.env` to run against a live model instead of the built-in mock.
- Output files are written to `outputs/` (or a subdirectory) by default.
