# Stochastic-Gap-Audit – Pre-deployment reliability score, 5-min audit, 100% local

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-137%20passed-brightgreen.svg)]()

> Score any LLM's reliability before you ship — one CSV output, no dashboard, five minutes, works offline.

## The Problem  
Developers lack a simple, local tool to assess model reliability before deployment without relying on complex dashboards or external services. Existing workflows often require integrating multiple theoretical frameworks or manual analysis, which is time-consuming and error-prone. This project fills the gap by providing a fast, local audit that outputs actionable risk scores in CSV format.

## Who it's for  
This tool is for machine learning engineers and data scientists who need to quickly evaluate model reliability during pre-deployment checks. For example, a developer preparing a recommendation system for production can use this to identify potential oversight risks in under 5 minutes.


## Install

```bash
git clone https://github.com/dakshjain-1616/stochastic-gap-audit
cd stochastic-gap-audit
pip install -r requirements.txt
```

## Quickstart

```python
from stochastic_gap_audit import run_audit

# Run a 5-minute local reliability audit on 100 prompts
results = run_audit(
    model="gpt-4o",
    n_prompts=100,
    output_file="reliability_score.csv"
)

# Access the risk score directly
print(f"Reliability Score: {results['score']}")
```

## Key features

- Implements the "Stochastic Gap" framework from arXiv for theoretical reliability measurement.
- Runs Markovian simulation across 5 prompt tiers (math, code, factual, instruction, safety).
- Outputs actionable `reliability_score.csv` without requiring a dashboard or cloud infrastructure.
- 100% local execution with optional OpenRouter API integration for model access.

## Run tests

```bash
pytest tests/ -q
# 137 passed
```

## Project structure

```
stochastic-gap-audit/
├── stochastic_gap_audit/  ← main library
├── examples/              ← usage demos
├── tests/                 ← test suite
├── scripts/               ← demo scripts
└── requirements.txt
```