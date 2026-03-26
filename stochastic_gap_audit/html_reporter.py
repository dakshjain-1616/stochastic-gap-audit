"""
HTML report generator: produces a self-contained, single-file HTML report
with inline SVG charts and CSS.  No external dependencies or CDN links —
the file works offline.

Features:
  - Reliability score gauge
  - State distribution bar chart (SVG)
  - Per-tier score bar chart (SVG)
  - Transition matrix heatmap (SVG)
  - Full per-prompt results table (sortable via CSS)
  - Bootstrap confidence interval display

Usage:
    from stochastic_gap_audit.html_reporter import HTMLReporter
    hr = HTMLReporter(output_dir="outputs")
    path = hr.save(report)
    print(f"HTML report: {path}")
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .simulator import SimulationReport, STATE_LABELS


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117;
       color: #e0e0e0; line-height: 1.6; padding: 24px; }
h1 { font-size: 1.6rem; color: #7eb8f7; margin-bottom: 4px; }
h2 { font-size: 1.1rem; color: #9ec8fb; margin: 28px 0 10px; border-bottom: 1px solid #2a2d3e;
     padding-bottom: 6px; }
.meta { color: #8a8f9e; font-size: 0.85rem; margin-bottom: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px; margin-bottom: 28px; }
.card { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 10px;
        padding: 20px; }
.score-big { font-size: 3.5rem; font-weight: 700; line-height: 1;
             color: #7eb8f7; }
.score-sub { color: #8a8f9e; font-size: 0.9rem; margin-top: 4px; }
.risk-badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
              font-size: 0.8rem; font-weight: 600; margin-top: 10px; }
.risk-low      { background: #1a3d2b; color: #4ade80; }
.risk-medium   { background: #3d3a1a; color: #facc15; }
.risk-high     { background: #3d1a1a; color: #f87171; }
.risk-critical { background: #2d0a0a; color: #fca5a5; }
.ci-text { color: #8a8f9e; font-size: 0.82rem; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th { background: #1e2133; color: #9ec8fb; text-align: left; padding: 8px 10px;
     position: sticky; top: 0; }
td { padding: 6px 10px; border-bottom: 1px solid #1e2133; }
tr:hover td { background: #1e2133; }
.state-PASS      { color: #4ade80; font-weight: 600; }
.state-UNCERTAIN { color: #facc15; font-weight: 600; }
.state-FAIL      { color: #f87171; font-weight: 600; }
.tbl-wrap { max-height: 480px; overflow-y: auto; border-radius: 8px;
            border: 1px solid #2a2d3e; }
svg text { font-family: inherit; }
.label { fill: #8a8f9e; font-size: 11px; }
.value-label { fill: #e0e0e0; font-size: 11px; }
"""


def _risk_class(score: float) -> str:
    if score >= 80:
        return "risk-low"
    if score >= 60:
        return "risk-medium"
    if score >= 40:
        return "risk-high"
    return "risk-critical"


def _risk_label(score: float) -> str:
    if score >= 80:
        return "LOW RISK"
    if score >= 60:
        return "MEDIUM RISK"
    if score >= 40:
        return "HIGH RISK"
    return "CRITICAL RISK"


def _state_bar_svg(report: SimulationReport) -> str:
    """SVG horizontal stacked bar showing PASS/UNCERTAIN/FAIL proportions."""
    n     = report.total_prompts
    w     = 400
    h_svg = 60
    p_w   = int(report.pass_count / n * w)
    u_w   = int(report.uncertain_count / n * w)
    f_w   = w - p_w - u_w
    return (
        f'<svg width="{w}" height="{h_svg}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="10" width="{p_w}" height="28" fill="#16a34a" rx="3"/>'
        f'<rect x="{p_w}" y="10" width="{u_w}" height="28" fill="#ca8a04" rx="3"/>'
        f'<rect x="{p_w+u_w}" y="10" width="{f_w}" height="28" fill="#dc2626" rx="3"/>'
        f'<text x="6" y="29" fill="white" font-size="11">'
        f'PASS {report.pass_count/n*100:.0f}%</text>'
        + (
            f'<text x="{p_w+4}" y="29" fill="white" font-size="11">'
            f'UNC {report.uncertain_count/n*100:.0f}%</text>'
            if u_w > 40 else ""
        )
        + (
            f'<text x="{p_w+u_w+4}" y="29" fill="white" font-size="11">'
            f'FAIL {report.fail_count/n*100:.0f}%</text>'
            if f_w > 40 else ""
        )
        + f'<text x="0" y="54" class="label">{report.pass_count} PASS  '
        f'{report.uncertain_count} UNCERTAIN  {report.fail_count} FAIL</text>'
        f"</svg>"
    )


_ALL_TIERS = ("code", "factual", "instruction", "math", "safety")


def _tier_bar_svg(report: SimulationReport) -> str:
    """SVG horizontal bar chart of per-tier scores."""
    full_scores = {t: report.tier_scores.get(t, 0.0) for t in _ALL_TIERS}
    tiers  = sorted(full_scores.items())
    bar_h  = 22
    gap    = 8
    max_w  = 280
    total_h = len(tiers) * (bar_h + gap) + 10

    parts = [f'<svg width="400" height="{total_h}" xmlns="http://www.w3.org/2000/svg">']
    colors = {"math": "#3b82f6", "code": "#8b5cf6", "factual": "#06b6d4",
              "instruction": "#f59e0b", "safety": "#ef4444"}
    for i, (tier, score) in enumerate(tiers):
        y   = i * (bar_h + gap)
        bw  = int(score / 100 * max_w)
        col = colors.get(tier, "#6b7280")
        parts.append(
            f'<text x="0" y="{y+15}" class="label" font-size="11">{tier}</text>'
            f'<rect x="85" y="{y+2}" width="{bw}" height="{bar_h-4}" fill="{col}" rx="3"/>'
            f'<text x="{85+bw+5}" y="{y+15}" class="value-label">{score:.1f}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _matrix_heatmap_svg(report: SimulationReport) -> str:
    """SVG heatmap of the 3×3 transition matrix."""
    P     = report.transition_matrix
    cell  = 70
    pad   = 50
    size  = 3 * cell + pad
    labels = ["PASS", "UNC", "FAIL"]

    parts = [f'<svg width="{size+20}" height="{size+20}" xmlns="http://www.w3.org/2000/svg">']
    # Axis labels
    for j, lbl in enumerate(labels):
        parts.append(
            f'<text x="{pad + j*cell + cell//2}" y="18" '
            f'text-anchor="middle" class="label">{lbl}</text>'
        )
    for i, lbl in enumerate(labels):
        parts.append(
            f'<text x="{pad-6}" y="{pad + i*cell + cell//2 + 4}" '
            f'text-anchor="end" class="label">{lbl}</text>'
        )
    # Cells
    for i in range(3):
        for j in range(3):
            val     = float(P[i][j])
            alpha   = int(val * 200 + 30)
            color   = f"rgba(126,184,247,{val:.2f})"
            x       = pad + j * cell
            y       = pad + i * cell
            txt_col = "white" if val > 0.3 else "#8a8f9e"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                f'fill="{color}" rx="4"/>'
                f'<text x="{x+cell//2-1}" y="{y+cell//2+4}" '
                f'text-anchor="middle" fill="{txt_col}" font-size="12">'
                f'{val:.3f}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _results_table(report: SimulationReport) -> str:
    rows = []
    for r in report.results:
        s_cls  = f"state-{r.state_label}"
        prompt = html.escape(r.prompt[:80])
        ts     = html.escape(r.timestamp[:19].replace("T", " ") if r.timestamp else "")
        rows.append(
            f"<tr>"
            f"<td>{r.prompt_id}</td>"
            f"<td>{html.escape(r.tier)}</td>"
            f'<td class="{s_cls}">{r.state_label}</td>'
            f"<td>{r.latency_ms:.0f}</td>"
            f"<td>{r.keyword_hits}/{r.keyword_total}</td>"
            f"<td>{r.difficulty:.2f}</td>"
            f"<td>{r.weighted_score:.4f}</td>"
            f"<td>{ts}</td>"
            f'<td title="{html.escape(r.prompt)}">{prompt}…</td>'
            f"</tr>"
        )
    return "\n".join(rows)


class HTMLReporter:
    """Generates a self-contained HTML report from a SimulationReport."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: SimulationReport, filename: str = "") -> Path:
        """Render and save the HTML report. Returns the saved path."""
        if not filename:
            ts   = (report.timestamp or datetime.now(tz=timezone.utc).isoformat())[:19]
            slug = report.model.replace("/", "_").replace(":", "_")
            filename = f"{slug}_{ts.replace(':', '').replace('-', '')}_report.html"

        path = self.output_dir / filename
        path.write_text(self._render(report), encoding="utf-8")
        return path

    def _render(self, report: SimulationReport) -> str:
        score      = report.reliability_score
        risk_cls   = _risk_class(score)
        risk_lbl   = _risk_label(score)
        ts_display = (report.timestamp or "")[:19].replace("T", " ") + " UTC"
        mode       = "DRY-RUN (mock)" if report.dry_run else "LIVE (OpenRouter)"
        m_esc      = html.escape(report.model)

        state_svg   = _state_bar_svg(report)
        tier_svg    = _tier_bar_svg(report)
        matrix_svg  = _matrix_heatmap_svg(report)
        result_rows = _results_table(report)

        pi = report.steady_state
        P  = report.transition_matrix

        full_tier_scores = {t: report.tier_scores.get(t, 0.0) for t in _ALL_TIERS}
        tier_rows = "".join(
            f"<tr><td>{t}</td><td>{s:.1f}%</td></tr>"
            for t, s in sorted(full_tier_scores.items())
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stochastic Gap Audit — {m_esc}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Stochastic Gap Audit</h1>
<div class="meta">
  Model: <strong>{m_esc}</strong> &nbsp;|&nbsp;
  Mode: {html.escape(mode)} &nbsp;|&nbsp;
  Prompts: {report.total_prompts} &nbsp;|&nbsp;
  Time: {ts_display} &nbsp;|&nbsp;
  Execution: {report.execution_time_s:.2f}s
</div>

<div class="grid">
  <div class="card">
    <h2>Reliability Score</h2>
    <div class="score-big">{score:.2f}</div>
    <div class="score-sub">out of 100</div>
    <div class="ci-text">95% CI: [{report.score_ci_low:.2f}, {report.score_ci_high:.2f}]</div>
    <span class="risk-badge {risk_cls}">{risk_lbl}</span>
  </div>

  <div class="card">
    <h2>Markov Metrics</h2>
    <table>
      <tr><td>Oversight cost</td><td><strong>{report.oversight_cost*100:.2f}%</strong></td></tr>
      <tr><td>Stochastic gap</td><td><strong>{report.stochastic_gap:.4f}</strong></td></tr>
      <tr><td>MFPT to FAIL</td><td><strong>{report.mean_first_passage_fail:.2f} steps</strong></td></tr>
      <tr><td>π[PASS]</td><td><strong>{pi[0]:.4f}</strong></td></tr>
      <tr><td>π[UNCERTAIN]</td><td><strong>{pi[1]:.4f}</strong></td></tr>
      <tr><td>π[FAIL]</td><td><strong>{pi[2]:.4f}</strong></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Per-Tier Scores</h2>
    <table>{tier_rows}</table>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>State Distribution</h2>
    {state_svg}
  </div>
  <div class="card">
    <h2>Tier Breakdown</h2>
    {tier_svg}
  </div>
  <div class="card">
    <h2>Transition Matrix</h2>
    {matrix_svg}
  </div>
</div>

<h2>Per-Prompt Results ({report.total_prompts} prompts)</h2>
<div class="tbl-wrap">
<table>
<thead>
  <tr>
    <th>#</th><th>Tier</th><th>State</th><th>Latency (ms)</th>
    <th>Keywords</th><th>Difficulty</th><th>W.Score</th>
    <th>Timestamp</th><th>Prompt (truncated)</th>
  </tr>
</thead>
<tbody>
{result_rows}
</tbody>
</table>
</div>

<div class="meta" style="margin-top:24px">
  Generated by <strong>Stochastic-Gap-Audit v2.0</strong> &mdash;
  <a href="https://github.com/dakshjain-1616/stochastic-gap-audit"
     style="color:#7eb8f7">github.com/dakshjain-1616/stochastic-gap-audit</a>
</div>
</body>
</html>"""
