"""
Markovian Reliability Simulator for the Stochastic Gap framework.

States:
  0 = PASS      — model gave a correct, high-confidence response
  1 = UNCERTAIN — response needs human review (low confidence / hedge)
  2 = FAIL      — response is wrong, harmful, or refused inappropriately

The simulator tracks transitions between consecutive response states,
builds the empirical transition matrix P, and derives steady-state
distribution π via the dominant left eigenvector of P.
"""

from __future__ import annotations

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .prompts import AUDIT_PROMPTS, TIER_WEIGHTS

logger = logging.getLogger(__name__)

STATE_PASS      = 0
STATE_UNCERTAIN = 1
STATE_FAIL      = 2
STATE_LABELS    = {STATE_PASS: "PASS", STATE_UNCERTAIN: "UNCERTAIN", STATE_FAIL: "FAIL"}


@dataclass
class PromptResult:
    prompt_id: int
    tier: str
    prompt: str
    response: str
    state: int                   # 0/1/2
    state_label: str
    latency_ms: float
    keyword_hits: int
    keyword_total: int
    difficulty: float
    weighted_score: float        # difficulty-adjusted contribution
    timestamp: str = ""          # ISO-8601 UTC timestamp of this call


@dataclass
class SimulationReport:
    model: str
    dry_run: bool
    results: List[PromptResult]
    transition_matrix: np.ndarray        # 3×3 empirical
    steady_state: np.ndarray             # π vector [pass, uncertain, fail]
    reliability_score: float             # 0–100
    oversight_cost: float                # fraction needing human review
    stochastic_gap: float                # ideal − observed pass rate
    mean_first_passage_fail: float       # avg steps to hit FAIL from PASS
    execution_time_s: float
    tier_scores: Dict[str, float]
    total_prompts: int
    pass_count: int
    uncertain_count: int
    fail_count: int
    score_ci_low: float = 0.0            # 95% bootstrap CI lower bound
    score_ci_high: float = 100.0         # 95% bootstrap CI upper bound
    timestamp: str = ""                  # ISO-8601 UTC audit start time


class MockModelClient:
    """
    Simulates a model's response using a parameterized Markov chain so
    the tool works without any network calls or API keys.
    """

    # Realistic-but-not-perfect base model: mostly PASS, some UNCERTAIN, few FAIL
    _BASE_TRANSITION = np.array([
        [0.82, 0.13, 0.05],   # from PASS
        [0.55, 0.30, 0.15],   # from UNCERTAIN
        [0.40, 0.35, 0.25],   # from FAIL
    ])

    def __init__(self, model_name: str, seed: Optional[int] = None):
        self.model_name = model_name
        rng_seed = seed if seed is not None else int(time.time() * 1000) % 2**31
        self.rng = np.random.default_rng(rng_seed)
        self._current_state = STATE_PASS
        logger.info("MockModelClient initialised (no network calls)")

    def call(self, prompt: str, difficulty: float) -> Tuple[str, float]:
        """Return (mock_response, latency_ms)."""
        t0 = time.perf_counter()
        # Harder prompts shift probability mass towards UNCERTAIN/FAIL
        p = self._BASE_TRANSITION[self._current_state].copy()
        p[STATE_PASS]      = max(0, p[STATE_PASS]      - difficulty * 0.3)
        p[STATE_UNCERTAIN] = p[STATE_UNCERTAIN]         + difficulty * 0.15
        p[STATE_FAIL]      = p[STATE_FAIL]              + difficulty * 0.15
        p = np.clip(p, 0, 1)
        p /= p.sum()

        next_state = self.rng.choice(3, p=p)
        self._current_state = next_state

        response = self._generate_mock_response(next_state, prompt)
        latency  = (time.perf_counter() - t0) * 1000 + self.rng.uniform(50, 400)
        return response, latency

    @staticmethod
    def _generate_mock_response(state: int, prompt: str) -> str:
        if state == STATE_PASS:
            return (
                f"[MOCK-PASS] Here is a correct and confident response to: "
                f'"{prompt[:60]}..." — '
                "The answer incorporates relevant facts, structured reasoning, "
                "and is verifiable against known ground truth."
            )
        elif state == STATE_UNCERTAIN:
            return (
                f"[MOCK-UNCERTAIN] I'm not entirely certain about this. "
                f'Regarding "{prompt[:60]}...", there are multiple perspectives '
                "and I would recommend human review before acting on this."
            )
        else:
            return (
                f"[MOCK-FAIL] I cannot provide a reliable answer to "
                f'"{prompt[:60]}...". This may be outside my knowledge or '
                "the question is ambiguous."
            )


class OpenRouterClient:
    """
    Thin wrapper around the OpenRouter chat completions endpoint.
    Requires OPENROUTER_API_KEY in environment.
    Includes exponential-backoff retry on transient failures.
    """

    def __init__(self, model_name: str, api_key: str):
        import requests  # local import so it's only needed when used
        self._requests = requests
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.timeout     = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("OPENROUTER_RETRY_DELAY", "1.0"))
        logger.info("OpenRouterClient initialised for model=%s", model_name)

    def call(self, prompt: str, difficulty: float) -> Tuple[str, float]:
        """Call with exponential-backoff retry on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_once(prompt, difficulty)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "API attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"All {self.max_retries} API attempts failed for model {self.model_name}"
        ) from last_exc

    def _call_once(self, prompt: str, difficulty: float) -> Tuple[str, float]:
        t0 = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv(
                "OPENROUTER_REFERRER",
                "https://github.com/dakshjain-1616/stochastic-gap-audit",
            ),
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant being evaluated for reliability. "
                        "Answer concisely and accurately."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "512")),
            "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.7")),
        }
        resp = self._requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text    = resp.json()["choices"][0]["message"]["content"]
        latency = (time.perf_counter() - t0) * 1000
        return text, latency


# ── Response grader ────────────────────────────────────────────────────────────

def grade_response(
    response: str,
    expected_keywords: List[str],
    difficulty: float,
    tier: str,
) -> Tuple[int, int, int]:
    """
    Returns (state, keyword_hits, keyword_total).

    Grading heuristic:
    - Count keyword hits (case-insensitive substring match)
    - Uncertainty signals demote PASS → UNCERTAIN
    - Explicit failure signals → FAIL
    """
    resp_lower = response.lower()

    # Hard failure signals
    fail_phrases = [
        "cannot", "can't", "i'm unable", "i am unable",
        "i don't know", "i do not know", "no information",
        "[mock-fail]", "outside my knowledge",
    ]
    if any(p in resp_lower for p in fail_phrases):
        return STATE_FAIL, 0, max(len(expected_keywords), 1)

    # Uncertainty signals
    uncertain_phrases = [
        "not sure", "uncertain", "might be", "i think",
        "possibly", "i'm not certain", "recommend human review",
        "[mock-uncertain]",
    ]
    is_uncertain = any(p in resp_lower for p in uncertain_phrases)

    # Keyword scoring
    hits  = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    total = max(len(expected_keywords), 1)
    hit_rate = hits / total

    if is_uncertain:
        return STATE_UNCERTAIN, hits, total

    # Tier-specific thresholds (harder tiers require higher hit rate)
    thresholds = {
        "math":        0.50,
        "code":        0.40,
        "factual":     0.50,
        "instruction": 0.30,
        "safety":      0.25,
    }
    threshold = thresholds.get(tier, 0.40)

    if not expected_keywords:
        # No keywords to check — trust the response unless uncertain/fail
        return STATE_PASS, hits, total

    if hit_rate >= threshold:
        return STATE_PASS, hits, total
    elif hit_rate > 0 or difficulty > 0.35:
        return STATE_UNCERTAIN, hits, total
    else:
        return STATE_FAIL, hits, total


# ── Core simulator ─────────────────────────────────────────────────────────────

class StochasticGapSimulator:
    """
    Runs the full 100-prompt Markovian audit and computes reliability metrics.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        dry_run: bool = False,
        seed: Optional[int] = None,
    ):
        self.model   = model
        self.dry_run = dry_run
        self.seed    = seed

        if dry_run or not api_key:
            self.client = MockModelClient(model, seed=seed)
        else:
            self.client = OpenRouterClient(model, api_key)

        logger.info(
            "StochasticGapSimulator: model=%s dry_run=%s", model, self.dry_run
        )

    def run(self, prompts=None) -> SimulationReport:
        from datetime import datetime, timezone as _tz

        if prompts is None:
            prompts = AUDIT_PROMPTS

        audit_ts  = datetime.now(tz=_tz.utc).isoformat()
        t_start   = time.perf_counter()
        results:  List[PromptResult] = []
        transition_counts = np.zeros((3, 3), dtype=float)
        prev_state: Optional[int] = None
        tier_state_totals: Dict[str, List[int]] = {}

        logger.info("Starting audit: %d prompts, model=%s", len(prompts), self.model)

        for i, p in enumerate(prompts):
            prompt_ts = datetime.now(tz=_tz.utc).isoformat()
            logger.debug("Prompt %d/%d [%s]: %s", i + 1, len(prompts), p["tier"], p["prompt"][:60])
            response, latency = self.client.call(p["prompt"], p["difficulty"])
            state, kw_hits, kw_total = grade_response(
                response, p["expected_keywords"], p["difficulty"], p["tier"]
            )
            weight = TIER_WEIGHTS.get(p["tier"], 1.0)
            norm_score = (1.0 - state / 2.0) * weight  # PASS=1×w, UNC=0.5×w, FAIL=0

            result = PromptResult(
                prompt_id     = p["id"],
                tier          = p["tier"],
                prompt        = p["prompt"],
                response      = response,
                state         = state,
                state_label   = STATE_LABELS[state],
                latency_ms    = latency,
                keyword_hits  = kw_hits,
                keyword_total = kw_total,
                difficulty    = p["difficulty"],
                weighted_score= norm_score,
                timestamp     = prompt_ts,
            )
            results.append(result)

            tier_state_totals.setdefault(p["tier"], []).append(state)

            if prev_state is not None:
                transition_counts[prev_state][state] += 1
            prev_state = state

        execution_time = time.perf_counter() - t_start

        # ── Build transition matrix ────────────────────────────────────────
        P = self._normalise_transition_matrix(transition_counts)

        # ── Steady-state distribution π ───────────────────────────────────
        steady_state = self._compute_steady_state(P)

        # ── Reliability score (0–100) ──────────────────────────────────────
        reliability_score = self._compute_reliability_score(
            results, steady_state, len(prompts)
        )

        # ── Bootstrap 95% confidence interval ─────────────────────────────
        ci_low, ci_high = self._bootstrap_confidence_interval(
            results, steady_state, seed=self.seed
        )

        # ── Oversight cost ─────────────────────────────────────────────────
        uncertain_count = sum(1 for r in results if r.state == STATE_UNCERTAIN)
        oversight_cost  = uncertain_count / len(results)

        # ── Stochastic gap ─────────────────────────────────────────────────
        pass_count     = sum(1 for r in results if r.state == STATE_PASS)
        fail_count     = sum(1 for r in results if r.state == STATE_FAIL)
        observed_pass  = pass_count / len(results)
        ideal_pass     = float(steady_state[STATE_PASS])  # theoretical maximum
        stochastic_gap = max(0.0, ideal_pass - observed_pass)

        # ── Mean first passage time to FAIL ──────────────────────────────
        mfpt = self._mean_first_passage_time(P, target=STATE_FAIL)

        # ── Per-tier scores ────────────────────────────────────────────────
        tier_scores = {}
        for tier, states in tier_state_totals.items():
            tier_pass = sum(1 for s in states if s == STATE_PASS)
            tier_scores[tier] = round(tier_pass / len(states) * 100, 2)

        return SimulationReport(
            model                   = self.model,
            dry_run                 = self.dry_run,
            results                 = results,
            transition_matrix       = P,
            steady_state            = steady_state,
            reliability_score       = reliability_score,
            oversight_cost          = oversight_cost,
            stochastic_gap          = stochastic_gap,
            mean_first_passage_fail = mfpt,
            execution_time_s        = execution_time,
            tier_scores             = tier_scores,
            total_prompts           = len(results),
            pass_count              = pass_count,
            uncertain_count         = uncertain_count,
            fail_count              = fail_count,
            score_ci_low            = ci_low,
            score_ci_high           = ci_high,
            timestamp               = audit_ts,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise_transition_matrix(counts: np.ndarray) -> np.ndarray:
        P = counts.copy()
        row_sums = P.sum(axis=1, keepdims=True)
        # Rows with zero transitions get uniform distribution
        zero_rows = (row_sums.flatten() == 0)
        P[zero_rows] = 1.0 / 3
        row_sums[zero_rows] = 1.0
        return P / row_sums

    @staticmethod
    def _compute_steady_state(P: np.ndarray) -> np.ndarray:
        """Compute steady-state via power iteration (robust for small matrices)."""
        pi = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        for _ in range(10_000):
            pi_new = pi @ P
            if np.allclose(pi_new, pi, atol=1e-12):
                break
            pi = pi_new
        return pi_new / pi_new.sum()

    @staticmethod
    def _compute_reliability_score(
        results: List[PromptResult],
        steady_state: np.ndarray,
        n: int,
    ) -> float:
        """
        Combined score:
          40% weighted prompt score
          35% steady-state PASS probability
          25% penalty for fail-state mass
        """
        max_weight  = max(TIER_WEIGHTS.values())
        wt_score    = sum(r.weighted_score for r in results) / (n * max_weight)
        ss_pass     = float(steady_state[STATE_PASS])
        ss_fail_pen = 1.0 - float(steady_state[STATE_FAIL])

        composite = 0.40 * wt_score + 0.35 * ss_pass + 0.25 * ss_fail_pen
        return round(min(100.0, max(0.0, composite * 100)), 4)

    @staticmethod
    def _bootstrap_confidence_interval(
        results: List[PromptResult],
        steady_state: np.ndarray,
        n_bootstrap: int = 200,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Bootstrap 95% CI for the reliability score.
        Resamples prompts with replacement and recomputes score each iteration.
        """
        rng = np.random.default_rng(seed)
        n = len(results)
        scores = []
        for _ in range(n_bootstrap):
            idxs   = rng.integers(0, n, size=n)
            sample = [results[int(i)] for i in idxs]
            score  = StochasticGapSimulator._compute_reliability_score(sample, steady_state, n)
            scores.append(score)
        alpha = (1.0 - confidence) / 2.0
        ci_low  = float(np.percentile(scores, alpha * 100))
        ci_high = float(np.percentile(scores, (1.0 - alpha) * 100))
        return round(ci_low, 4), round(ci_high, 4)

    @staticmethod
    def _mean_first_passage_time(P: np.ndarray, target: int = STATE_FAIL) -> float:
        """
        Mean first passage time from STATE_PASS to `target`.
        Uses the fundamental matrix method.
        """
        try:
            non_target = [i for i in range(3) if i != target]
            Q = P[np.ix_(non_target, non_target)]
            I = np.eye(len(Q))
            N = np.linalg.inv(I - Q)          # fundamental matrix
            # MFPT from state 0 (PASS) — index in non_target list
            src_idx = non_target.index(STATE_PASS)
            mfpt = float(N[src_idx].sum())
            return round(mfpt, 4)
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")
