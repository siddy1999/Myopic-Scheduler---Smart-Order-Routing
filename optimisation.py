# optimisation.py
"""Core stochastic optimiser described in Cont & Kukanov (2017).

This file contains only the maths‑heavy pieces:
    •  `calculate_filled_amounts` – deterministic queue fill rule
    •  `cost_func`               – cost function v(X, ξ)
    •  `StochasticOrderOptimizer` – stochastic approximation solver
    •  `InsufficientDataError`   – thin custom exception

It **imports** stochastic inputs ξ from `market_metrics.generate_xi` and thus
has *no* direct dependency on Kafka, lookup tables, or file I/O.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from .market_metrics import generate_xi  # ξ‑sampler

__all__: list[str] = [
    "calculate_filled_amounts",
    "cost_func",
    "StochasticOrderOptimizer",
    "InsufficientDataError",
]

# ─────────────────────────────────────────────────────────────────────────────
# Queue‑fill rule
# ─────────────────────────────────────────────────────────────────────────────

def calculate_filled_amounts(Q: np.ndarray, L: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Eq. (2) in the paper – how much of each limit order is filled."""
    Q, L, xi = map(lambda x: np.asarray(x, dtype=float), (Q, L, xi))
    if not (Q.shape == L.shape == xi.shape):
        raise ValueError("Q, L, xi must have identical shapes")
    return np.maximum(xi - Q, 0) - np.maximum(xi - Q - L, 0)

# ─────────────────────────────────────────────────────────────────────────────
# Cost function
# ─────────────────────────────────────────────────────────────────────────────

def cost_func(X: np.ndarray, xi: np.ndarray, p: Dict) -> float:
    """Total cost v(X, ξ) [Eq. (5)]."""
    M, L = float(X[0]), X[1:]
    filled = calculate_filled_amounts(p["Q"], L, xi)
    total_filled = M + filled.sum()
    S = p["S"]

    market_cost = p["s"] * (p["h"] + p["f"]) * M
    limit_cost  = -p["s"] * np.sum((p["h"] + p["r"]) * filled)
    total_orders = M + L.sum()
    catch_up = max(S - total_filled, 0)

    impact    = p["theta"] * (total_orders + catch_up)
    underfill = p["lambda_u"] * max(S - total_filled, 0)
    overfill  = p["lambda_o"] * max(total_filled - S, 0)

    return market_cost + limit_cost + impact + underfill + overfill

# ─────────────────────────────────────────────────────────────────────────────
# Stochastic approximation optimiser
# ─────────────────────────────────────────────────────────────────────────────

class StochasticOrderOptimizer:
    """Implements Section 4 algorithm (descending step‑size)."""

    def __init__(self, params: Dict, n_exchanges: int):
        self.p = params
        self.K = n_exchanges
        self.dim = self.K + 1  # M + L₁…Lₖ

    # ── helper building blocks ──────────────────────────────────────────────
    def _grad(self, X: np.ndarray, xi: np.ndarray):
        M, L, Q = float(X[0]), X[1:], self.p["Q"]
        filled = calculate_filled_amounts(Q, L, xi)
        tot = M + filled.sum()
        shortfall = float(tot < self.p["S"])
        surplus   = float(tot > self.p["S"])
        g = np.zeros(self.dim)
        s = self.p["s"]
        signed_hf = s * (self.p["h"] + self.p["f"])
        g[0] = (signed_hf + self.p["theta"]
                 - (self.p["lambda_u"] + self.p["theta"]) * shortfall
                 + self.p["lambda_o"] * surplus)
        for k in range(self.K):
            isfilled = float(xi[k] > Q[k] + L[k])
            # signed cost of a filled limit order: -(h + r_k) flips with s
            signed_hr = -s * (self.p["h"] + self.p["r"][k])
            g[k + 1] = (self.p["theta"] + isfilled *
                         ( signed_hr
                          - (self.p["lambda_u"] + self.p["theta"]) * shortfall
                          + self.p["lambda_o"] * surplus))
        return g

    def _gamma(self, N: int):
        h, f, t, lu, lo = (self.p[x] for x in ("h", "f", "theta", "lambda_u", "lambda_o"))
        first = N * (h + f + t + lu + lo) ** 2
        second = sum(N * (h + rk + t + lu + lo) ** 2 for rk in self.p["r"])
        return (self.K ** 0.5 * self.p["S"]) / np.sqrt(first + second)

    def _project(self, X: np.ndarray):
        M, L = float(X[0]), np.maximum(X[1:], 0)
        S = self.p["S"]
        M = min(max(M, 0), S)
        L = np.minimum(L, S - M)
        if M + L.sum() < S:  # scale up proportionally
            scale = S / max(M + L.sum(), 1e-12)
            L *= scale
        return np.concatenate(([M], L))

    # ── public API ──────────────────────────────────────────────────────────
    def optimize(self, N: int = 1000, method: str = "exp") -> Tuple[np.ndarray, float]:
        X = np.full(self.dim, self.p["S"] / self.dim)
        X_acc = np.zeros_like(X)
        gamma = self._gamma(N)
        for _ in range(1, N + 1):
            xi = generate_xi(self.p["outflows"], self.p["T"], method)
            X -= gamma * self._grad(X, xi)
            X = self._project(X)
            X_acc += X
        X_star = X_acc / N
        # Monte‑Carlo estimate of V
        V_est = np.mean([cost_func(X_star, generate_xi(self.p["outflows"], self.p["T"]), self.p) for _ in range(100)])
        return X_star, float(V_est)


class InsufficientDataError(ValueError):
    """Raised when future price data is missing (adverse‑selection calc)."""
    pass
