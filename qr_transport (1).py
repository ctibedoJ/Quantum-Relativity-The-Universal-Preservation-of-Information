#!/usr/bin/env python3
"""
qr_transport.py

Universal algebraic Q-R / Q^alpha transport kernel.

Core theorem:
    R_i = s_* q_i^{alpha_i(q)}
    alpha_i(q) = log(R_i/s_*) / log(q_i)

Carrier-jump theorem:
    q_i -> q'_i
    alpha'_i = alpha_i log(q_i)/log(q'_i)
    s_* (q'_i)^{alpha'_i} = R_i

The code is domain-free: any positive spectral radius sequence and any
admissible carrier q_i>1 can be transported by the same law.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import csv
import hashlib
import json
import math

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _as_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(arr) == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def sha256_json(obj: Any) -> str:
    payload = json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_positive_radius(R: ArrayLike, eps: float = 1e-300) -> np.ndarray:
    R = _as_array(R, "R")
    if not np.all(np.isfinite(R)):
        raise ValueError("R contains non-finite values")
    if np.any(R <= 0):
        raise ValueError("R must be strictly positive")
    return np.maximum(R, eps)


def validate_carrier(q: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    q = _as_array(q, "q")
    if not np.all(np.isfinite(q)):
        raise ValueError("q contains non-finite values")
    if np.any(q <= 1.0):
        raise ValueError("q must satisfy q_i > 1")
    return np.maximum(q, 1.0 + eps)


def geometric_scale(R: ArrayLike) -> float:
    R = validate_positive_radius(R)
    return float(np.exp(np.mean(np.log(R))))


def alpha_from_radius(R: ArrayLike, q: ArrayLike, s_star: Optional[float] = None) -> np.ndarray:
    R = validate_positive_radius(R)
    q = validate_carrier(q)
    if len(R) != len(q):
        raise ValueError("R and q must have the same length")
    s = geometric_scale(R) if s_star is None else float(s_star)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("s_star must be positive and finite")
    return np.log(R / s) / np.log(q)


def radius_from_alpha(alpha: ArrayLike, q: ArrayLike, s_star: float) -> np.ndarray:
    alpha = _as_array(alpha, "alpha")
    q = validate_carrier(q)
    if len(alpha) != len(q):
        raise ValueError("alpha and q must have same length")
    s = float(s_star)
    if not math.isfinite(s) or s <= 0:
        raise ValueError("s_star must be positive and finite")
    return s * np.power(q, alpha)


def jump_alpha(alpha: ArrayLike, q_from: ArrayLike, q_to: ArrayLike) -> np.ndarray:
    alpha = _as_array(alpha, "alpha")
    q_from = validate_carrier(q_from)
    q_to = validate_carrier(q_to)
    if not (len(alpha) == len(q_from) == len(q_to)):
        raise ValueError("alpha, q_from, and q_to must have same length")
    return alpha * np.log(q_from) / np.log(q_to)


def metric(pred: ArrayLike, truth: ArrayLike) -> Dict[str, float]:
    pred = validate_positive_radius(pred)
    truth = validate_positive_radius(truth)
    if len(pred) != len(truth):
        raise ValueError("pred and truth must have same length")
    rel = float(np.linalg.norm(pred - truth) / max(float(np.linalg.norm(truth)), 1e-300))
    max_abs = float(np.max(np.abs(pred - truth)))
    return {
        "relative_l2_error": rel,
        "max_abs_error": max_abs,
        "spectral_match_percent": 100.0 if rel < 1e-12 else 100.0 * max(0.0, 1.0 - rel),
    }


@dataclass
class TransportState:
    R: np.ndarray
    q: np.ndarray
    s_star: float
    alpha: np.ndarray
    R_hat: np.ndarray

    def metric(self) -> Dict[str, float]:
        return metric(self.R_hat, self.R)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class JumpState:
    R: np.ndarray
    q_from: np.ndarray
    q_to: np.ndarray
    s_star: float
    alpha_from: np.ndarray
    alpha_to: np.ndarray
    R_hat_to: np.ndarray

    def metric(self) -> Dict[str, float]:
        return metric(self.R_hat_to, self.R)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))


def transport(R: ArrayLike, q: ArrayLike) -> TransportState:
    R = validate_positive_radius(R)
    q = validate_carrier(q)
    if len(R) != len(q):
        raise ValueError("R and q must have same length")
    s = geometric_scale(R)
    alpha = alpha_from_radius(R, q, s)
    R_hat = radius_from_alpha(alpha, q, s)
    return TransportState(R=R, q=q, s_star=s, alpha=alpha, R_hat=R_hat)


def jump_transport(state: TransportState, q_to: ArrayLike) -> JumpState:
    q_to = validate_carrier(q_to)
    if len(q_to) != len(state.q):
        raise ValueError("q_to must have same length as state.q")
    alpha_to = jump_alpha(state.alpha, state.q, q_to)
    R_hat_to = radius_from_alpha(alpha_to, q_to, state.s_star)
    return JumpState(
        R=state.R,
        q_from=state.q,
        q_to=q_to,
        s_star=state.s_star,
        alpha_from=state.alpha,
        alpha_to=alpha_to,
        R_hat_to=R_hat_to,
    )


def path_incidence_qbase(N: int, rank_insert: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if N < 2:
        raise ValueError("N must be >=2")
    I = np.zeros((N - 1, N), dtype=float)
    for k in range(N - 1):
        I[k, k] = -1.0
        I[k, k + 1] = 1.0
    L = I.T @ I
    Q = L @ L
    eig = np.clip(np.linalg.eigvalsh(Q), 0.0, None)
    lam = eig / max(float(eig.max()), 1e-300)
    rank = (np.arange(N, dtype=float) + 1.0) / (N + 1.0)
    q = 1.0 + lam + (rank if rank_insert else 0.0)
    return q, lam, rank


def permutation_carrier(q: ArrayLike, permutation: Optional[ArrayLike] = None) -> np.ndarray:
    q = validate_carrier(q)
    if permutation is None:
        perm = np.arange(len(q))
        if len(q) > 3:
            perm[1::2] = np.arange(len(q))[1::2][::-1]
    else:
        perm = np.asarray(permutation, dtype=int)
        if sorted(perm.tolist()) != list(range(len(q))):
            raise ValueError("permutation must permute 0..N-1")
    return q[perm]


def deterministic_random_carrier(N: int, seed: int = 42, rank: Optional[ArrayLike] = None) -> np.ndarray:
    if N < 2:
        raise ValueError("N must be >=2")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    M = A.T @ A
    Q = M @ M
    eig = np.clip(np.linalg.eigvalsh(Q), 0.0, None)
    lam = eig / max(float(eig.max()), 1e-300)
    if rank is None:
        r = (np.arange(N, dtype=float) + 1.0) / (N + 1.0)
    else:
        r = _as_array(rank, "rank")
        if len(r) != N:
            raise ValueError("rank length must equal N")
    return 1.0 + lam + r


def perturb_carrier(q: ArrayLike, eps: float = 1e-4, mode: str = "sin") -> np.ndarray:
    q = validate_carrier(q)
    N = len(q)
    i = np.arange(N, dtype=float)
    if mode == "sin":
        factor = 1.0 + eps * np.sin(2.0 * np.pi * (i + 1.0) / max(N, 1))
    elif mode == "cos":
        factor = 1.0 + eps * np.cos(2.0 * np.pi * (i + 1.0) / max(N, 1))
    elif mode == "linear":
        x = (i - np.mean(i)) / max(float(np.ptp(i)), 1e-300)
        factor = 1.0 + eps * x
    else:
        raise ValueError("mode must be sin, cos, or linear")
    return validate_carrier(q * factor)


def inject_radius(R: ArrayLike, kind: str = "amplitude", strength: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    R = validate_positive_radius(R)
    N = len(R)
    x = np.linspace(0.0, 1.0, N)
    if kind == "amplitude":
        g = np.full(N, float(strength))
    elif kind == "tilt":
        g = np.exp(float(strength) * (x - np.mean(x)))
    elif kind == "curved":
        c = x - np.mean(x)
        g = np.exp(float(strength) * (c * c - np.mean(c * c)))
    else:
        raise ValueError("kind must be amplitude, tilt, or curved")
    return R * g, g


def phase_scramble_radius(R: ArrayLike, seed: int = 42) -> np.ndarray:
    R = validate_positive_radius(R)
    x = np.log(R)
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(x - np.mean(x))
    phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, len(F)))
    phase[0] = 1.0
    if len(phase) > 1:
        phase[-1] = 1.0
    y = np.fft.irfft(F * phase, n=len(x)) + np.mean(x)
    return np.exp(y)


def local_gram_channel(*vectors: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vecs = [_as_array(v, f"vector_{k}") for k, v in enumerate(vectors)]
    if not vecs:
        raise ValueError("at least one vector required")
    N = len(vecs[0])
    if any(len(v) != N for v in vecs):
        raise ValueError("all vectors must have same length")

    def triad(x: np.ndarray) -> np.ndarray:
        return np.asarray([[x[max(0, i - 1)], x[i], x[min(N - 1, i + 1)]] for i in range(N)])

    tris = [triad(v) for v in vecs]
    pdet = []
    erank = []
    for i in range(N):
        A = np.vstack([T[i] for T in tris])
        A = A - A.mean(axis=1, keepdims=True)
        G = A @ A.T
        eig = np.clip(np.linalg.eigvalsh(G), 0.0, None)
        tol = max(float(eig.max()) * 1e-12, 1e-300)
        pos = eig[eig > tol]
        r = max(1, len(pos))
        p = float(np.prod(pos)) if len(pos) else 1e-300
        erank.append(r)
        pdet.append(p)
    pdet = np.asarray(pdet, dtype=float)
    erank = np.asarray(erank, dtype=float)
    pstar = float(np.exp(np.mean(np.log(np.maximum(pdet, 1e-300)))))
    det_rank = np.log(np.maximum(pdet, 1e-300) / pstar) / erank
    return pdet, erank, det_rank


def read_csv_radius_carrier(path: str | Path, dataset: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    path = Path(path)
    groups: Dict[str, list] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("dataset", "default")
            if dataset is not None and key != dataset:
                continue
            groups.setdefault(key, []).append(row)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, rows in groups.items():
        def col(name: str, default: float = float("nan")) -> np.ndarray:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(name, default)))
                except Exception:
                    vals.append(default)
            return np.asarray(vals, dtype=float)
        R = col("R")
        q = col("qbase")
        idx = col("index_value")
        rank = col("rank")
        lam = col("lambda_Q", 0.0)
        pdet = col("pdet")
        erank = col("effective_rank")
        mask = np.isfinite(R) & np.isfinite(q) & (R > 0) & (q > 1)
        n = int(np.sum(mask))
        out[key] = {
            "R": R[mask],
            "q": q[mask],
            "index": idx[mask] if len(idx) == len(mask) else np.arange(n),
            "rank": rank[mask] if len(rank) == len(mask) else (np.arange(n) + 1.0) / (n + 1.0),
            "lambda_Q": lam[mask] if len(lam) == len(mask) else np.zeros(n),
            "pdet": pdet[mask] if len(pdet) == len(mask) else np.full(n, np.nan),
            "effective_rank": erank[mask] if len(erank) == len(mask) else np.full(n, np.nan),
        }
    return out


def audit(R: ArrayLike, q: ArrayLike, seed: int = 42) -> Dict[str, Any]:
    state = transport(R, q)
    N = len(state.R)
    rank = (np.arange(N, dtype=float) + 1.0) / (N + 1.0)
    q_perm = permutation_carrier(state.q)
    jump_perm = jump_transport(state, q_perm)
    random_frames = []
    for k in range(4):
        qr = deterministic_random_carrier(N, seed + k, rank)
        jr = jump_transport(state, qr)
        random_frames.append({"seed": seed + k, "metric": jr.metric()})
    qbase_frames = {}
    for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
        qp = perturb_carrier(state.q, eps=eps)
        jp = jump_transport(state, qp)
        qbase_frames[f"sin_eps_{eps:g}"] = {"metric": jp.metric()}
    R_scr = phase_scramble_radius(state.R, seed + 777)
    scr_state = transport(R_scr, state.q)
    injections = {}
    for kind, strength in [("amplitude", 1.25), ("tilt", 0.15), ("curved", 0.08)]:
        Rg, g = inject_radius(state.R, kind=kind, strength=strength)
        ag = alpha_from_radius(Rg, state.q, state.s_star)
        pg = radius_from_alpha(ag, state.q, state.s_star)
        expected_shift = np.log(g) / np.log(state.q)
        injections[kind] = {
            "metric": metric(pg, Rg),
            "alpha_shift_max_abs_error": float(np.max(np.abs((ag - state.alpha) - expected_shift))),
        }
    pdet, erank, det_rank = local_gram_channel(np.log(state.R), np.log(state.q), rank)
    cert = {
        "engine": "universal_qr_transport_kernel_v1",
        "rule": {
            "core": "R_i=s_* q_i^{alpha_i(q)}",
            "jump": "alpha'_i=alpha_i log(q_i)/log(q'_i)",
            "universal_cross_domain": True,
        },
        "n": N,
        "canonical": {"metric": state.metric(), "s_star": state.s_star},
        "permutation_jump": {"metric": jump_perm.metric()},
        "deterministic_random_jumps": random_frames,
        "qbase_perturbation_jumps": qbase_frames,
        "phase_scrambled_signal": {"metric": scr_state.metric()},
        "injection_covariance": injections,
        "determinant_rank_channel": {
            "pdet_min": float(np.min(pdet)),
            "pdet_max": float(np.max(pdet)),
            "effective_rank_min": float(np.min(erank)),
            "effective_rank_max": float(np.max(erank)),
        },
    }
    cert = _clean(cert)
    cert["certificate_hash"] = sha256_json(cert)
    return cert
