"""T09 - Proactive rule pruning for UnimNeuron Safe.

The supplied ENF_UnimNeuron_Safe.py is NOT modified (it is B0).
Variants only add per-rule monitoring state and periodic pruning.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class CountingMixin:
    """Counts rule creations and capacity-triggered deletions (max_rules).

    The original _create_rule always appends one rule and, if max_rules is
    exceeded, deletes the least-supported rule. If the number of rules is the
    same before and after, a capacity deletion happened. A run with
    capacity_deletions > 0 is invalid for the main comparison.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_created = 0
        self.capacity_deletions = 0

    def _create_rule(self, x, y, sigma):
        before = self.n_rules_
        super()._create_rule(x, y, sigma)
        self.n_created += 1
        if self.n_rules_ == before:
            self.capacity_deletions += 1


class B0Counted(CountingMixin, EvolvingNeuroFuzzyUnimSafe):
    """B0: the original UnimNeuron Safe, unchanged, plus counters."""

    n_pruned_ = 0        # B0 never prunes proactively
    n_recreated = 0
    n_class_blocks = 0


class S1Fixed(CountingMixin, EvolvingNeuroFuzzyUnimSafe):
    """S1 - Fixed inactivity pruning.

    Each rule keeps `inactive`: observations since its last relevant
    activation (or since creation). A rule is removed when inactive > H.
    Usage is measured BEFORE learning the current label.
    """

    def __init__(
        self,
        *args,
        beta: float = 0.90,            # relative selectivity: a_r >= beta * a_max
        act_min_rel: float = 0.30,     # a_max must be >= this
        dist_factor_rel: float = 3.0,  # d_r <= dist_factor_rel * sigma_r
        H: int = 200,                  # max tolerated inactivity (observations)
        prune_every: int = 1,          # N: inspection interval
        recreate_window: int = 500,    # W: window to count a new rule as 'recreated'
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        self.act_min_rel = float(act_min_rel)
        self.dist_factor_rel = float(dist_factor_rel)
        self.H = int(H)
        self.prune_every = int(prune_every)
        self.recreate_window = int(recreate_window)
        self.n_recreated = 0
        # log of pruning events: (t, center) -> used to count recreated rules
        self.prune_log: List[Tuple[int, np.ndarray, float]] = []   # (t, center, sigma)
        self._recent_pruned: List[Tuple[int, np.ndarray, float]] = []
        self.n_class_blocks = 0   # removals blocked by the class-coverage safeguard

    # --------------------------------------------------------------
    # Main loop: monitor -> prune -> learn (original Safe) -> init new rules
    # --------------------------------------------------------------

    def _update_one(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=float).ravel()
        t = self.K + 1  # index of the current observation

        if self.n_rules_ > 0:
            # 1-3. measure usage on the CURRENT rules, before learning
            acts, dists = self._compute_rule_activations_and_dists(x)
            self._monitor(t, acts, dists)

            # 4. periodic pruning
            if t % self.prune_every == 0:
                self._prune(t)

        # 5. original UnimNeuron Safe learning (unchanged)
        super()._update_one(x, y)

        # 6. state for rules created in this step
        for r in self.rules:
            if not hasattr(r, "born_t"):
                self._init_state(r, t)
                self._check_recreated(r, t)

    # --------------------------------------------------------------
    # Pieces (kept separate so S2 can reuse / override them)
    # --------------------------------------------------------------

    def _check_recreated(self, r, t: int) -> None:
        """A new rule is 'recreated' if its center is within sigma (Chebyshev)
        of a rule pruned in the last W observations. Each pruned rule counts once."""
        W = self.recreate_window
        self._recent_pruned = [e for e in self._recent_pruned if t - e[0] <= W]
        for i, (tp, c_old, s_old) in enumerate(self._recent_pruned):
            if float(np.max(np.abs(r.center - c_old))) < s_old:
                self.n_recreated += 1
                del self._recent_pruned[i]
                return

    def _relevant_mask(self, acts: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """Boolean array: True where rule r had a relevant activation."""
        a_max = float(acts.max())
        if a_max < self.act_min_rel:
            return np.zeros(acts.shape[0], dtype=bool)
        sigmas = np.array([r.sigma for r in self.rules], dtype=float)
        return (acts >= self.beta * a_max) & (dists <= self.dist_factor_rel * sigmas)

    def _init_state(self, r, t: int) -> None:
        r.inactive = 0
        r.born_t = t

    def _monitor(self, t: int, acts: np.ndarray, dists: np.ndarray) -> None:
        relevant = self._relevant_mask(acts, dists)
        for r, is_rel in zip(self.rules, relevant):
            if is_rel:
                r.inactive = 0
            else:
                r.inactive += 1

    def _is_candidate(self, r, t: int) -> bool:
        return r.inactive > self.H

    def _select_candidates(self, t: int) -> np.ndarray:
        """Boolean mask: True where the rule meets the variant's pruning criterion."""
        return np.array([self._is_candidate(r, t) for r in self.rules], dtype=bool)

    def _keep_priority(self, r) -> float:
        """Higher = more worth keeping. S1/S2: most recently active."""
        return -float(r.inactive)

    def _prune(self, t: int) -> None:
        cand = self._select_candidates(t)
        if not cand.any():
            return

        # safeguard 1: class coverage -- never remove the last rule of a class
        dominant = np.array([int(np.argmax(r.class_probs())) for r in self.rules])
        for c in np.unique(dominant):
            idx = np.flatnonzero(dominant == c)
            if cand[idx].all():                       # every rule of class c would go
                best = max(idx, key=lambda i: self._keep_priority(self.rules[i]))
                cand[best] = False
                self.n_class_blocks += 1

        # safeguard 2: never leave the model empty
        if cand.all():
            best = max(range(len(self.rules)), key=lambda i: self._keep_priority(self.rules[i]))
            cand[best] = False

        for r, c in zip(self.rules, cand):
            if c:
                entry = (t, r.center.copy(), float(r.sigma))
                self.prune_log.append(entry)
                self._recent_pruned.append(entry)
        self.rules = [r for r, c in zip(self.rules, cand) if not c]

    @property
    def n_pruned_(self) -> int:
        return len(self.prune_log)


class S2Adaptive(S1Fixed):
    """S2 - Adaptive inactivity: H_r = H0 if n_r < n_min else max(H0, k * mean_gap_r).
    H (inherited from S1) plays the role of H0. With k = 0, S2 == S1."""

    def __init__(self, *args, k: float = 3.0, n_min: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = float(k)
        self.n_min = int(n_min)

    def _init_state(self, r, t: int) -> None:
        super()._init_state(r, t)   # inactive = 0, born_t = t
        r.n_act = 0
        r.mean_gap = 0.0

    def _monitor(self, t: int, acts: np.ndarray, dists: np.ndarray) -> None:
        relevant = self._relevant_mask(acts, dists)
        for r, is_rel in zip(self.rules, relevant):
            if is_rel:
                if r.n_act >= 1:
                    gap = r.inactive + 1
                    r.mean_gap += (gap - r.mean_gap) / r.n_act
                r.n_act += 1
                r.inactive = 0
            else:
                r.inactive += 1

    def tolerance(self, r) -> float:
        if r.n_act < self.n_min:
            return float(self.H)
        return max(float(self.H), self.k * r.mean_gap)

    def _is_candidate(self, r, t: int) -> bool:
        return r.inactive > self.tolerance(r)


class S3EMA(S1Fixed):
    """S3 - EMA usage pruning.

    u_r(t) = a_r / sum(a)            (normalized weight in prediction)
    E_r(t) = (1 - alpha) E_r(t-1) + alpha u_r(t)
    Candidate if E_r < theta * mean(E) and t - born_t >= G.
    New rules start with E = current mean(E) (neutral start).
    The beta / 0.30 / 3 sigma relevance test of S1-S2 is NOT used.
    """

    def __init__(self, *args, alpha: float = 0.01, theta: float = 0.3, G: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.G = int(G)

    def _mean_ema(self) -> float:
        vals = [r.ema for r in self.rules if hasattr(r, "ema")]
        return float(np.mean(vals)) if vals else 1.0

    def _init_state(self, r, t: int) -> None:
        r.born_t = t
        r.ema = self._mean_ema()

    def _monitor(self, t: int, acts: np.ndarray, dists: np.ndarray) -> None:
        s = float(acts.sum())
        u = acts / s if s > 0.0 else np.zeros_like(acts)
        for r, u_r in zip(self.rules, u):
            r.ema = (1.0 - self.alpha) * r.ema + self.alpha * float(u_r)

    def _select_candidates(self, t: int) -> np.ndarray:
        # threshold computed ONCE per inspection, before any removal
        thr = self.theta * self._mean_ema()
        return np.array([(t - r.born_t >= self.G) and (r.ema < thr) for r in self.rules], dtype=bool)

    def _keep_priority(self, r) -> float:
        """S3: highest EMA usage."""
        return float(r.ema)
