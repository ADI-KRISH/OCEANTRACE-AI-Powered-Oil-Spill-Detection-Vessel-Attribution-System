"""Suspect scoring: a transparent baseline and a learned ranker.

Both expose the same interface -- take the feature table, return it with a
`score`, an `attribution_pct` and a plain-language `evidence` list per vessel --
so the pipeline can swap them without knowing which is in play.

The baseline is deliberately kept as hand-set weighted log-odds rather than a
fitted logistic model. It needs no labels, it always runs, and every number in it
can be defended line by line to someone who has to act on the output. The learned
ranker is more accurate where training data exists; the baseline is what you ship
when it does not, and what you fall back to when the ranker is out of domain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Baseline: weighted log-odds over transformed features
# ---------------------------------------------------------------------------

#: Weight per evidence term, in log-odds. Positive = more suspicious.
BASELINE_WEIGHTS = {
    "prox": 3.0,           # was the vessel where the oil started?
    "near": 2.0,           # closest approach, saturating
    "dwell": 1.2,          # did it linger in the area?
    "timing": 0.8,         # do we actually observe it around that time?
    "slow": 0.9,           # slow-steaming discharge signature
    "loiter": 0.7,         # aimless low-speed manoeuvring
    "dark_gap": 1.5,       # went dark over the origin
    "dark_frac": 0.5,      # generally dark during the window
    "axis": 1.0,           # course parallel to the slick's long axis
    "vtype": 1.0,          # tanker/cargo prior
    "size": 0.4,           # slop capacity proxy
}
BASELINE_BIAS = -3.2

#: Thresholds above which a term is worth stating in the evidence list.
_EVIDENCE_MIN = 0.25


def _transform(f: pd.DataFrame) -> pd.DataFrame:
    """Map raw features onto bounded [0, 1] (or signed) evidence terms.

    Distances and times are turned into saturating scores so that a vessel 200 km
    away is not penalised a thousand times more than one 20 km away -- past a
    point, further is just "not a suspect".
    """
    t = pd.DataFrame(index=f.index)
    t["prox"] = f.prox_score
    t["near"] = np.exp(-f.min_dist_km / 8.0)
    t["dwell"] = f.dwell_frac
    t["timing"] = np.exp(-f.time_gap_min / 45.0)
    t["slow"] = f.slow_steaming
    t["loiter"] = f.loiter_score
    t["dark_gap"] = f.gap_over_origin
    t["dark_frac"] = f.dark_frac
    # Centred: a course *perpendicular* to the slick axis is exculpatory, so this
    # term is allowed to go negative rather than merely to zero.
    t["axis"] = 2.0 * (f.course_align - 0.5)
    t["vtype"] = f.vtype_prior
    t["size"] = f.size_score
    return t


_EVIDENCE_TEXT = {
    "prox": "Position overlaps the estimated discharge point",
    "near": "Passed within {min_dist_km:.1f} km of the origin",
    "dwell": "Spent {dwell_pct:.0f}% of the window inside the search radius",
    "timing": "AIS fix within {time_gap_min:.0f} min of the discharge time",
    "slow": "Slow-steaming ({slow_pct:.0f}% of fixes at 2-9 kn)",
    "loiter": "Low-speed manoeuvring with unstable heading",
    "dark_gap": "AIS went dark over the origin ({ais_gap_max_min:.0f} min)",
    "dark_frac": "Reporting gaps cover {dark_pct:.0f}% of the discharge window",
    "axis": "Course aligns with the slick's long axis",
    "vtype": "{vtype_label} -- elevated prior for oil discharge",
    "size": "Large vessel ({length:.0f} m)",
}


def _vtype_label(vtype: float) -> str:
    if vtype is None or not np.isfinite(vtype):
        return "Vessel type unknown"
    v = int(vtype)
    if 80 <= v <= 89:
        return "Tanker"
    if 70 <= v <= 79:
        return "Cargo vessel"
    if v == 30:
        return "Fishing vessel"
    if 31 <= v <= 32:
        return "Towing vessel"
    if 50 <= v <= 59:
        return "Special craft"
    if 60 <= v <= 69:
        return "Passenger vessel"
    if v in (36, 37):
        return "Sailing / pleasure craft"
    return f"Vessel type {v}"


def _evidence_for(row, terms, contribs, top_k: int = 4) -> list[str]:
    """Plain-language justification, strongest contribution first."""
    order = sorted(contribs.items(), key=lambda kv: -kv[1])
    ctx = {
        "min_dist_km": row.min_dist_km,
        "dwell_pct": row.dwell_frac * 100,
        "time_gap_min": row.time_gap_min,
        "slow_pct": row.slow_steaming * 100,
        "ais_gap_max_min": row.ais_gap_max_min,
        "dark_pct": row.dark_frac * 100,
        "vtype_label": _vtype_label(row.vtype),
        "length": row.length if np.isfinite(row.length) else 0.0,
    }
    out = []
    for name, contrib in order:
        if contrib < _EVIDENCE_MIN or len(out) >= top_k:
            continue
        # Suppress terms whose underlying signal is not actually present.
        if name == "dark_gap" and row.gap_over_origin <= 0:
            continue
        if name == "size" and not np.isfinite(row.length):
            continue
        try:
            out.append(_EVIDENCE_TEXT[name].format(**ctx))
        except (KeyError, ValueError):
            continue
    if not out:
        out.append("Present in the area, but no distinguishing evidence")
    return out


@dataclass
class BaselineScorer:
    """Transparent, label-free weighted log-odds scorer."""

    weights: dict = None
    bias: float = BASELINE_BIAS
    name: str = "baseline"

    def __post_init__(self):
        self.weights = dict(BASELINE_WEIGHTS if self.weights is None else self.weights)

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        if feats.empty:
            return feats.assign(score=[], prob=[], attribution_pct=[], evidence=[])
        terms = _transform(feats)
        logit = np.full(len(feats), float(self.bias))
        for k, w in self.weights.items():
            logit = logit + w * terms[k].to_numpy(dtype=float)

        out = feats.copy()
        out["score"] = logit
        out["prob"] = 1.0 / (1.0 + np.exp(-logit))

        ev = []
        for i, (_, row) in enumerate(feats.iterrows()):
            contribs = {k: float(self.weights[k] * terms[k].iat[i]) for k in self.weights}
            ev.append(_evidence_for(row, terms, contribs))
        out["evidence"] = ev
        return _finalise(out)


# ---------------------------------------------------------------------------
# Learned ranker: LightGBM LambdaMART
# ---------------------------------------------------------------------------

class LearnedScorer:
    """LambdaMART ranker over the same 14 features.

    Trained per-scenario (one query group = one spill), which is the right
    formulation: we never need a calibrated probability that vessel X is guilty in
    the abstract, only a correct *ordering* of the vessels present at one spill.
    """

    name = "learned"

    def __init__(self, model=None, model_path: str | None = None):
        if model is None and model_path is not None:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
        self.model = model

    @classmethod
    def train(
        cls,
        feats: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        seed: int = 0,
    ) -> "LearnedScorer":
        """Fit on stacked scenarios. `groups` is the scenario id per row."""
        import lightgbm as lgb

        order = np.argsort(groups, kind="stable")
        X = feats.iloc[order][FEATURE_NAMES]
        y = np.asarray(labels)[order]
        g = np.asarray(groups)[order]
        _, counts = np.unique(g, return_counts=True)

        ds = lgb.Dataset(X, label=y, group=counts, free_raw_data=False)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            # Candidate pools are small and scenarios are few; without these the
            # trees memorise individual scenarios instead of learning the signal.
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 15,
            "seed": seed,
            "verbosity": -1,
        }
        booster = lgb.train(params, ds, num_boost_round=n_estimators)
        return cls(model=booster)

    def save(self, path: str) -> None:
        self.model.save_model(str(path))

    def score(self, feats: pd.DataFrame) -> pd.DataFrame:
        if feats.empty:
            return feats.assign(score=[], prob=[], attribution_pct=[], evidence=[])
        raw = self.model.predict(feats[FEATURE_NAMES])
        out = feats.copy()
        out["score"] = raw
        out["prob"] = 1.0 / (1.0 + np.exp(-raw))
        # Evidence stays baseline-derived on purpose: the ranker orders suspects,
        # but the justification shown to an analyst must remain human-auditable.
        base = BaselineScorer()
        terms = _transform(feats)
        ev = []
        for i, (_, row) in enumerate(feats.iterrows()):
            contribs = {k: float(base.weights[k] * terms[k].iat[i]) for k in base.weights}
            ev.append(_evidence_for(row, terms, contribs))
        out["evidence"] = ev
        return _finalise(out)


def _finalise(out: pd.DataFrame) -> pd.DataFrame:
    """Rank, and normalise scores into a share-of-blame percentage."""
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    # Softmax over scores rather than normalised probabilities: attribution is a
    # statement about *relative* suspicion among the vessels that were present.
    s = out.score.to_numpy(dtype=float)
    z = np.exp(s - s.max())
    out["attribution_pct"] = 100.0 * z / z.sum() if z.sum() > 0 else 0.0
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def get_scorer(kind: str = "baseline", model_path: str | None = None):
    """Factory: ``"baseline"`` or ``"learned"`` (needs `model_path`)."""
    if kind == "baseline":
        return BaselineScorer()
    if kind == "learned":
        if not model_path:
            raise ValueError("learned scorer requires model_path")
        return LearnedScorer(model_path=model_path)
    raise ValueError(f"unknown scorer {kind!r}")
