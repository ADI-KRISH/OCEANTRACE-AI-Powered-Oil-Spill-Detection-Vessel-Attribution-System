"""Full-chain synthetic evaluation for Module 3.

    python notebooks/eval_attribution.py [--n 20] [--seed 0]

Slick -> real backward drift ensemble -> attribution, over N synthetic scenarios
with known ground truth (see attribution/simulate.py). Reports Top-1, Recall@3,
MRR and median rank -- the numbers quoted in attribution/README.md's Accuracy
section, kept reproducible here rather than hand-pasted from a one-off run.

Kept as a script rather than a .ipynb so it runs in CI and diffs cleanly in git,
matching notebooks/eval_detection.py.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attribution.pipeline import attribute
from attribution.simulate import make_scenario
from drift.forcing import AnalyticForcing
from drift.hindcast import hindcast_origin


def run(n: int, seed0: int = 0, n_vessels: int = 180):
    ranks, chance_ranks = [], []
    tried = 0
    seed = seed0
    while len(ranks) < n and tried < n * 4:
        sc = make_scenario(seed=seed, n_vessels=n_vessels)
        seed += 1
        tried += 1
        if sc is None:
            continue
        age_h = sc.age_s / 3600.0
        est = hindcast_origin(
            sc.slick_lat, sc.slick_lon, sc.t_detect, age_h=age_h,
            age_range_h=(age_h / 2.5, age_h * 2.5),
            forcing=AnalyticForcing(seed=seed),
            slick_bearing_deg=sc.origin.slick_bearing_deg)
        r = attribute(sc.ais, origin=est)
        hit = r.suspects[r.suspects.mmsi == sc.culprit_mmsi]
        if hit.empty:
            continue
        ranks.append(int(hit["rank"].iloc[0]))
        chance_ranks.append((len(r.suspects) + 1) / 2.0)

    ranks = np.array(ranks)
    n_ok = len(ranks)
    top1 = float(np.mean(ranks == 1)) if n_ok else float("nan")
    top3 = float(np.mean(ranks <= 3)) if n_ok else float("nan")
    mrr = float(np.mean(1.0 / ranks)) if n_ok else float("nan")
    median = float(np.median(ranks)) if n_ok else float("nan")

    print(f"scenarios: {n_ok} usable of {tried} tried (culprit recalled by the "
         f"candidate filter in {n_ok}/{tried})")
    print(f"chance median rank (pool size + 1) / 2 averaged: "
         f"{np.mean(chance_ranks):.1f}" if chance_ranks else "n/a")
    print(f"\nTop-1        {top1:.1%}")
    print(f"Recall@3     {top3:.1%}")
    print(f"MRR          {mrr:.3f}")
    print(f"median rank  {median:.1f}")
    return {"n": n_ok, "top1": top1, "top3": top3, "mrr": mrr, "median_rank": median}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.n, a.seed)
