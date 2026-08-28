# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""When does the population separation appear? Per-epoch mode-space analysis.

Companion to ``mnist_population_distances.py``. That script measures the
separation in a trained model; this one asks *when during training* it
appears, working in the mode-coefficient basis of Section 6.3.1 rather than
in attribution space.

The result is a prediction rather than a description. Section 6.3 argues that
cancellation becomes effective when representations stabilize, and Figure 14
places the kernel-regime transition between epochs 5 and 10. If the
population separation is a consequence of that structure it should become
detectable in the same window and not before.

Consumes the SVD-snapshot artifact from ``mnist_mode_svd_snapshots.py``,
which must contain ``C_all_per_epoch`` -- the slot-ordered (N_audit, K)
coefficient matrix per epoch. The per-group split stored under
``C_signed_*_per_epoch`` is NOT sufficient: it discards slot identity, which
is what defines the permutation blocks.

Deduplication and base-example permutation follow
``mnist_population_distances.py``; see that script for why point-level
permutation is invalid here.

Reading the output
------------------
Two quantities matter and they answer different questions. The p-values say
whether the separation is detectable at each epoch. The *distances* say
whether an onset is real: a block permutation loses power relative to a
point-level one, so early-epoch nulls could in principle reflect the test
rather than the geometry. If the covariance distance also grows by orders of
magnitude across training while the control stays flat, the onset is a
property of the model rather than of the test.

Usage
-----
    python scripts/mnist_population_temporal.py \\
        --artifact data/mnist_expc_svd_full.pt \\
        --out figures/temporal_seed4.json

    python scripts/mnist_population_temporal.py \\
        --aggregate figures/temporal_seed*.json \\
        --out-md figures/table_population_temporal.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kernel_tools import (
    adversarial_group_masks, bures_distance, mmd, project_pooled,
)
from mnist_population_distances import block_permutation_test, slot_base


def analyse(artifact: str, shrinkage: float, n_perm: int,
            n_perm_checkpoint: int, checkpoints: list[int], seed: int) -> dict:
    art = torch.load(artifact, weights_only=False)
    if "C_all_per_epoch" not in art:
        raise SystemExit(
            f"{artifact} lacks 'C_all_per_epoch'. Re-run "
            "mnist_mode_svd_snapshots.py: the per-group split alone discards "
            "the slot identity that the base-example permutation requires."
        )

    C_all = art["C_all_per_epoch"]
    epochs = art.get("epochs", list(range(len(C_all))))
    N = int(C_all[0].shape[0])
    masks = adversarial_group_masks(N)
    base = slot_base(N)

    clean_all = np.where(masks["clean"])[0]
    clean_first = np.array([clean_all[base[clean_all] == i][0]
                            for i in sorted(set(base[clean_all]))])
    idx = {"clean": clean_first,
           "adv": np.where(masks["adv"])[0],
           "rand": np.where(masks["rand"])[0]}

    print(f"artifact: {artifact}")
    print(f"  K={art.get('K')}  epochs={len(C_all)}  N_audit={N}  "
          f"n_base={len(set(base.tolist()))}")
    print(f"  groups after dedup: clean={len(idx['clean'])} "
          f"adv={len(idx['adv'])} rand={len(idx['rand'])}")

    cov_fn = lambda X, Y: bures_distance(X, Y, shrinkage=shrinkage).cov_term
    mean_fn = lambda X, Y: mmd(X, Y, unbiased=True)

    out = {"artifact": artifact, "K": int(art.get("K", 0)),
           "epochs": [int(e) for e in epochs], "n_perm": n_perm,
           "n_perm_checkpoint": n_perm_checkpoint, "per_epoch": []}

    print("\n epoch |  clean-vs-adv cov     p_block | ctrl cov    p_block | nperm")
    for t, Cm in enumerate(C_all):
        ep = int(epochs[t]) if t < len(epochs) else t
        X = np.asarray(Cm, dtype=np.float64)
        npm = n_perm_checkpoint if ep in checkpoints else n_perm
        row = {"epoch": ep, "n_perm": npm}
        for a, b in [("clean", "adv"), ("clean", "rand")]:
            A, B = X[idx[a]], X[idx[b]]
            ba, bb = base[idx[a]], base[idx[b]]
            Ap, Bp = project_pooled(A, B)
            cd, cp = block_permutation_test(Ap, Bp, ba, bb, cov_fn,
                                            n_perm=npm, seed=seed)
            md, mp = block_permutation_test(Ap, Bp, ba, bb, mean_fn,
                                            n_perm=npm, seed=seed)
            row[f"{a}_vs_{b}"] = {"cov_distance": cd, "cov_p_block": cp,
                                  "mean_distance": md, "mean_p_block": mp}
        ca, cr = row["clean_vs_adv"], row["clean_vs_rand"]
        print(f"  {ep:4d} |  {ca['cov_distance']:.4e}    {ca['cov_p_block']:.4f} "
              f"| {cr['cov_distance']:.3e}  {cr['cov_p_block']:.4f} | {npm}",
              flush=True)
        out["per_epoch"].append(row)

    out["summary"] = _summarize(out["per_epoch"])
    s = out["summary"]
    print(f"\n  clean-vs-adv significant: {s['n_sig_clean_vs_adv']}/{s['n_epochs']} epochs")
    print(f"  control significant:      {s['n_sig_control']}/{s['n_epochs']} epochs")
    print(f"  sustained onset epoch:    {s['sustained_onset_epoch']}")
    print(f"  significant in epochs 0-4: {s['early_epochs_sig'] or 'none'}")
    print(f"  cov distance growth ep0->last: {s['cov_growth']:.0f}x "
          f"(control {s['control_growth']:.1f}x)")
    return out


def _summarize(per_epoch: list[dict], alpha: float = 0.05) -> dict:
    eps = [r["epoch"] for r in per_epoch]
    ca = [r["clean_vs_adv"]["cov_p_block"] for r in per_epoch]
    cr = [r["clean_vs_rand"]["cov_p_block"] for r in per_epoch]
    sustained = None
    for i, (e, p) in enumerate(zip(eps, ca)):
        if p < alpha and all(q < alpha for q in ca[i:]):
            sustained = e
            break
    d0 = per_epoch[0]["clean_vs_adv"]["cov_distance"]
    dT = per_epoch[-1]["clean_vs_adv"]["cov_distance"]
    r0 = per_epoch[0]["clean_vs_rand"]["cov_distance"]
    rT = per_epoch[-1]["clean_vs_rand"]["cov_distance"]
    return {
        "n_epochs": len(eps),
        "n_sig_clean_vs_adv": sum(1 for p in ca if p < alpha),
        "n_sig_control": sum(1 for p in cr if p < alpha),
        "first_sig_epoch": next((e for e, p in zip(eps, ca) if p < alpha), None),
        "sustained_onset_epoch": sustained,
        "early_epochs_sig": [e for e, p in zip(eps, ca) if p < alpha and e <= 4],
        "cov_growth": float(dT / d0) if d0 > 0 else float("inf"),
        "control_growth": float(rT / r0) if r0 > 0 else float("inf"),
    }


def aggregate(paths: list[str], out_md: str | None) -> None:
    runs = [json.load(open(p)) for p in paths]
    n = len(runs)
    sums = [r["summary"] for r in runs]
    onset = np.array([s["sustained_onset_epoch"] for s in sums], dtype=float)
    nsig = np.array([s["n_sig_clean_vs_adv"] for s in sums])
    growth = np.array([s["cov_growth"] for s in sums])
    cgrowth = np.array([s["control_growth"] for s in sums])
    sd = lambda v: float(v.std(ddof=1)) if n > 1 else 0.0

    print(f"\nAggregating {n} run(s)\n")
    print("  run | onset | epochs sig | control sig | sig in 0-4")
    for p, s in zip(paths, sums):
        print(f"  {Path(p).stem:12s} | {str(s['sustained_onset_epoch']):>5s} | "
              f"{s['n_sig_clean_vs_adv']:>4d}/{s['n_epochs']:<4d} | "
              f"{s['n_sig_control']:>4d}/{s['n_epochs']:<4d} | "
              f"{s['early_epochs_sig'] or 'none'}")
    print(f"\n  onset: {onset.mean():.1f} +/- {sd(onset):.1f} "
          f"(range {int(onset.min())}-{int(onset.max())})")
    print(f"  epochs significant: {nsig.mean():.1f} +/- {sd(nsig):.1f}")
    print(f"  control significant in any epoch of any run: "
          f"{sum(s['n_sig_control'] for s in sums)}/"
          f"{sum(s['n_epochs'] for s in sums)} epoch-runs")
    print(f"\n  covariance distance growth ep0->last: {growth.mean():.0f}x "
          f"+/- {sd(growth):.0f}")
    print(f"  control growth:                       {cgrowth.mean():.1f}x "
          f"+/- {sd(cgrowth):.1f}")
    print("\n  The growth contrast is what distinguishes a real onset from a")
    print("  power artifact: a test-power explanation would leave the")
    print("  distances flat and move only the p-values.")

    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w") as f:
            f.write("# Onset of population separation during training\n\n")
            f.write(f"Mode-coefficient basis, {n} independent training runs, "
                    "base-example permutation.\n\n")
            f.write("| run | sustained onset | epochs significant | "
                    "control significant | significant in epochs 0-4 |\n")
            f.write("|---|---|---|---|---|\n")
            for p, s in zip(paths, sums):
                f.write(f"| {Path(p).stem} | epoch {s['sustained_onset_epoch']} "
                        f"| {s['n_sig_clean_vs_adv']}/{s['n_epochs']} "
                        f"| {s['n_sig_control']}/{s['n_epochs']} "
                        f"| {'none' if not s['early_epochs_sig'] else s['early_epochs_sig']} |\n")
            f.write(f"\nOnset {onset.mean():.1f} +/- {sd(onset):.1f}. "
                    f"Covariance distance grows {growth.mean():.0f}x "
                    f"+/- {sd(growth):.0f} from the first to the last epoch "
                    f"while the control grows {cgrowth.mean():.1f}x "
                    f"+/- {sd(cgrowth):.1f}.\n")
        print(f"\nWrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=str, default=None,
                    help="SVD-snapshot artifact from mnist_mode_svd_snapshots.py")
    ap.add_argument("--aggregate", type=str, nargs="+", default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-md", type=str,
                    default="figures/table_population_temporal.md")
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=200,
                    help="Permutations at non-checkpoint epochs")
    ap.add_argument("--n-perm-checkpoint", type=int, default=1000,
                    help="Permutations at checkpoint epochs")
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[0, 4, 9, 19, 29, 39, 49])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.aggregate:
        paths = [p for pat in args.aggregate for p in sorted(glob.glob(pat))] \
            or args.aggregate
        aggregate(paths, args.out_md)
        return

    if not args.artifact:
        ap.error("pass --artifact to analyse one run, or --aggregate to combine")

    out = analyse(args.artifact, args.shrinkage, args.n_perm,
                  args.n_perm_checkpoint, args.checkpoints, args.seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
