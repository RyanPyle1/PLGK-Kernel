# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Population distances between clean / adversarial / random audit groups.

Produces the paper's population-similarity table: the first-order (MMD) and
second-order (Bures) terms of the kernel Bures-Wasserstein distance between
each pair of audit populations, with base-example permutation p-values.

The claim under test is an *asymmetry* -- that the structure discriminating
clean from adversarial populations is predominantly second-order -- so the
two terms are reported separately rather than combined.

Consumes a Task-A adversarial audit artifact from
``mnist_adversarial_audit.py`` (which contains ``PNTK``). Each audit point n
is represented by its attribution signature ``PNTK[:, n]``: how the training
set explains n.

Units of inference
------------------
``build_adversarial_audit_sets`` lays out 16 slots per source example
(4 attacks x [clean, adv, rand1, rand2]) and writes the SAME clean image into
all four attack slots. The clean group is therefore duplicated 4x, and all
three groups are nested within the same source examples. This script

  (i)  deduplicates clean to one signature per source example, and
  (ii) permutes whole source examples rather than individual points.

Point-level permutation shuffles exact duplicates and same-image siblings
across groups, violating exchangeability and inflating significance; both
p-values are reported so the inflation is visible.

Sample size
-----------
The paper's default ``--n-successful 16`` yields only 16 independent units
per group, which is underpowered for this test. The reported results use
``--n-successful 64``. A reproduction at the default may fail to detect the
separation; that is a power limitation, not a failure to replicate.

Usage
-----
    # one artifact
    python scripts/mnist_population_distances.py \\
        --artifact data/mnist_adv_task_a.pt --out figures/population_seed4.json

    # aggregate several seeds into the paper's table
    python scripts/mnist_population_distances.py \\
        --aggregate figures/population_seed*.json \\
        --out-md figures/table_population_distances.md
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
    adversarial_group_masks, bures_distance, mmd, permutation_test,
    project_pooled,
)

PAIRS = [("clean", "adv"), ("clean", "rand"), ("adv", "rand")]


def slot_base(n_audit: int) -> np.ndarray:
    """Return the source-example index of each audit slot."""
    if n_audit % 16 != 0:
        raise ValueError(
            f"n_audit={n_audit} is not a multiple of 16; this does not look "
            "like an audit set from build_adversarial_audit_sets()."
        )
    return np.repeat(np.arange(n_audit // 16), 16)


def block_permutation_test(A, B, base_A, base_B, stat_fn, n_perm=1000, seed=0):
    """Permutation test that relabels whole source examples.

    Each (source example, origin group) pair is one indivisible block, so
    points sharing a source image never split across the two pseudo-groups.
    """
    rng = np.random.default_rng(seed)
    observed = float(stat_fn(A, B))
    pooled = np.vstack([A, B])
    pooled_base = np.concatenate([base_A, base_B])
    origin = np.concatenate([np.zeros(len(A), int), np.ones(len(B), int)])
    blocks = sorted(set(zip(pooled_base.tolist(), origin.tolist())))
    block_idx = {b: np.where((pooled_base == b[0]) & (origin == b[1]))[0]
                 for b in blocks}
    n_a_blocks = sum(1 for b in blocks if b[1] == 0)

    count = 0
    for _ in range(n_perm):
        order = rng.permutation(len(blocks))
        ia = np.concatenate([block_idx[blocks[j]] for j in order[:n_a_blocks]])
        ib = np.concatenate([block_idx[blocks[j]] for j in order[n_a_blocks:]])
        if len(ia) < 2 or len(ib) < 2:
            continue
        if stat_fn(pooled[ia], pooled[ib]) >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def analyse(artifact: str, shrinkage: float, n_perm: int, seed: int,
            normalize: bool) -> dict:
    art = torch.load(artifact, weights_only=False)
    if art.get("PNTK") is None:
        raise SystemExit(f"{artifact} has no PNTK; use a Task-A artifact.")
    PNTK = art["PNTK"].numpy().astype(np.float64)
    N = PNTK.shape[1]
    sig = PNTK.T
    if normalize:
        sig = sig / np.clip(np.linalg.norm(sig, axis=1, keepdims=True), 1e-30, None)

    masks = adversarial_group_masks(N)
    base = slot_base(N)

    # Deduplicate clean: one signature per source example.
    clean_all = np.where(masks["clean"])[0]
    clean_first = np.array([clean_all[base[clean_all] == i][0]
                            for i in sorted(set(base[clean_all]))])
    idx = {"clean": clean_first,
           "adv": np.where(masks["adv"])[0],
           "rand": np.where(masks["rand"])[0]}

    # Confirm the duplication the dedup assumes, rather than trusting it.
    dup_spread = 0.0
    for i in sorted(set(base[clean_all])):
        blk = sig[clean_all[base[clean_all] == i]]
        dup_spread = max(dup_spread, float(np.abs(blk - blk[0]).max()))

    print(f"artifact: {artifact}   normalized={normalize}")
    print(f"  clean {len(idx['clean']):4d} (deduped from {len(clean_all)})  "
          f"adv {len(idx['adv']):4d}  rand {len(idx['rand']):4d}   "
          f"n_base={len(set(base.tolist()))}")
    print(f"  max within-example spread of duplicated clean signatures: "
          f"{dup_spread:.2e}")

    mean_fn = lambda X, Y: mmd(X, Y, unbiased=True)
    cov_fn = lambda X, Y: bures_distance(X, Y, shrinkage=shrinkage).cov_term

    out = {"artifact": artifact, "normalized": bool(normalize),
           "shrinkage": shrinkage, "n_perm": n_perm,
           "clean_dedup_max_spread": dup_spread, "pairs": {}}

    print("\n  pair            term   distance      p_point   p_block")
    for a, b in PAIRS:
        A, B = sig[idx[a]], sig[idx[b]]
        ba, bb = base[idx[a]], base[idx[b]]
        Ap, Bp = project_pooled(A, B)
        rec = {}
        for term, fn in (("mean", mean_fn), ("cov", cov_fn)):
            d, p_pt = permutation_test(Ap, Bp, fn, n_perm=n_perm, seed=seed)
            _, p_bl = block_permutation_test(Ap, Bp, ba, bb, fn,
                                             n_perm=n_perm, seed=seed)
            rec[term] = {"distance": d, "p_point": p_pt, "p_block": p_bl}
            print(f"  {a}_vs_{b:5s}  {term:5s}  {d:.6e}   "
                  f"{p_pt:.4f}    {p_bl:.4f}")
        out["pairs"][f"{a}_vs_{b}"] = rec
    return out


def aggregate(paths: list[str], out_md: str | None) -> None:
    runs = [json.load(open(p)) for p in paths]
    n = len(runs)
    print(f"\nAggregating {n} run(s)\n")
    rows = []
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        md = np.array([r["pairs"][key]["mean"]["distance"] for r in runs])
        cd = np.array([r["pairs"][key]["cov"]["distance"] for r in runs])
        cp = np.array([r["pairs"][key]["cov"]["p_block"] for r in runs])
        mp = np.array([r["pairs"][key]["mean"]["p_block"] for r in runs])
        sd = lambda v: float(v.std(ddof=1)) if n > 1 else 0.0
        rows.append({
            "pair": key,
            "mean": float(md.mean()), "mean_sd": sd(md),
            "cov": float(cd.mean()), "cov_sd": sd(cd),
            "cov_p_min": float(cp.min()), "cov_p_max": float(cp.max()),
            "mean_p_min": float(mp.min()), "mean_p_max": float(mp.max()),
            "n_cov_sig": int((cp < 0.05).sum()),
            "n_mean_sig": int((mp < 0.05).sum()),
        })
        print(f"  {key:16s} mean {md.mean():8.4f} +/- {sd(md):.4f} "
              f"(sig {int((mp<0.05).sum())}/{n})   "
              f"cov {cd.mean():8.4f} +/- {sd(cd):.4f} "
              f"(sig {int((cp<0.05).sum())}/{n})")

    ca = rows[0]
    ratio = ca["cov"] / ca["mean"] if ca["mean"] > 0 else float("inf")
    print(f"\n  clean-vs-adv second-to-first order ratio: {ratio:.2f}x")
    print(f"  covariance significant in {ca['n_cov_sig']}/{n} runs; "
          f"mean term in {ca['n_mean_sig']}/{n}")

    if out_md:
        floor = 1.0 / (runs[0]["n_perm"] + 1)
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w") as f:
            f.write("# Population distances in attribution space\n\n")
            f.write(f"Mean +/- sd over {n} independent training runs. "
                    f"p ranges are min-max of the base-example permutation "
                    f"p-value; the floor is 1/{runs[0]['n_perm']+1} "
                    f"= {floor:.4f}.\n\n")
            f.write("| pair | mean term (MMD, not MMD^2) | cov term (Bures) "
                    "| p (cov) |\n|---|---|---|---|\n")
            for r in rows:
                pl = ("< 0.001" if r["cov_p_max"] <= floor * 1.01
                      else f"{r['cov_p_min']:.3f}-{r['cov_p_max']:.3f}")
                f.write(f"| {r['pair'].replace('_vs_', ' vs ')} | "
                        f"{r['mean']:.3f} +/- {r['mean_sd']:.3f} | "
                        f"{r['cov']:.3f} +/- {r['cov_sd']:.3f} | {pl} |\n")
        print(f"\nWrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=str, default=None,
                    help="Task-A artifact from mnist_adversarial_audit.py")
    ap.add_argument("--aggregate", type=str, nargs="+", default=None,
                    help="Per-run JSONs to combine into the paper's table")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-md", type=str,
                    default="figures/table_population_distances.md")
    ap.add_argument("--shrinkage", type=float, default=0.1,
                    help="Covariance shrinkage toward a scaled identity")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", action="store_true",
                    help="Row-normalize signatures (robustness check: removes "
                         "the gross magnitude difference between groups)")
    args = ap.parse_args()

    if args.aggregate:
        paths = [p for pat in args.aggregate for p in sorted(glob.glob(pat))] \
            or args.aggregate
        aggregate(paths, args.out_md)
        return

    if not args.artifact:
        ap.error("pass --artifact to analyse one run, or --aggregate to combine")

    out = analyse(args.artifact, args.shrinkage, args.n_perm, args.seed,
                  args.normalize)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
