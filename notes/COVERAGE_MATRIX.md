# PLGK Paper — Empirical Coverage Matrix

Maps every empirical claim in the paper (each figure, each table, each quoted
metric) to the script in this repository that produces it.

The README groups scripts by execution order and shared dependencies; this
matrix is organised by paper section instead, so a reader starting from a
figure number can find its producer directly.

One item, Figure 38, has no producing script here. It is listed as such, and
`scripts/svhn_fig38_kernel_regime.py` documents the full metric specification.

## Section 5 — Auditing (main text)

Section 5 is theory + definitions (MRE, LC, trapezoidal correction). No standalone experimental claims in §5.1–5.4; all §5 numerical claims appear in §6 or the appendix. **No matrix rows for §5 itself.**

---

## Section 6 — Experiments (main text)

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 1 | §6.1 Fig 1 | MNIST/LeNet5 audit reconstruction MRE + LC over training time | `scripts/mnist_train_and_audit.py` + `scripts/mnist_fig01_reconstruction.py` | Quoted numbers: main text says "high fidelity"; Table A.5 says MRE=0.010, Corr=0.999998 |
| 2 | §6.1 Fig 2 | MNIST single-mistake audit (input 4 mis-ID'd as 6, sorted per-training-sample influences) | `scripts/mnist_fig02_04_worst_error.py` | Reuses the Group 1 audit artifact |
| 3 | §6.1 Fig 3 | Top 25 helpful + top 25 harmful training points (same mistake) | `scripts/mnist_fig02_04_worst_error.py` | Same |
| 4 | §6.1 Fig 4 | Top 25 helpful/harmful, restricted to class 6 | `scripts/mnist_fig02_04_worst_error.py` | Same |
| 5 | §6.1 (Appendix A.11 reference) | Comparison vs TracIn (referenced in §6.1) | `scripts/mnist_table4_trak_comparison.py` | Backing table = A.11 Fig 26 |
| 6 | §6.2 Fig 5 | 2D UMAP projection of IPS, colored by true class + misclassified overlay | `scripts/mnist_fig05_06_ips_umap.py` | |
| 7 | §6.2 Fig 6 | UMAP with similarity vs target point (whole + zoomed to same class) | `scripts/mnist_fig05_06_ips_umap.py` | |
| 8 | §6.2 Fig 7 | Class 7 examples at extreme UMAP dim1 / dim2 values | `scripts/mnist_fig07_class7_umap_extremes.py` | |
| 9 | §6.2 Table 1 | IPS-based supervised difficulty predictors: AUROC/AUPR/Spearman | `scripts/mnist_table1_ips_difficulty.py` | Compares IPS ℓ2 / IPS class margin / hidden top-k / MSP / logit margin |
| 10 | §6.3 Fig 8 | Clean/adv/random image triplets | `scripts/mnist_fig08_adv_triplets.py` | Uses foolbox 4 attacks: FGSM, PGD-L∞, BIM-L∞, DeepFool-L∞ |
| 11 | §6.3 MRE quote | "loss audit reaching final MRE 0.01, corr 0.999997; prediction audit MRE 0.019, corr 0.9999999" | `scripts/mnist_adversarial_audit.py` | |
| 12 | §6.3 Fig 9 | Clean vs adversarial per-example influence scatter (corr > 0.6) | `scripts/mnist_fig09_clean_vs_adv_scatter.py` | |
| 13 | §6.3 Fig 10 | Per-layer influence over time (sum + abs sum, adv/clean/random) | `scripts/mnist_fig10_per_layer_temporal.py` | Caption finding: adversarial influence concentrates in the first two FC layers from epoch 10 onward |
| 14 | §6.3 Fig 11 | Perturbation-ray temporal contributions + super-linear total influence | `scripts/mnist_fig11_perturbation_ray.py` | "Task B" = the two interpolation rays |
| 15 | §6.3 Fig 12 | Per-epoch mean influence magnitude + per-epoch mean loss (clean/adv/random) | `scripts/mnist_fig12_per_epoch_influence.py` | Task A = paired clean/adv/random |
| 16 | §6.3 Fig 13 | Cumulative Sn + Gn (net + gross influence): cancellation evidence | `scripts/mnist_fig13_cancellation.py` | Same Task A pipeline |
| 17 | §6.3 Fig 14 | Kernel-regime metrics (ρfeat, ρact, subspace overlap) vs gross influence + loss | `scripts/mnist_fig14_kernel_regime.py` | |
| 18 | §6.3.1 Fig 15 | Mean Δc_k between adv/clean, scattered vs σk + signed train aggregation (top-9 modes) | `scripts/mnist_fig15_16_mode_decomposition.py` | Consumes the Group 3 mode-SVD artifact |
| 19 | §6.3.1 Fig 16 | Per-mode input sensitivity ∥∇xck∥ + loss-corrected, top-50 modes, three example types | `scripts/mnist_fig15_16_mode_decomposition.py` | Same script |
| 20 | §6.3.1 Fig 17 | Per-mode input sensitivity per-epoch (top-10 vs next 40) | `scripts/mnist_fig17_input_sensitivity_temporal.py` | Expensive to regenerate; needs per-epoch checkpoints (see script docstring) |
| 21 | §6.3.1 numerical claims | "PR=2.4, top-5 modes explain 84% of ΔL", ρ correlations 0.72/0.80, "~80× lower magnitude for random" | `scripts/mnist_fig15_16_mode_decomposition.py` | Quoted numbers come from the same mode-decomposition pipeline |
| 22 | §6.3.1 Fig 18 | Cancellation ratio CR(K, t) over time + across modes at fixed epoch | `scripts/mnist_fig18_cancellation_ratio.py` | |
| 23 | §6.3 Fig 19 | Grouped example losses over training (clean/adv/random/cured-adv/cured-random) | `scripts/mnist_fig19_20_table2_adv_cure.py` | |
| 24 | §6.3 Table 2 | Final test accuracy per data type: Clean 100%, Adv 0%, Cured Adv 87.5%, Cured Random 100%, Adv+Random 15.6% | `scripts/mnist_fig19_20_table2_adv_cure.py` | T2 (label-free) is the path the paper uses; T1 is a control |
| 25 | §6.3 Fig 20 | Visualization of clean/adv/cured perturbations for a single example | `scripts/mnist_fig19_20_table2_adv_cure.py` | |
| 26 | §6.3.1 quote | "correction is not just adv reversal: cosine ≈ −0.3" | `scripts/mnist_fig19_20_table2_adv_cure.py` | |
| 27 | §6.3.1 Table 3 | Energy concentration K@90%/95%/99%, standard vs PGD-AT | *not included* — PGD-AT rank comparison | Explored during revision; not part of the published results |

---

## Section 7 — PLGK Scaling

Section 7 body is narrative; it references A.18 numbers ("mean 0.94 IPS-audit correlation", "IPS scales to ImageNet") which come from Appendix A.18 (below). No standalone §7 rows.

---

## Appendix A — Details, extensions, replications

### A.4 MNIST Data Pruning

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 28 | Fig 21 | Final test acc after pruning (least-influential, IPS clusters, class-balanced, random) | `scripts/mnist_fig21_22_pruning.py` | Consumes the Group 1 artifact WITH IPS |
| 29 | Fig 22 | Final test acc after pruning MOST-influential (inverse of Fig 21) | `scripts/mnist_fig21_22_pruning.py` | Inverse of Fig 21; same script, destructive variants |

### A.5 MNIST Auditing Profiling

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 30 | A.5 Table | Baseline/Audit/NonTrap/Supersample/Subsample × time / memory / MLE / Corr on MNIST | `scripts/mnist_audit_profiling.py` + `scripts/mnist_table_a5_profiling.py` | Six variants: baseline, audit, nontrap, supersample_2, subsample_2, subsample_5 |

### A.7 T×(MP) Kernel

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 31 | Fig 23 | Eigenmode loading over time (Mode 1 tracks gradient magnitude; modes 2-5 track adaptive→kernel) | `scripts/mnist_fig14_kernel_regime.py` | |
| 32 | Fig 24 | Participation Ratio (PR) over time | Same as Fig 23 | |

### A.8 Online Feature Modes (Oja SVD)

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 33 | Fig 25 | Energy captured by Oja projector over training time (>99% by epoch 10) | `scripts/mnist_fig18_cancellation_ratio.py` | The online-SVD Oja rule feeds into the D-1 cell |

### A.11 TracIn Comparison

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 34 | Fig 26 | TracIn reconstruction MRE + Corr over training (compared to audit) | `scripts/mnist_table4_trak_comparison.py` | |

### A.12 TRAK Comparison

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 35 | Table 4 | Audit (trapezoidal) MLE=0.0104, Corr=0.999998, Time=4040s vs TRAK 1ckpt / 5ckpt | `scripts/mnist_table4_trak_comparison.py` | Requires `pip install traker` |

### A.15 SVD Modes with Random Perturbation

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 36 | Fig 27 | Random vs clean mode activity (companion to Fig 15) | `scripts/mnist_fig15_16_mode_decomposition.py` | The Adv vs Random ratio figure at line ~573 of paper_at_3source.py |

### A.16, A.17 SVHN Replication

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 37 | A.17 Table | Baseline/Audit/Supersample/Subsample × time / memory / MLE / Corr on SVHN | `scripts/svhn_audit_profiling.py` + `scripts/svhn_table_a17_profiling.py` | Four variants: baseline, audit, supersample_2, subsample_2 |
| 38 | A.17.1 Fig 28 | SVHN audit reconstruction over training (MRE + Corr) | `scripts/svhn_fig28_reconstruction.py` | The SVHN model is BatchNorm-free; BN is not auditable under this accumulator |
| 39 | A.17.1 Fig 29 | SVHN single-mistake audit (9 mis-ID'd as 6) | `scripts/svhn_fig29_31_worst_error.py` | |
| 40 | A.17.1 Fig 30, 31 | Helpful/harmful 9s + 6s (SVHN) | `scripts/svhn_fig29_31_worst_error.py` | |
| 41 | A.17.2 Fig 32, 33, 34 | SVHN UMAP + similarity target + class 7 dim1/dim2 examples | `scripts/svhn_fig32_34_ips_umap.py` | |
| 42 | A.17.3 Fig 35, 36 | SVHN adversarial: image triplets + Clean vs Adv influence scatter | `scripts/svhn_fig35_adv_triplets.py` + `scripts/svhn_fig36_clean_vs_adv_scatter.py` | Uses foolbox (FGSM, PGD-Linf, DeepFool, CW) |
| 43 | A.17.3 Figs 37, 38, 39, 40 | SVHN per-epoch mag/loss, kernel-regime metrics, mode activity adv vs clean, random vs clean | `scripts/svhn_fig37_per_epoch_influence.py`, `scripts/svhn_fig39_mode_activity.py`; **Fig 38 not included** (see `scripts/svhn_fig38_kernel_regime.py`) | |

### A.18 Scaling Experiments (revision-round headliners)

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 44 | A.18.1 Table 5 row 1 | CIFAR / ResNet-18-GN / 11.2M params / Audit 14.9→126s (8.5×) / 6.0 GiB / 56.6% acc | `scripts/scaling/run_cifar10_audit.py` | Uses `models/resnet_gn.py` |
| 45 | A.18.1 Table 5 row 3 | ImageNet / ResNet-18-GN / 11.7M / IPS-only 2489→6772s (2.7×) / 22.4 GiB / 41.3% top-1 | `scripts/scaling/run_ips_only.py` | Uses `imagenet_prep.py` for dataloading |
| 46 | A.18.1 Table 6 | CIFAR-10 ResNet-18-GN IPS-Audit per-sample correlation over 20/50/80/120 epochs (0.941→0.945) | `scripts/scaling/run_cifar10_audit.py` + `scripts/scaling/compare_ips_vs_audit.py` | May be composed from paired audit + IPS runs; `compare_ips_vs_audit.py` produces the correlation table |
| 47 | A.18.2 Table 5 row 2 | CIFAR / ViT-Tiny/4 / 2.70M / Audit 15.8→178s (11.3×) / 4.62 GiB / 46.8% acc | `scripts/scaling/run_cifar10_audit.py` (`--model vittiny4`) | Same run script; model swap via env var/CLI |
| 48 | A.18.2 quote | "at DeiT-standard lr=5e-4 corr was 0.28; at chosen lr=1.5e-6 landed at 0.68 at ep 90" | `scripts/scaling/run_cifar10_audit.py` (`--model vittiny4`) | Same driver as Table 5 row 2, at the two learning rates quoted |

### Referenced but not shown in the paper text (framework docs)

| # | Paper location | What it shows | Producing script | Notes |
|---|---|---|---|---|
| 49 | Algorithm 1 | Compute top-K modes + drives from final-time train features | `kernel_tools/svd_modes.py` (`lowrank_svd_train_side`) | No standalone plot |
| 50 | Algorithm 2 | Mode-aware cure via PGD | `kernel_tools/adversarial_cure.py` + `scripts/mnist_fig19_20_table2_adv_cure.py` | |

---

## Population-similarity appendix (added 2026-08-22)

| # | Paper location | What it shows | Producing script |
|---|---|---|---|
| P1 | Appendix, Table 1 | MMD (first-order) vs Bures (second-order) distance between clean / adversarial / eps-matched-random audit populations; 5 seeds, base-example permutation | `scripts/mnist_population_distances.py` |
| P2 | Appendix, Table 2 | Onset epoch of the clean/adversarial covariance separation in mode space; 3 seeds | `scripts/mnist_population_temporal.py` |

Library support: `kernel_tools/population.py` (unbiased MMD, Bures split into
mean + covariance terms, pooled-centered projection, base-example permutation,
adversarial slot masks).

**Dependency note.** P2 requires `C_all_per_epoch` from
`mnist_mode_svd_snapshots.py`. That key was added when these experiments
landed; artifacts produced by an earlier version of that script contain only
the per-group `C_signed_*_per_epoch` split, which discards the slot identity
the base-example permutation needs, and P2 will error out with that
explanation rather than silently computing an invalid test.

**Sample-size note.** Both rows use `--n-successful 64`, not the repo default
of 16. The audit set nests all three groups within the same source examples
and duplicates each clean image four times, so 16 successful examples yields
only 16 independent units per group — underpowered for these tests. A
reproduction at the default may fail to detect the separation; that is a
power limitation rather than a failure to replicate.

**Not included.** A subspace/principal-angle comparison was explored and
dropped before publication; it is not part of the paper and has no row here.
