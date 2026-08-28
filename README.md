# Path-Integrated Loss-Gradient Kernels — Public Code

Reproduction code for the paper *Path-Integrated Loss-Gradient Kernels:
Auditing and Similarity for Trained Neural Networks* (TMLR, accepted 2026).

Accompanies the TMLR paper by Pyle, Ju, and Patel; see **License and
citation** below.

## Layout

```
plgk_paper_repo/
├── kernel_tools/         Shared library: models, audit + IPS accumulators, trainer
├── scripts/              Per-figure / per-experiment scripts (see mapping below)
│   └── scaling/          Group 6 drivers (CIFAR-10 / ImageNet, ResNet-18 + ViT)
├── data/                 Datasets + saved experiment artifacts (created on first run)
├── figures/              Regenerable outputs; gitignored
├── notes/
│   └── COVERAGE_MATRIX.md    Full paper claim → producing script mapping
├── requirements.txt
└── README.md
```

`kernel_tools/` holds every numerical routine; the scripts are thin drivers
that parse arguments, load or produce an artifact, and plot. A reader wanting
the method itself should start with `kernel_tools/audit.py` (the loss audit,
Definition 11) and `kernel_tools/ips.py` (the similarity kernel, Section 4).

## Reproducing the paper's figures / tables

The scripts are organized in 6 groups, matching the paper structure. Each
downstream script consumes an artifact `.pt` from a training script.

### Group 1 — MNIST core: audit + IPS (Sections 5, 6.1, 6.2)

```bash
# ~95 min CPU / ~15 min single GPU
python scripts/mnist_train_and_audit.py --epochs 50 --train-batches 200
#     -> data/mnist_experiment1.pt

python scripts/mnist_fig01_reconstruction.py       # Figure 1
python scripts/mnist_fig02_04_worst_error.py       # Figures 2, 3, 4
```

Section 6.2's IPS figures need the IPS matrix, which the `--do-ips` flag on
`mnist_train_and_audit.py` produces alongside the audit:

```bash
python scripts/mnist_train_and_audit.py --epochs 50 --train-batches 200 --do-ips

python scripts/mnist_fig05_06_ips_umap.py           # Figures 5, 6
python scripts/mnist_fig07_class7_umap_extremes.py  # Figure 7
python scripts/mnist_table1_ips_difficulty.py       # Table 1
```

### Group 2 — MNIST adversarial + cure (Section 6.3)

```bash
# Step 1: Task A (paired clean/adv/random audit) + Task B (interpolation rays)
#         Consumes Group 1's trained model. ~2× Group 1 wall time.
python scripts/mnist_adversarial_audit.py \
    --source-artifact data/mnist_experiment1.pt

# Step 2: plots (all consume the Task A / Task B artifacts above)
python scripts/mnist_fig08_adv_triplets.py          # Figure 8
python scripts/mnist_fig09_clean_vs_adv_scatter.py  # Figure 9
python scripts/mnist_fig10_per_layer_temporal.py    # Figure 10
python scripts/mnist_fig11_perturbation_ray.py      # Figure 11
python scripts/mnist_fig12_per_epoch_influence.py   # Figure 12
python scripts/mnist_fig13_cancellation.py          # Figure 13

# Step 3: mode-aware adversarial cure (Fig 19 + Fig 20 + Table 2)
#         DEPENDS ON Group 3's V_modes/Lambda SVD artifact — will error out
#         with clear guidance if the SVD file is missing.
python scripts/mnist_fig19_20_table2_adv_cure.py
```

**Cross-group dependency**: the cure script needs `V_modes` + `Lambda` from
Group 3's `mnist_mode_svd_snapshots.py`. Run Group 3 first, then Group 2's
cure step. Everything else in Group 2 stands alone from Group 3.

Figure 14 (kernel-regime metrics × loss/influence) is also §6.3 but the
underlying feature-function-overlap / subspace-overlap / activation-overlap
data belongs to Group 3's SVD tracking; Fig 14 script lives with Group 3.

### Group 3 — MNIST mode analysis (Section 6.3.1)

```bash
# Step 1: retrain + capture per-epoch SVD of Φ_train (also emits the SVD
#         cure artifact Group 2 depends on).
#         ~2× Group 1 wall time. Adopts Group 2's adversarial audit set.
python scripts/mnist_mode_svd_snapshots.py \
    --adv-bundle data/mnist_adv_data.pt \
    --K 128

# Step 2: plots
python scripts/mnist_fig14_kernel_regime.py                       # Figure 14
python scripts/mnist_fig15_16_mode_decomposition.py --compute-dcdx # Figures 15, 16
python scripts/mnist_fig18_cancellation_ratio.py                  # Figure 18

# Figure 17 (per-epoch temporal input sensitivity) is expensive to
# regenerate — requires per-epoch model checkpoints + O(epochs × N × K)
# double-autograd. Ships with a clear "compute deferred" error if the
# cache is missing; add per-epoch checkpointing to the snapshot driver
# and rerun with --compute if you need it.
python scripts/mnist_fig17_input_sensitivity_temporal.py \
    --cache data/mnist_dcdx_temporal_K50.npz  # errors cleanly if missing
```

Group 2's cure script (`mnist_fig19_20_table2_adv_cure.py`) consumes the SVD
cure artifact produced by Group 3, so run Group 3 first; it exits with
guidance if that artifact is missing.

### Group 3b — Population-level distances (population-similarity appendix)

Population-to-population distances in the audit geometry: the first-order
(MMD) and second-order (Bures) terms between the clean / adversarial /
eps-matched-random audit groups, plus when during training the separation
appears.

```bash
# --- Main result: MMD vs Bures across the three population pairs ---
# Needs Task-A artifacts at --n-successful 64 (the paper's default of 16
# leaves only 16 independent units per group and is underpowered here).
for s in 4 5 6 7 8; do
  python scripts/mnist_train_and_audit.py --seed $s       --out data/seed${s}_experiment1.pt
  python scripts/mnist_adversarial_audit.py --seed $s --n-successful 64       --source-artifact data/seed${s}_experiment1.pt       --out-adv-data data/seed${s}_adv_data.pt       --out-task-a data/seed${s}_adv_task_a.pt --skip-task-b
  python scripts/mnist_population_distances.py       --artifact data/seed${s}_adv_task_a.pt       --out figures/population_seed${s}.json
done
python scripts/mnist_population_distances.py     --aggregate "figures/population_seed*.json"

# --- Sub-experiment: when does the separation appear? ---
# Consumes the mode-SVD snapshots; 3 seeds suffice for the onset.
for s in 4 5 6; do
  python scripts/mnist_mode_svd_snapshots.py --seed $s --K 128       --adv-bundle data/seed${s}_adv_data.pt       --out data/seed${s}_svd_full.pt
  python scripts/mnist_population_temporal.py       --artifact data/seed${s}_svd_full.pt       --out figures/temporal_seed${s}.json
done
python scripts/mnist_population_temporal.py     --aggregate "figures/temporal_seed*.json"
```

Both scripts also accept `--normalize` (row-normalize the attribution
signatures) as a robustness check: it removes the gross magnitude difference
between groups, and the conclusion is unchanged.

**Wall time**: the audit dominates. Each `mnist_adversarial_audit.py` at
`--n-successful 64` is ~4x the default (the audit computes per-sample
gradients over all 1024 audit points every step). The analysis scripts
themselves run in minutes on CPU from the cached artifacts.

### Group 4 — MNIST data pruning + profiling + TRAK (Appendices A.4, A.5, A.12)

```bash
# --- Data pruning (Figs 21, 22) ---
# Depends on Group 1 artifact WITH IPS.
python scripts/mnist_data_pruning.py \
    --source-artifact data/mnist_experiment1.pt \
    --out data/mnist_pruning_results.pkl
python scripts/mnist_fig21_22_pruning.py

# --- Audit profiling (Table A.5) ---
# Runs all six variants sequentially (~6-10 hours CPU / ~1-2 hours GPU at
# paper defaults). Use --variants to subset.
python scripts/mnist_audit_profiling.py
python scripts/mnist_table_a5_profiling.py

# --- TRAK comparison (Table 4) ---
# Requires: pip install traker
python scripts/mnist_table4_trak_comparison.py
```

Wall-time note: pruning is by far the heaviest — 6 methods × 7 pruning
percentages + 2 random × 7 × 5 seeds = 112 retrains × ~15 min GPU each,
so a full sweep is ~28 hours on one consumer GPU. Reduce with ``--ks``,
``--methods``, ``--random-seeds`` for iteration; full sweep only for the
reproducibility rerun.

### Group 5 — SVHN replication (Appendix A.17) — Fig 38 not included

Small-CNN (~140k params, GroupNorm variant) on SVHN. Mirrors the MNIST
pipeline. See `notes/COVERAGE_MATRIX.md` rows 38-43 for figure mapping.

```bash
# Step 1: train + audit (+ optional IPS with JL projection to 512 dim).
#         ~30 min GPU audit-only, ~60 min with --do-ips.
python scripts/svhn_train_and_audit.py --epochs 50 --train-batches 200
python scripts/svhn_train_and_audit.py --epochs 50 --do-ips  # for the IPS figures

# Step 2: audit/IPS plots
python scripts/svhn_fig28_reconstruction.py   # Fig 28 (Corr + MRE)
python scripts/svhn_fig29_31_worst_error.py   # Figs 29, 30, 31
python scripts/svhn_fig32_34_ips_umap.py      # Figs 32-34 (needs --do-ips artifact)

# Step 3: adversarial (Task A only per paper — no ray, no cure)
python scripts/svhn_adversarial_audit.py --source-artifact data/svhn_experiment1.pt
python scripts/svhn_fig35_adv_triplets.py             # Fig 35
python scripts/svhn_fig36_clean_vs_adv_scatter.py     # Fig 36
python scripts/svhn_fig37_per_epoch_influence.py      # Fig 37
python scripts/svhn_fig39_mode_activity.py --compare adv  # Fig 39
python scripts/svhn_fig39_mode_activity.py --compare rand # Fig 40
```

```bash
# Step 4: profiling (Table A.17 — 4 variants: baseline, audit, supersample_2, subsample_2)
python scripts/svhn_audit_profiling.py
python scripts/svhn_table_a17_profiling.py
```

Fig 38 (Rich→Lazy kernel-regime metrics) is the one paper figure without a
producing script here: it requires a separate re-training pass with
feature/activation/subspace-overlap tracking. `scripts/svhn_fig38_kernel_regime.py`
documents the full metric specification and exits with that message;
`scripts/mnist_fig14_kernel_regime.py` is the closest working template.

### Group 6 — Scaling: CIFAR + ImageNet + ViT (Appendix A.18)

Three scaling rows (Table 5) plus per-sample IPS-vs-audit correlation
(Table 6). Models are BN-free (GroupNorm / LayerNorm) because BatchNorm
breaks per-sample gradient evaluation via ``torch.func.vmap(grad)``.

```bash
# --- Table 5 row 1: CIFAR / ResNet-18-GN (11.2M params) ---
python scripts/scaling/run_cifar10_audit.py \
    --model resnet18gn --epochs 120 --do-ips

# --- Table 5 row 2: CIFAR / ViT-Tiny/4 (2.70M params) ---
# Note the low learning rate; paper's DeiT-standard 5e-4 drops audit corr
# to ~0.28. --lr defaults to 1.5e-6 for --model vittiny4.
python scripts/scaling/run_cifar10_audit.py \
    --model vittiny4 --epochs 120 --do-ips

# --- Table 5 row 3: ImageNet / ResNet-18-GN (IPS-only) ---
# Full audit is 51 GiB at ImageNet scale — IPS-only is the tractable variant.
# Requires ImageFolder-formatted ImageNet at $IMAGENET_ROOT.
python scripts/scaling/run_ips_only.py \
    --dataset imagenet --data-root $IMAGENET_ROOT --epochs 90

# --- Table 5 tabulator ---
python scripts/scaling/scaling_table5.py

# --- Table 6: IPS-vs-audit per-sample correlation across epochs ---
# Each run stores per-checkpoint IPS + audit; use the --epochs list argument
# to rerun the CIFAR driver at each checkpoint horizon (20/50/80/120), then
# the comparator reads the resulting artifacts.
for ep in 20 50 80 120; do
  python scripts/scaling/run_cifar10_audit.py \
      --model resnet18gn --epochs $ep --do-ips \
      --out data/cifar10_resnet18gn_audit_ep${ep}.pt
done
python scripts/scaling/compare_ips_vs_audit.py
```

**Wall time**: on a single A100, CIFAR audit runs are ~4-6h each at 120
epochs; ImageNet IPS-only is a multi-GPU day. On consumer GPUs, multiply
by ~3-5x. CPU is impractical for any of these — use ``--epochs 1
--train-batches 5`` for smoke tests only.

**Paper's Table 5 numbers** (for validation):

| Config | Audit t/epoch | Overhead | Peak CUDA | Test Acc |
|---|---:|---:|---:|---:|
| CIFAR / ResNet-18-GN | 14.9 → 126s | 8.5× | 6.0 GiB | 56.6% |
| CIFAR / ViT-Tiny/4 | 15.8 → 178s | 11.3× | 4.62 GiB | 46.8% |
| ImageNet / ResNet-18-GN (IPS) | 2489 → 6772s | 2.7× | 22.4 GiB | 41.3% |

Paper's Table 6 corr (CIFAR ResNet-GN): 0.941 → 0.945 across 20/50/80/120 ep.

## Datasets

No datasets are distributed with this repository. MNIST, SVHN and CIFAR-10 are
downloaded automatically by ``torchvision`` on first use, into ``data/``
(which is gitignored). ImageNet is the one exception: it is not
auto-downloadable and must be supplied as an ImageFolder tree at
``$IMAGENET_ROOT`` — see Group 6.

Experiment artifacts (``data/*.pt``) are likewise not distributed. They are
regenerable from the scripts and individually large: a Task-A audit artifact
at ``--n-successful 64`` is ~2.7 GB, and an exact IPS matrix at MNIST scale
is ~10 GB.

## Optional GPU acceleration

The audit accumulator keeps its large buffers on the CPU by design, because
at ImageNet scale they do not fit in VRAM. At MNIST scale they total ~2.2 GB
and that residency is the dominant cost: the per-step einsum runs ~70x faster
with the buffers on-device.

Scripts that produce audits (``mnist_adversarial_audit.py``,
``mnist_mode_svd_snapshots.py``) therefore accept ``--gpu-audit``, which is
**off by default** — the CPU path is the reference implementation and a
reproduction needs neither a GPU nor trust in a second code path.

Before relying on it, run the equivalence gate:

```bash
python scripts/verify_gpu_audit.py --steps 20
```

It trains identical short runs under both accumulators and compares the
resulting PNTK matrices (observed: correlation 0.99999999, mean absolute
deviation ~1e-9). Note this establishes that the two *accumulators* agree on
identical inputs; it does not make full training runs bit-reproducible, since
cuDNN kernel selection is nondeterministic regardless of this choice.

## Environment

Python 3.10+, PyTorch 2.x. See `requirements.txt` for pinned versions.
Tested on Windows 11 + WSL2, single CPU and single consumer GPU. See
individual scripts for per-experiment wall-time expectations.

## License and citation

Released under the MIT License (see `LICENSE`).

If you use this code, please cite:

> Ryan Pyle, Yilong Ju, and Ankit Patel. *Path-Integrated Loss-Gradient
> Kernels: Auditing and Similarity for Trained Neural Networks.*
> Transactions on Machine Learning Research, 2026.

`CITATION.cff` carries the same metadata in machine-readable form. Its
top-level `authors` field lists the author of this code; the paper's full
author list is under `preferred-citation`.

## Coverage matrix

`notes/COVERAGE_MATRIX.md` maps every empirical claim in the paper (each
figure, each table, quoted metrics) to the exact producing script. The groups
above are ordered by execution dependency; the matrix is ordered by paper
section, so it is the faster route from a figure number to its producer.
