# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""PLGK Paper — supporting library.

Consolidates the training/audit/IPS pipeline shared by the paper's MNIST and
SVHN experiments. This is the CPU/single-GPU reference implementation used
to produce the paper's main-text figures; the scaled ImageNet/ViT variants
live in the sibling ``scaling/`` subtree (see ``scripts/README.md``).
"""

from .models import LNModel, SVHNModel, count_params
from .audit import AuditAccumulator
from .ips import IPSAccumulator
from .train import train_with_audit, collect_full_eval
from .data import mnist_loaders, svhn_loaders
from .scaling_models import ResNet18GN, ResNet18LN, ViTTiny4
from .scaling_data import cifar10_loaders, imagenet_loaders
from .ips_scores import (
    compute_ips_scores,
    compute_hidden_scores,
    compute_confidence_scores,
    evaluate_error_detection,
    evaluate_loss_prediction,
    run_ips_utility_experiments,
)
from .umap_utils import ips_to_distance, umap_from_ips
from .adversarial_data import build_adversarial_audit_sets, DEFAULT_EPSILONS
from .adversarial_cure import adversarial_cure, build_cure_data
from .svd_modes import (
    lowrank_svd_train_side,
    compute_param_shapes,
    flat_modes_to_param_list,
    param_list_to_flat_modes,
    modal_cancellation_ratio,
)
from .population import (
    adversarial_group_masks,
    mmd,
    bures_distance,
    permutation_test,
    project_pooled,
    PopulationDistance,
)
from .pruning import (
    PruningScores,
    compute_pruning_scores,
    compute_ips_spectral_clusters,
    get_harmful_indices,
    get_least_impactful_abs_indices,
    get_least_impactful_signed_indices,
    get_most_redundant_indices,
    get_class_balanced_least_impactful_indices,
    get_random_indices,
    get_random_class_balanced_indices,
    get_cluster_prune_indices,
    # destructive counterparts (Fig 22)
    get_most_helpful_indices,
    get_highest_impact_abs_indices,
    get_highest_impact_signed_indices,
    get_class_bal_highest_impact_indices,
    get_smallest_clusters_indices,
)

__all__ = [
    "LNModel",
    "SVHNModel",
    "count_params",
    "AuditAccumulator",
    "IPSAccumulator",
    "train_with_audit",
    "collect_full_eval",
    "mnist_loaders",
    "svhn_loaders",
    "ResNet18GN",
    "ResNet18LN",
    "ViTTiny4",
    "cifar10_loaders",
    "imagenet_loaders",
    "compute_ips_scores",
    "compute_hidden_scores",
    "compute_confidence_scores",
    "evaluate_error_detection",
    "evaluate_loss_prediction",
    "run_ips_utility_experiments",
    "ips_to_distance",
    "umap_from_ips",
    "build_adversarial_audit_sets",
    "DEFAULT_EPSILONS",
    "adversarial_cure",
    "build_cure_data",
    "lowrank_svd_train_side",
    "compute_param_shapes",
    "flat_modes_to_param_list",
    "param_list_to_flat_modes",
    "modal_cancellation_ratio",
    "adversarial_group_masks",
    "mmd",
    "bures_distance",
    "permutation_test",
    "project_pooled",
    "PopulationDistance",
    "PruningScores",
    "compute_pruning_scores",
    "compute_ips_spectral_clusters",
    "get_harmful_indices",
    "get_least_impactful_abs_indices",
    "get_least_impactful_signed_indices",
    "get_most_redundant_indices",
    "get_class_balanced_least_impactful_indices",
    "get_random_indices",
    "get_random_class_balanced_indices",
    "get_cluster_prune_indices",
    "get_most_helpful_indices",
    "get_highest_impact_abs_indices",
    "get_highest_impact_signed_indices",
    "get_class_bal_highest_impact_indices",
    "get_smallest_clusters_indices",
]
