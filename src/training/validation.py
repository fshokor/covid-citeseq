"""Bootstrap-stability and permutation-null validation for RNA -> protein models.

Generalizes nb07's validation methodology (aggregate top-K bootstrap
recurrence + permutation-null significance) and nb09's addition
(rank-1-specific stability) to work with any variant registered in
src.models.architectures.VARIANTS, not just one hardcoded architecture.

Every function here retrains a model, so validating one variant costs
n_bootstraps + n_null_perms extra fit_model calls. Use a smaller
aux_num_epochs/aux_patience than the reference model's training run to keep
this bounded -- these are cheaper diagnostic runs, not the final fit.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.architectures import build_variant
from src.training.train import fit_model, make_loader
from src.analysis.evaluate import compute_importance


@dataclass
class ValidationConfig:
    """Shared config for every bootstrap/null retrain of a given variant."""

    rna_dim: int
    protein_dim: int
    hidden_dim: Optional[int]
    batch_size: int
    train_device: torch.device
    aux_num_epochs: int = 40
    aux_patience: int = 8


# --------------------------------------------------------------------------
# Single retrain runs
# --------------------------------------------------------------------------

def train_on_bootstrap(
    variant_name: str,
    train_idx: np.ndarray,
    val_loader: DataLoader,
    X: np.ndarray,
    Y: np.ndarray,
    config: ValidationConfig,
    seed: int,
) -> nn.Module:
    """Train `variant_name` on a bootstrap resample of train_idx (with replacement).

    val_loader is passed in already built (fixed across all bootstrap runs for
    a given variant) so it isn't rebuilt on every call.
    """
    rng = np.random.RandomState(seed)
    boot_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
    boot_loader = make_loader(X, Y, boot_idx, config.batch_size, shuffle=True)

    model, penalty_fn = build_variant(
        variant_name, rna_dim=config.rna_dim, protein_dim=config.protein_dim, hidden_dim=config.hidden_dim,
    )
    model, _ = fit_model(
        model=model,
        train_loader=boot_loader,
        val_loader=val_loader,
        penalty_fn=penalty_fn,
        lr=1e-4,
        num_epochs=config.aux_num_epochs,
        patience=config.aux_patience,
        train_device=config.train_device,
        checkpoint_path=None,
        verbose=False,
    )
    return model


def train_on_permuted_protein(
    variant_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    config: ValidationConfig,
    seed: int,
) -> nn.Module:
    """Train `variant_name` with protein rows shuffled -- destroys real RNA-protein pairing.

    Protein-protein covariance is preserved (rows are shuffled as a unit), only
    the RNA<->protein cell pairing is destroyed -- so any importance signal
    that emerges here is a pure model-capacity/noise artifact, useful as a
    per-protein significance floor.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(X.shape[0])
    Y_perm = Y[perm]

    perm_train_loader = make_loader(X, Y_perm, train_idx, config.batch_size, shuffle=True)
    perm_val_loader = make_loader(X, Y_perm, val_idx, config.batch_size, shuffle=False)

    model, penalty_fn = build_variant(
        variant_name, rna_dim=config.rna_dim, protein_dim=config.protein_dim, hidden_dim=config.hidden_dim,
    )
    model, _ = fit_model(
        model=model,
        train_loader=perm_train_loader,
        val_loader=perm_val_loader,
        penalty_fn=penalty_fn,
        lr=1e-4,
        num_epochs=config.aux_num_epochs,
        patience=config.aux_patience,
        train_device=config.train_device,
        checkpoint_path=None,
        verbose=False,
    )
    return model


# --------------------------------------------------------------------------
# Bootstrap stability
# --------------------------------------------------------------------------

def bootstrap_stability(
    reference_model: nn.Module,
    variant_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    protein_names: list[str],
    config: ValidationConfig,
    n_bootstraps: int = 5,
    top_k: int = 20,
    importance_path: str = "auto",
    seed_offset: int = 2000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain on n_bootstraps resamples; score recurrence against the reference model.

    Two views on the same runs:
      agg_df    -- for each protein's reference top-K genes: importance score and
                   bootstrap_frequency (fraction of bootstrap runs where that gene
                   also lands in that run's own top-K).
      rank1_df  -- for each protein: the reference #1 gene, and rank1_match_frequency
                   (fraction of bootstrap runs where the SAME gene is #1 again).
                   More decision-relevant than the aggregate metric, which mixes
                   a very stable signal (rank 1) with noisier ones (ranks 2-K).
    """
    val_loader = make_loader(X, Y, val_idx, config.batch_size, shuffle=False)
    reference_importance = compute_importance(reference_model, gene_names, protein_names, path=importance_path)

    top_k_sets = {p: [] for p in protein_names}
    top1_genes_boot = {p: [] for p in protein_names}

    for b in range(n_bootstraps):
        if verbose:
            print(f"  Bootstrap {b + 1}/{n_bootstraps} ({variant_name})...")
        boot_model = train_on_bootstrap(variant_name, train_idx, val_loader, X, Y, config, seed=seed_offset + b)
        boot_importance = compute_importance(boot_model, gene_names, protein_names, path=importance_path)
        for protein in protein_names:
            ranked = boot_importance.loc[protein].sort_values(ascending=False)
            top_k_sets[protein].append(set(ranked.head(top_k).index))
            top1_genes_boot[protein].append(ranked.index[0])

    agg_rows = []
    rank1_rows = []
    for protein in protein_names:
        ranked = reference_importance.loc[protein].sort_values(ascending=False)
        ref_top1 = ranked.index[0]

        for gene, score in ranked.head(top_k).items():
            freq = sum(gene in s for s in top_k_sets[protein]) / len(top_k_sets[protein])
            agg_rows.append({"protein": protein, "gene": gene, "importance": score, "bootstrap_frequency": freq})

        rank1_freq = sum(g == ref_top1 for g in top1_genes_boot[protein]) / len(top1_genes_boot[protein])
        rank1_rows.append({"protein": protein, "reference_top1_gene": ref_top1, "rank1_match_frequency": rank1_freq})

    return pd.DataFrame(agg_rows), pd.DataFrame(rank1_rows)


# --------------------------------------------------------------------------
# Permutation null
# --------------------------------------------------------------------------

def permutation_null(
    variant_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    protein_names: list[str],
    config: ValidationConfig,
    n_null_perms: int = 5,
    percentile: float = 95.0,
    importance_path: str = "auto",
    seed_offset: int = 3000,
    verbose: bool = True,
) -> pd.Series:
    """Per-protein significance threshold from n_null_perms shuffled-protein retrains.

    Pools importance scores (across all genes) from every null run for a given
    protein, then takes the given percentile as that protein's noise floor.
    """
    null_importances = []
    for n in range(n_null_perms):
        if verbose:
            print(f"  Null permutation {n + 1}/{n_null_perms} ({variant_name})...")
        null_model = train_on_permuted_protein(variant_name, train_idx, val_idx, X, Y, config, seed=seed_offset + n)
        null_importances.append(compute_importance(null_model, gene_names, protein_names, path=importance_path))

    thresholds = {}
    for protein in protein_names:
        pooled = np.concatenate([df.loc[protein].values for df in null_importances])
        thresholds[protein] = np.percentile(pooled, percentile)
    return pd.Series(thresholds, name="null_threshold")


def validated_pairs(
    agg_df: pd.DataFrame,
    null_thresholds: pd.Series,
    min_bootstrap_frequency: float = 0.8,
) -> pd.DataFrame:
    """Merge bootstrap stability + null threshold: validated iff both criteria clear.

    agg_df must have 'protein', 'gene', 'importance', 'bootstrap_frequency'
    columns (the output of bootstrap_stability's first return value).
    """
    merged = agg_df.merge(null_thresholds.rename("null_threshold"), left_on="protein", right_index=True)
    merged["above_null"] = merged["importance"] > merged["null_threshold"]
    merged["validated"] = merged["above_null"] & (merged["bootstrap_frequency"] >= min_bootstrap_frequency)
    return merged


# --------------------------------------------------------------------------
# Convenience entry point
# --------------------------------------------------------------------------

def validate_variant(
    reference_model: nn.Module,
    variant_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    protein_names: list[str],
    config: ValidationConfig,
    n_bootstraps: int = 5,
    n_null_perms: int = 5,
    top_k: int = 20,
    min_bootstrap_frequency: float = 0.8,
    null_percentile: float = 95.0,
    importance_path: str = "auto",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Full bootstrap-stability + permutation-null validation for one trained variant.

    Runs bootstrap_stability then permutation_null then validated_pairs, and
    returns everything in one dict:
      'bootstrap_topk'   -- agg_df from bootstrap_stability
      'bootstrap_rank1'  -- rank1_df from bootstrap_stability
      'null_thresholds'  -- per-protein null threshold, as a DataFrame
      'validated'        -- agg_df merged with null thresholds + validated flag
    """
    agg_df, rank1_df = bootstrap_stability(
        reference_model, variant_name, train_idx, val_idx, X, Y, gene_names, protein_names,
        config, n_bootstraps=n_bootstraps, top_k=top_k, importance_path=importance_path, verbose=verbose,
    )
    thresholds = permutation_null(
        variant_name, train_idx, val_idx, X, Y, gene_names, protein_names,
        config, n_null_perms=n_null_perms, percentile=null_percentile,
        importance_path=importance_path, verbose=verbose,
    )
    validated = validated_pairs(agg_df, thresholds, min_bootstrap_frequency=min_bootstrap_frequency)

    return {
        "bootstrap_topk": agg_df,
        "bootstrap_rank1": rank1_df,
        "null_thresholds": thresholds.reset_index().rename(columns={"index": "protein"}),
        "validated": validated,
    }
