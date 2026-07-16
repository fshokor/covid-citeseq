"""Evaluation and interpretability helpers for all RNA -> protein model variants.

Lives in src/analysis/, separate from the model classes in src/models/. Model
type is detected by duck-typing on attribute names (direct / linear / fc1)
rather than isinstance checks against src.models.architectures classes --
this file stays a standalone dependency, consistent with how other files in
src/analysis/ (coupling.py, cd_gene_mapping.py) are loaded independently via
importlib in the notebooks.

Expected attribute shapes (see src/models/architectures.py):
    LassoLinear        -> model.linear   (gene -> protein)
    MLP                 -> model.fc1, model.fc2   (gene -> hidden -> protein)
    SkipConnectionMLP    -> model.direct, model.fc1, model.fc2
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

SPARSITY_THRESH = 1e-3  # near-zero cutoff for effective-gene / active-unit counts


def _architecture_kind(model: nn.Module) -> str:
    """Classify a model by attribute shape: 'skip', 'lasso', or 'mlp'."""
    if hasattr(model, "direct") and hasattr(model, "fc1"):
        return "skip"
    if hasattr(model, "linear"):
        return "lasso"
    if hasattr(model, "fc1") and hasattr(model, "fc2"):
        return "mlp"
    raise TypeError(f"Unrecognized model shape for {type(model)}: no direct/linear/fc1 found.")


# --------------------------------------------------------------------------
# Prediction accuracy
# --------------------------------------------------------------------------

def evaluate_per_protein(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    protein_names: list[str],
    return_preds: bool = False,
):
    """Per-protein Pearson r and R2 on CPU. Optionally also return raw predictions."""
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()
    rows = []
    for i, name in enumerate(protein_names):
        real, pred = Y[:, i], preds[:, i]
        if np.std(real) == 0 or np.std(pred) == 0:
            r, r2 = np.nan, np.nan
        else:
            r, _ = pearsonr(real, pred)
            r2 = r2_score(real, pred)
        rows.append({"protein": name, "pearson_r": r, "r2": r2})
    metrics = pd.DataFrame(rows)
    return (metrics, preds) if return_preds else metrics


def train_test_metrics(
    model: nn.Module,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    protein_names: list[str],
) -> pd.DataFrame:
    """Merge train and test per-protein metrics into one table with the train-test gap."""
    train_m = evaluate_per_protein(model, X_train, Y_train, protein_names)
    test_m = evaluate_per_protein(model, X_test, Y_test, protein_names)
    merged = train_m.merge(test_m, on="protein", suffixes=("_train", "_test"))
    merged["r_gap"] = merged["pearson_r_train"] - merged["pearson_r_test"]
    return merged.sort_values("pearson_r_test", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Importance
# --------------------------------------------------------------------------

def compute_importance(
    model: nn.Module,
    gene_names: list[str],
    protein_names: list[str],
    path: str = "auto",
) -> pd.DataFrame:
    """Per-architecture importance matrix, shape (n_proteins, n_genes).

    lasso  -> |W|
    mlp    -> |W2 @ W1| (path-weight product through the hidden layer)
    skip   -> path controls which term:
                'direct'   -> |W_direct|
                'hidden'   -> |W2 @ W1|
                'combined' -> |W_direct| + |W2 @ W1|
                'auto'     -> 'combined'
    """
    model.to("cpu")
    kind = _architecture_kind(model)

    if kind == "lasso":
        W = model.linear.weight.detach().numpy()
        return pd.DataFrame(np.abs(W), index=protein_names, columns=gene_names)

    if kind == "mlp":
        W1 = model.fc1.weight.detach().numpy()
        W2 = model.fc2.weight.detach().numpy()
        return pd.DataFrame(np.abs(W2 @ W1), index=protein_names, columns=gene_names)

    # kind == "skip"
    W_direct = np.abs(model.direct.weight.detach().numpy())
    W1 = model.fc1.weight.detach().numpy()
    W2 = model.fc2.weight.detach().numpy()
    W_hidden = np.abs(W2 @ W1)

    if path == "direct":
        importance = W_direct
    elif path == "hidden":
        importance = W_hidden
    elif path in ("combined", "auto"):
        importance = W_direct + W_hidden
    else:
        raise ValueError(f"Unknown path {path!r} for a skip-connection model.")
    return pd.DataFrame(importance, index=protein_names, columns=gene_names)


def cognate_gene_rank(importance_df: pd.DataFrame, gene_map: pd.DataFrame) -> pd.DataFrame:
    """Rank of each protein's cognate RNA gene by importance score (1 = top predictor).

    gene_map must have 'gene' and 'adt_name' columns (protein <-> cognate gene).
    """
    rows = []
    for _, row in gene_map.iterrows():
        gene, adt = row["gene"], row["adt_name"]
        if adt not in importance_df.index or gene not in importance_df.columns:
            continue
        ranks = importance_df.loc[adt].rank(ascending=False)
        top_gene = importance_df.loc[adt].idxmax()
        rows.append({
            "protein": adt,
            "cognate_gene": gene,
            "cognate_rank": int(ranks[gene]),
            "n_genes": len(ranks),
            "top_predictor_gene": top_gene,
        })
    return pd.DataFrame(rows).sort_values("cognate_rank").reset_index(drop=True)


def effective_genes_per_protein(
    model: nn.Module,
    protein_names: list[str],
    gene_names: list[str],
    thresh: float = SPARSITY_THRESH,
) -> pd.Series:
    """Median genes actually feeding each protein's prediction, post-sparsity.

    lasso -> nonzero entries in that protein's weight row.
    mlp   -> active hidden units (fc2) x active genes feeding them (fc1), union.
    skip  -> nonzero entries in the direct-path weight row (the hidden path is
             not sparsity-counted here -- it's a shared mediated signal, not
             per-gene direct evidence).
    """
    model.to("cpu")
    kind = _architecture_kind(model)

    if kind == "lasso":
        W = model.linear.weight.detach().numpy()
        counts = (np.abs(W) >= thresh).sum(axis=1)
        return pd.Series(counts, index=protein_names)

    if kind == "skip":
        W = model.direct.weight.detach().numpy()
        counts = (np.abs(W) >= thresh).sum(axis=1)
        return pd.Series(counts, index=protein_names)

    # kind == "mlp"
    W1 = model.fc1.weight.detach().numpy()
    W2 = model.fc2.weight.detach().numpy()
    counts = []
    for pi in range(len(protein_names)):
        active_units = np.where(np.abs(W2[pi, :]) >= thresh)[0]
        if len(active_units) == 0:
            counts.append(0)
            continue
        active_genes = (np.abs(W1[active_units, :]) >= thresh).any(axis=0)
        counts.append(int(active_genes.sum()))
    return pd.Series(counts, index=protein_names)


# --------------------------------------------------------------------------
# Raw correlation sanity check
# --------------------------------------------------------------------------

def raw_correlation_check(
    cognate_ranks: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    protein_names: list[str],
    weak_corr_thresh: float = 0.1,
    strong_corr_thresh: float = 0.3,
) -> pd.DataFrame:
    """Model-independent sanity check: raw Pearson r, no model weights involved.

    For each protein, computes its raw correlation with its cognate gene and
    with the model's chosen top-predictor gene directly from the data. This
    catches two failure modes that importance-weight-based metrics alone can miss:

    cognate_ranks must have columns 'protein', 'cognate_gene', 'cognate_rank',
    'top_predictor_gene' (the output of cognate_gene_rank).

    Adds two flags:
      top_predictor_weak_raw_corr   -- the model's #1 pick barely correlates with
                                        the protein at all -- possible weight
                                        artifact (e.g. a large weight on a gene
                                        that happens to have near-zero variance),
                                        not a real predictive relationship.
      cognate_strong_but_not_top    -- the cognate gene has a strong raw
                                        correlation but the model still didn't
                                        rank it #1 -- the model missed an easy case.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    protein_idx = {p: i for i, p in enumerate(protein_names)}

    def raw_pearson(protein_col: np.ndarray, gene_name) -> float:
        if gene_name not in gene_idx or (isinstance(gene_name, float) and np.isnan(gene_name)):
            return np.nan
        return pearsonr(protein_col, X[:, gene_idx[gene_name]])[0]

    rows = []
    for _, row in cognate_ranks.iterrows():
        protein, cognate_gene, top_gene = row["protein"], row["cognate_gene"], row["top_predictor_gene"]
        if protein not in protein_idx:
            continue
        y = Y[:, protein_idx[protein]]
        rows.append({
            "protein": protein,
            "cognate_gene": cognate_gene,
            "cognate_rank": row["cognate_rank"],
            "cognate_raw_r": raw_pearson(y, cognate_gene),
            "top_predictor_gene": top_gene,
            "top_predictor_raw_r": raw_pearson(y, top_gene),
        })

    check_df = pd.DataFrame(rows)
    check_df["top_predictor_weak_raw_corr"] = check_df["top_predictor_raw_r"].abs() < weak_corr_thresh
    check_df["cognate_strong_but_not_top"] = (
        (check_df["cognate_raw_r"].abs() > strong_corr_thresh) & (check_df["cognate_rank"] > 1)
    )
    return check_df


def raw_correlation_summary(check_df: pd.DataFrame) -> dict:
    """Condense a raw_correlation_check() table into headline numbers for one variant.

    raw_corr_rank_spearman_rho is expected negative: a stronger raw cognate
    correlation should correspond to a better (lower) model rank. A value near
    zero or positive means the model's ranking isn't tracking the raw signal at all.
    """
    rho, p = spearmanr(check_df["cognate_raw_r"].abs(), check_df["cognate_rank"])
    return {
        "raw_corr_rank_spearman_rho": rho,
        "raw_corr_rank_spearman_p": p,
        "n_weak_raw_corr_top_pick": int(check_df["top_predictor_weak_raw_corr"].sum()),
        "n_strong_cognate_not_top": int(check_df["cognate_strong_but_not_top"].sum()),
    }
