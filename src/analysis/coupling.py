"""
coupling.py
-----------
Per-gene RNA-protein and RNA-ATAC coupling scores.
All functions operate on numpy arrays; AnnData handling stays in notebooks.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Optional


def per_gene_pearson(
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    min_cells: int = 50,
) -> pd.DataFrame:
    """
    Pearson r between X[:,i] and Y[:,i] for each gene i.
    X, Y must be (cells × genes), same shape, same ordering.

    Returns DataFrame with columns: gene, pearson_r, p_value.
    Genes with too few non-zero observations are skipped.
    """
    assert X.shape == Y.shape, "X and Y must have the same shape"
    records = []
    for i, gene in enumerate(gene_names):
        x, y = X[:, i], Y[:, i]
        nonzero = (x != 0) | (y != 0)
        if nonzero.sum() < min_cells:
            continue
        r, p = pearsonr(x[nonzero], y[nonzero])
        records.append({"gene": gene, "pearson_r": r, "p_value": p})
    return pd.DataFrame(records).sort_values("pearson_r", ascending=False)


def coupling_score_by_celltype(
    X: np.ndarray,
    Y: np.ndarray,
    gene_names: list[str],
    cell_types: np.ndarray,
    min_cells: int = 50,
) -> pd.DataFrame:
    """
    Per-gene Pearson r stratified by cell type.
    Returns a (genes × cell_types) DataFrame of Pearson r values.
    """
    results = {}
    for ct in np.unique(cell_types):
        mask = cell_types == ct
        if mask.sum() < min_cells:
            continue
        df = per_gene_pearson(X[mask], Y[mask], gene_names, min_cells)
        results[ct] = df.set_index("gene")["pearson_r"]
    return pd.DataFrame(results)


def deviation_score(
    X_rna: np.ndarray,
    X_protein: np.ndarray,
) -> np.ndarray:
    """
    Per-cell deviation: mean absolute residual from the linear
    RNA→protein fit across all genes.
    Higher score = protein diverges more from RNA in that cell.

    Returns (n_cells,) array.
    """
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_rna, X_protein)
    residuals = X_protein - model.predict(X_rna)
    return np.abs(residuals).mean(axis=1)
