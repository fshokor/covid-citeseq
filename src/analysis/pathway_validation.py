"""Biological validation of gene-protein pairs against three external databases.

Ports nb10's STRING / OmniPath / TRRUST checks into reusable functions. Unlike
nb10, caches (STRING partners, OmniPath complexes, TRRUST table) are explicit
parameters/return values rather than module-level globals -- callers own the
cache and can persist it across notebook runs however they like.

1. STRING    -- functional/physical association, with per-evidence-channel
                scores (escore=experimental, dscore=curated DB, tscore=text-mining)
                so a hard-evidence hit can be told apart from literature-mining only.
2. OmniPath  -- shared-complex membership (CORUM, ComplexPortal, hu.MAP by default).
3. TRRUST    -- literature-curated transcription-factor -> target-gene regulation,
                checked in both directions.
"""

import io
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

STRING_API_BASE = "https://string-db.org/api"
STRING_SPECIES = 9606  # human
STRING_MIN_SCORE = 0.4  # STRING's own 'medium confidence' cutoff, 0-1 scale
REQUEST_DELAY_SEC = 1.0  # be polite to the STRING API

OMNIPATH_COMPLEX_RESOURCES = "CORUM,ComplexPortal,hu.MAP"
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"

STRING_EVIDENCE_COLS = ["score", "escore", "dscore", "tscore", "ascore", "nscore", "fscore", "pscore"]


# --------------------------------------------------------------------------
# 1. STRING
# --------------------------------------------------------------------------

def get_string_partners(gene_symbol: str, species: int = STRING_SPECIES, limit: int = 50) -> pd.DataFrame:
    """Fetch a gene's top STRING interaction partners, with per-channel evidence scores.

    Returns an empty DataFrame on any request/parse failure -- callers should
    treat that the same as "no known partners", not raise.
    """
    url = f"{STRING_API_BASE}/tsv/interaction_partners"
    params = {"identifiers": gene_symbol, "species": species, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        if not resp.text.strip():
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(resp.text), sep="\t")
    except (requests.RequestException, pd.errors.ParserError):
        return pd.DataFrame()


def fetch_string_partner_cache(
    genes: list[str],
    species: int = STRING_SPECIES,
    limit: int = 50,
    delay_sec: float = REQUEST_DELAY_SEC,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch and cache STRING partners for a list of genes, rate-limited.

    Call once per unique gene set; pass the returned cache into
    check_string_interaction repeatedly rather than re-fetching per pair.
    """
    cache: dict[str, pd.DataFrame] = {}
    for i, gene in enumerate(genes):
        cache[gene] = get_string_partners(gene, species=species, limit=limit)
        time.sleep(delay_sec)
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(genes)}...")
    return cache


def check_string_interaction(gene_a: str, gene_b: str, partner_cache: dict[str, pd.DataFrame]) -> dict:
    """Look up gene_b among gene_a's cached STRING partners (checked both directions).

    Returns a dict of evidence scores (empty dict if no known interaction).
    """
    for query, target in [(gene_a, gene_b), (gene_b, gene_a)]:
        partners = partner_cache.get(query, pd.DataFrame())
        if not partners.empty and "preferredName_B" in partners.columns:
            match = partners[partners["preferredName_B"].str.upper() == target.upper()]
            if not match.empty:
                return {col: float(match[col].iloc[0]) for col in STRING_EVIDENCE_COLS if col in match.columns}
    return {}


def string_validate_pairs(
    pairs: pd.DataFrame,
    partner_cache: dict[str, pd.DataFrame],
    gene_a_col: str = "cognate_gene",
    gene_b_col: str = "top_predictor_gene",
    min_score: float = STRING_MIN_SCORE,
) -> pd.DataFrame:
    """Check every (gene_a_col, gene_b_col) row against the STRING partner cache.

    Adds string_known_interaction, string_<evidence_col> for each channel,
    string_high_confidence, string_hard_evidence (experimental or curated DB),
    and string_textmining_only.
    """
    rows = []
    for _, row in pairs.iterrows():
        evidence = check_string_interaction(row[gene_a_col], row[gene_b_col], partner_cache)
        result = {c: row[c] for c in pairs.columns}
        result["string_known_interaction"] = bool(evidence)
        for col in STRING_EVIDENCE_COLS:
            result[f"string_{col}"] = evidence.get(col, np.nan)
        rows.append(result)

    out = pd.DataFrame(rows)
    out["string_high_confidence"] = out["string_score"] >= min_score
    out["string_hard_evidence"] = (
        (out["string_escore"].fillna(0) >= min_score) | (out["string_dscore"].fillna(0) >= min_score)
    )
    out["string_textmining_only"] = (
        out["string_known_interaction"]
        & ~out["string_hard_evidence"]
        & (out["string_tscore"].fillna(0) >= min_score)
    )
    return out


# --------------------------------------------------------------------------
# 2. OmniPath complexes
# --------------------------------------------------------------------------

def load_omnipath_complexes(
    resources: str = OMNIPATH_COMPLEX_RESOURCES,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch complex data from OmniPath, filtered to the given resource list.

    If cache_path is given and exists, reads from it instead of hitting the
    network; on a successful fetch, writes to cache_path if provided.
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    url = "https://omnipathdb.org/complexes"
    params = {"resources": resources}
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        if cache_path is not None:
            df.to_csv(cache_path, sep="\t", index=False)
        return df
    except (requests.RequestException, pd.errors.ParserError) as e:
        print(f"OmniPath fetch failed ({e}).")
        return pd.DataFrame()


def build_gene_to_complexes(
    complexes: pd.DataFrame,
    subunit_col: str = "components_genesymbols",
    name_col: str = "name",
) -> dict[str, set]:
    """Map each gene symbol to the set of complex names it's a subunit of."""
    gene_to_complexes: dict[str, set] = {}
    if complexes.empty or subunit_col not in complexes.columns:
        return gene_to_complexes
    for _, row in complexes.iterrows():
        subunits = str(row[subunit_col]).split("_")
        complex_name = row.get(name_col, "unknown")
        for gene in subunits:
            gene = gene.strip().upper()
            gene_to_complexes.setdefault(gene, set()).add(complex_name)
    return gene_to_complexes


def check_shared_complex(gene_a: str, gene_b: str, gene_to_complexes: dict[str, set]) -> tuple[bool, str]:
    """Check whether two genes share at least one complex."""
    shared = gene_to_complexes.get(gene_a.upper(), set()) & gene_to_complexes.get(gene_b.upper(), set())
    return (len(shared) > 0, "; ".join(sorted(shared)))


def complex_validate_pairs(
    pairs: pd.DataFrame,
    gene_to_complexes: dict[str, set],
    gene_a_col: str = "cognate_gene",
    gene_b_col: str = "top_predictor_gene",
) -> pd.DataFrame:
    """Check every pair for shared-complex membership. Adds shared_complex, complex_names."""
    rows = []
    for _, row in pairs.iterrows():
        shared, names = check_shared_complex(row[gene_a_col], row[gene_b_col], gene_to_complexes)
        result = {c: row[c] for c in pairs.columns}
        result["shared_complex"] = shared
        result["complex_names"] = names
        rows.append(result)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. TRRUST
# --------------------------------------------------------------------------

def load_trrust(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """Load TRRUST human TF-target data. Falls back to cache_path if the download fails."""
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path, sep="\t", header=None, names=["tf", "target", "regtype", "references"])
    try:
        resp = requests.get(TRRUST_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None, names=["tf", "target", "regtype", "references"])
        if cache_path is not None:
            df.to_csv(cache_path, sep="\t", index=False, header=False)
        return df
    except (requests.RequestException, pd.errors.ParserError) as e:
        print(f"TRRUST download failed ({e}) -- pass a local cache_path or download manually from "
              "https://www.grnpedia.org/trrust/")
        return pd.DataFrame()


def check_trrust_regulation(gene_a: str, gene_b: str, trrust: pd.DataFrame) -> dict:
    """Check whether gene_a regulates gene_b, or vice versa, per TRRUST."""
    if trrust.empty:
        return {"trrust_regulatory_pair": False, "trrust_direction": None, "trrust_regtype": None}
    a_to_b = trrust[(trrust["tf"].str.upper() == gene_a.upper()) & (trrust["target"].str.upper() == gene_b.upper())]
    if not a_to_b.empty:
        return {"trrust_regulatory_pair": True, "trrust_direction": f"{gene_a}->{gene_b}", "trrust_regtype": a_to_b["regtype"].iloc[0]}
    b_to_a = trrust[(trrust["tf"].str.upper() == gene_b.upper()) & (trrust["target"].str.upper() == gene_a.upper())]
    if not b_to_a.empty:
        return {"trrust_regulatory_pair": True, "trrust_direction": f"{gene_b}->{gene_a}", "trrust_regtype": b_to_a["regtype"].iloc[0]}
    return {"trrust_regulatory_pair": False, "trrust_direction": None, "trrust_regtype": None}


def trrust_validate_pairs(
    pairs: pd.DataFrame,
    trrust: pd.DataFrame,
    gene_a_col: str = "cognate_gene",
    gene_b_col: str = "top_predictor_gene",
) -> pd.DataFrame:
    """Check every pair for TF->target regulation, either direction. Adds trrust_* columns."""
    rows = []
    for _, row in pairs.iterrows():
        result = {c: row[c] for c in pairs.columns}
        result.update(check_trrust_regulation(row[gene_a_col], row[gene_b_col], trrust))
        rows.append(result)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Combined verdict
# --------------------------------------------------------------------------

def combined_verdict(
    string_results: pd.DataFrame,
    complex_results: pd.DataFrame,
    trrust_results: pd.DataFrame,
    key_col: str = "protein",
) -> pd.DataFrame:
    """Merge the three checks and flag whether any source supports each pair.

    A pair is 'biologically_supported' if STRING, shared-complex membership,
    or TRRUST regulation confirms it. evidence_type picks the strongest
    single reason, in priority order: complex > regulatory > STRING hard
    evidence > STRING (any) > none.

    Preserves 'cognate_gene' and 'top_predictor_gene' from string_results if
    present (they usually are -- string_validate_pairs etc. carry over every
    input column), so the returned table -- and anything saved from it, like
    unexplained_pairs.csv -- still identifies which genes each row is about.
    """
    id_cols = [c for c in ("cognate_gene", "top_predictor_gene") if c in string_results.columns]
    string_cols = [key_col, *id_cols, "string_known_interaction", "string_hard_evidence", "string_score"]
    complex_cols = [key_col, "shared_complex", "complex_names"]
    trrust_cols = [key_col, "trrust_regulatory_pair", "trrust_direction", "trrust_regtype"]

    combined = (
        string_results[[c for c in string_cols if c in string_results.columns]]
        .merge(complex_results[[c for c in complex_cols if c in complex_results.columns]], on=key_col)
        .merge(trrust_results[[c for c in trrust_cols if c in trrust_results.columns]], on=key_col)
    )

    combined["biologically_supported"] = (
        combined["string_known_interaction"] | combined["shared_complex"] | combined["trrust_regulatory_pair"]
    )
    combined["evidence_type"] = np.select(
        [combined["shared_complex"], combined["trrust_regulatory_pair"], combined["string_hard_evidence"],
         combined["string_known_interaction"]],
        ["complex", "regulatory", "string_hard_evidence", "string_textmining_or_other"],
        default="none",
    )
    return combined


# --------------------------------------------------------------------------
# Query-set scoping (nb09's trustworthy-core / artifact-flag logic)
# --------------------------------------------------------------------------
# Pathway lookups (esp. STRING, rate-limited) are wasted on pairs the model
# already got right (trustworthy core) or that are already known artifacts --
# scope down to the ambiguous middle before spending API calls.

TECHNICAL_GENE_PATTERNS = ["^HB[ABDGEZQ]", "^MT-", "^RPS", "^RPL"]  # hemoglobin, mitochondrial, ribosomal

# Broader ambient/housekeeping markers: cytoskeletal, glycolytic, translation
# elongation factors, ferritin, beta-2-microglobulin -- genes so highly and
# ubiquitously expressed that a "top predictor" hit on one of these is more
# likely reflecting overall cell size/health/ambient RNA than a specific
# regulatory relationship with the protein in question.
EXTENDED_TECHNICAL_GENE_PATTERNS = TECHNICAL_GENE_PATTERNS + [
    "^ACTB$", "^ACTG1$",       # beta/gamma actin
    "^TUBB", "^TUBA",          # tubulin
    "^GAPDH$",                 # glycolysis
    "^B2M$",                   # beta-2-microglobulin, ubiquitous
    "^TMSB4X$", "^TMSB10$",    # thymosin beta, very high ambient expression
    "^EEF1A1$", "^EEF2$",      # translation elongation factors
    "^FTL$", "^FTH1$",         # ferritin
    "^TPT1$",                  # tumor protein translationally controlled
]


def flag_technical_genes(genes: pd.Series, technical_gene_patterns: list[str] = TECHNICAL_GENE_PATTERNS) -> pd.Series:
    """Boolean mask: does each gene name match a known ambient/technical marker pattern?"""
    pattern = re.compile("|".join(technical_gene_patterns))
    return genes.apply(lambda g: bool(pattern.match(str(g))))


def build_trustworthy_core(
    cognate_ranks: pd.DataFrame,
    bootstrap_rank1: pd.DataFrame,
    min_bootstrap_frequency: float = 0.8,
) -> pd.DataFrame:
    """Proteins where the model's #1 gene is BOTH the literal cognate gene AND bootstrap-stable.

    cognate_ranks: output of evaluate.cognate_gene_rank (needs 'protein', 'cognate_rank').
    bootstrap_rank1: 'bootstrap_rank1' entry from training.validation.validate_variant's
                      output dict (needs 'protein', 'rank1_match_frequency').
    """
    merged = cognate_ranks.merge(bootstrap_rank1[["protein", "rank1_match_frequency"]], on="protein")
    merged["is_cognate_rank1"] = merged["cognate_rank"] == 1
    merged["is_bootstrap_stable"] = merged["rank1_match_frequency"] >= min_bootstrap_frequency
    merged["trustworthy"] = merged["is_cognate_rank1"] & merged["is_bootstrap_stable"]
    return merged


def flag_artifacts(
    check_df: pd.DataFrame,
    technical_gene_patterns: list[str] = TECHNICAL_GENE_PATTERNS,
    large_rank_gap_corr_thresh: float = 0.3,
    large_rank_gap_rank_thresh: int = 50,
) -> pd.DataFrame:
    """Flag top-predictor picks that likely aren't real biological signal.

    Expects the output of evaluate.raw_correlation_check (needs 'top_predictor_gene',
    'top_predictor_weak_raw_corr', 'cognate_raw_r', 'cognate_rank'). Three flags,
    combined into likely_artifact:
      top_predictor_is_technical -- top pick is hemoglobin/mitochondrial/ribosomal
                                     (reflects ambient RNA or cell state, not specific regulation)
      large_rank_gap              -- cognate gene has a strong raw correlation but
                                      ranked far from #1 anyway (model missed an easy case badly)
      top_predictor_weak_raw_corr -- carried over from raw_correlation_check
    """
    pattern = re.compile("|".join(technical_gene_patterns))
    out = check_df.copy()
    out["top_predictor_is_technical"] = flag_technical_genes(out["top_predictor_gene"], technical_gene_patterns)
    out["large_rank_gap"] = (
        (out["cognate_raw_r"].abs() > large_rank_gap_corr_thresh) & (out["cognate_rank"] > large_rank_gap_rank_thresh)
    )
    out["likely_artifact"] = (
        out["top_predictor_weak_raw_corr"] | out["top_predictor_is_technical"] | out["large_rank_gap"]
    )
    return out


def build_query_set(
    cognate_ranks: pd.DataFrame,
    trustworthy_core: pd.DataFrame,
    artifact_flags: pd.DataFrame,
) -> pd.DataFrame:
    """The gray-zone pairs worth a pathway lookup: not already trustworthy, not
    already flagged as artifacts, and the model didn't rank the cognate gene #1.
    """
    trustworthy_proteins = set(trustworthy_core.loc[trustworthy_core["trustworthy"], "protein"])
    artifact_proteins = set(artifact_flags.loc[artifact_flags["likely_artifact"], "protein"])
    return cognate_ranks[
        (~cognate_ranks["protein"].isin(trustworthy_proteins))
        & (~cognate_ranks["protein"].isin(artifact_proteins))
        & (cognate_ranks["cognate_rank"] > 1)
    ].copy()
