# Next Session

## Goal
nb02-equivalent: RNA-protein coupling analysis across COVID severity stages, using
the subsampled checkpoint (`data/processed/covid_subsampled.h5ad`) — mirrors nb02
from multiomics-relationship-modeling (linear/statistical coupling on matched genes).

## Context
- Checkpoint: 69,681 cells, single site (Cambridge, to avoid the site batch effect
  found in nb01), 18 patients (3 per severity tier), 6 tiers: Healthy, Asymptomatic,
  Mild, Moderate, Severe, Critical. LPS and Non_covid excluded.
- `.X` in the source h5ad is scaled/log-normalized (has negative values) — use
  `layers['raw']` for any renormalization, not `.X`.
- 192 ADT proteins (`AB_` prefix). Will need a CD→gene mapping step like
  `cd_gene_mapping.py` from multiomics-relationship-modeling to match ADT to GEX genes.

## Step-by-step
1. Load `covid_subsampled.h5ad`, split GEX/ADT via `var['feature_types']`.
2. Map ADT protein names to gene symbols (reuse/adapt `cd_gene_mapping.py`).
3. Renormalize from `layers['raw']` (don't reuse the original `.X`).
4. Compute RNA-protein coupling (Pearson r per matched gene) within each severity tier.
5. Compare coupling strength/structure across tiers — this is the core question:
   does coupling break down as severity increases, and is it cell-type-specific?

## What NOT to do
- Don't reintroduce multi-site data without re-checking for batch effect.
- Don't assume the CD→gene mapping from the NeurIPS benchmark project transfers
  directly — ADT panel is different; verify overlap first.