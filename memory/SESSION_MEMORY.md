# Session Memory

## Current State
Project just initialized. COVID-19 CITE-seq dataset (E-MTAB-10026, Stephenson et al. 2021)
being downloaded to Google Drive. Not yet loaded or inspected.

## Dataset
- Source: ArrayExpress E-MTAB-10026
- Files: covid_portal_210320_with_raw.h5ad (7.19 GB), annotation_200112.csv (24.9 MB)
- ~750,000 PBMCs, 130 patients, healthy through critical COVID-19 severity
- CITE-seq: RNA + surface protein (ADT)

## COVID-19 CITE-seq project — updated
- nb01 QC complete: ADT confirmed present (24,737 GEX + 192 ADT via `var['feature_types']`),
  cell-type annotations complete (0 missing, 51 fine / 18 coarse labels), severity column
  is `Status_on_day_collection_summary` (LPS_* tiers excluded — separate experimental
  condition, not COVID severity).
- Batch-effect check: `Site` was the one covariate not well-mixed in UMAP/harmony PCA
  (patient, sex, age, smoker, severity, status all mixed fine). Resolved by restricting
  to a single site (Cambridge) rather than trusting harmony correction — sidesteps the
  confound instead of correcting for it.
- Subsample checkpoint saved: `data/processed/covid_subsampled.h5ad`, 69,681 cells,
  18 patients (3/tier), 6 severity tiers, Cambridge only.
- `.X` in source h5ad is scaled (negative values) — `layers['raw']` is the true raw
  counts layer to build from.