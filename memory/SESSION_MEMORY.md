# Session Memory

## Current State
Project just initialized. COVID-19 CITE-seq dataset (E-MTAB-10026, Stephenson et al. 2021)
being downloaded to Google Drive. Not yet loaded or inspected.

## Dataset
- Source: ArrayExpress E-MTAB-10026
- Files: covid_portal_210320_with_raw.h5ad (7.19 GB), annotation_200112.csv (24.9 MB)
- ~750,000 PBMCs, 130 patients, healthy through critical COVID-19 severity
- CITE-seq: RNA + surface protein (ADT)

## Next step
Load annotation_200112.csv first to confirm column structure (patient ID, severity,
cell type) before loading the full h5ad in backed mode.
