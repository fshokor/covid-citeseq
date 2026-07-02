# Progress Log

## Session 1
- Repo initialized
- Dataset identified and download started (E-MTAB-10026)

## Session 2
- nb01 QC complete for COVID CITE-seq (E-MTAB-10026). Confirmed ADT present,
  cell types complete, severity column identified. Found Site batch effect via harmony
  PCA/UMAP check; resolved by restricting to single site (Cambridge). Built and saved
  subsampled checkpoint (69,681 cells, 18 patients, 6 tiers) for nb02.

  