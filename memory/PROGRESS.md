# Progress Log

## Session 1
- Repo initialized
- Dataset identified and download started (E-MTAB-10026)

## Session 2
- nb01 QC complete for COVID CITE-seq (E-MTAB-10026). Confirmed ADT present,
  cell types complete, severity column identified. Found Site batch effect via harmony
  PCA/UMAP check; resolved by restricting to single site (Cambridge). Built and saved
  subsampled checkpoint (69,681 cells, 18 patients, 6 tiers) for nb02.

## 2026-07-04 — nb04 explainability (Steps 1–5)
 
Built `nb04_explainability.ipynb` (5 steps, reads `covid_totalvi.h5ad`, no model
reload). Wired real `coupling.deviation_score` (importlib, BASE_PATH-relative) into
the severity probe; fixed sparse-`counts` densification (`_dense` helper).
 
Results:
- Step 1 UMAP: clean cell-type separation, patient/severity mixed — batch correction OK.
- Step 2 eta²: cell type 0.05–0.73/dim, patient ≤0.038, severity ≤0.014. No
  patient-dominated dims.
- Step 3 patient-held-out probe: latent 20d 0.170 vs chance 0.167; deviation 1d
  0.218±0.124 (noise); no patient-generalizable severity signal.
- Step 4 reconstruction deviation: flat by tier (0.59–0.64); cell-type-dominated
  (HSC_MK 0.92, DC1 0.78 = rare-population confidence readout); CV0178 mid-range in
  Severe — nb02 coupling spike does NOT recur.
- Step 5 pathway discovery: CD86 → genuine myeloid module (PLAUR, CLEC7A, IL1RN,
  IFI30, IGSF6); CD80/CD274 → low-expression keratin/clone-ID artifact. Saved
  `nb04_pathway_candidates.csv`.
- Step 6 deviation-by-cell-type (objective cut): coupling *magnitude* roughly uniform
  across well-powered cell types (~0.40–0.43); Step-4 rare-population ranking was an
  abundance artifact (n=1–42). Both measures track n_cells (recon ρ=−0.51, linear
  ρ=−0.40, opposite mechanisms) — neither is confound-free. Within-type × tier: no
  reliable severity trend.
Conclusion: RNA-protein relationship is severity-invariant; apparent severity is
confounded with patient identity; cell type dominates latent *identity* but coupling
*magnitude* is uniform across types; coupling is not 1:1 (protein reflects regulatory
modules). Meta-finding: even the correction diagnostic (deviation_score) has its own
abundance bias. Strong, self-aware confound-audit framing for the pitch.
 
Open: Step 5 detection-rate gate + raw-CLR cross-check (CD80/CD274 vs CD86); then
pitch-readiness write-up.