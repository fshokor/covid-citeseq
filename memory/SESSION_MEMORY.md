# Session Memory — COVID CITE-seq Project

## Objective
Characterise the **RNA-protein relationship** in COVID CITE-seq data and how it
varies across **cell type**, **severity**, and **patient**. Severity is a *lens* on
the relationship, not a prediction target. This feeds the Genopole Shaker pitch
(confound-auditing / model-trust platform).

Dataset: Stephenson et al. 2021 (E-MTAB-10026), Cambridge site only, 6 severity
tiers, 18 patients (3/tier).

## Notebook status
- **nb01 (QC)** — complete. `covid_subsampled.h5ad`: 69,681 cells → 62,420 after
  the missed-filter fix. `.X` scaled; `layers['raw']`/`layers['counts']` are the
  raw source. Site confound resolved by restricting to Cambridge.
- **nb02 (RNA-protein coupling)** — complete. 163 matched pairs (from 192 ADT after
  isotype/zero-variance removal; DROP_GENES = PDPN, KDR). Global median Pearson
  r = 0.133. Severe-tier coupling spike traced to patient CV0178 (documented, not
  excluded). CV0198 (97.5% B_malignant) excluded, replaced with CV0201.
  `coupling.py` (`per_gene_pearson`, `coupling_score_by_celltype`, `deviation_score`).
- **nb03 (totalVI)** — complete. Trained on 62,420 cells, 2,092 genes (163 matched
  ∪ top-2000 HVGs), 163 proteins, `batch_key='patient_id'`, `n_latent=20`. Clean
  ELBO convergence. `covid_totalvi.h5ad` has `obsm['X_totalVI']`,
  `obsm['protein_denoised']`, `obsm['protein_expression']` (raw), `layers['totalVI_normalized']`,
  `layers['counts']`, `uns['protein_names']`.
- **nb04 (explainability)** — built this session, Steps 1–5 + a cell-type deviation
  cut (Step 6) run. Reads from `covid_totalvi.h5ad`; no model reload. One diagnostic
  still open: Step 5 detection-rate gate + raw-CLR cross-check (see NEXT_SESSION).

## nb04 results
- **Step 1 (UMAP):** clean cell-type separation; patient and severity fully mixed.
  Batch correction on patient_id confirmed working.
- **Step 2 (per-dim eta²):** cell type 0.05–0.73 per latent dim; patient ≤0.038;
  severity ≤0.014. No patient-dominated dims. Severity barely present in latent.
- **Step 3 (patient-held-out probe, GroupKFold):** latent 20d = 0.170 vs chance
  0.167; deviation 1d = 0.218 ± 0.124 (~1 SE off chance = noise); majority row 0.000
  is degenerate under GroupKFold, not a real baseline. No patient-generalizable
  severity signal. Uses real `coupling.deviation_score` (Ridge RNA→protein on 163
  matched pairs; lognorm RNA + CLR protein).
- **Step 4 (reconstruction deviation, raw vs denoised):** flat across tiers
  (0.59–0.64). Strongly cell-type-structured: HSC_MK 0.92, DC1 0.78, HSC_CD38neg
  0.75 highest — rare progenitor/DC populations = model-confidence readout (few
  cells → posterior pulls to prior), not biology. CV0178 sits MID-range within
  Severe (0.631, between 0.645/0.629) — the nb02 coupling spike does NOT recur in
  reconstruction space.
- **Step 5 (pathway discovery, 2092×163 corr):** CD86 → coherent myeloid activation
  module (PLAUR, CLEC7A, IL1RN, IFI30, IGSF6) — genuine. CD80 & CD274 → near-identical
  keratin/clone-ID lists (KRT80, IGLC4, CLDN14, Z98257.1, …) = low-expression
  artifact, flagged not trusted. Saved `results/tables/nb04_pathway_candidates.csv`.
- **Step 6 (deviation by cell type — the objective cut):** `coupling.deviation_score`
  (totalVI-free) grouped by `full_clustering`. Well-powered types (n≥500) span only
  ~0.40–0.43 → coupling magnitude roughly uniform across cell types. Spearman across
  types: linear vs recon 0.632, linear vs n_cells −0.396, recon vs n_cells −0.512 —
  BOTH measures carry an abundance dependence (opposite mechanisms), so neither is a
  clean confound-free cut. Within top-6 well-powered types × tier: no reliable severity
  trend.

## Findings against the objective
- **Severity:** relationship is **severity-invariant** — two independent methods
  (linear probe at chance; reconstruction flat) agree severity is not encoded in the
  RNA-protein relationship or the joint latent in any patient-generalizable way.
- **Patient:** minor axis (≤0.038 latent variance) but is where apparent "severity"
  actually lives — severity is a patient-level label, so it collapses to patient
  identity under patient-held-out CV (n=18). This is the confound, stated precisely.
- **Cell type — RESOLVED this session.** Cell type dominates the latent *geometry*
  (identity strongly encoded, eta² up to 0.73), but RNA-protein *decoupling magnitude*
  is roughly uniform across well-powered cell types: linear `deviation_score` spans only
  ~0.40–0.43 across all types with n≥500 (CD83_CD14_mono lowest/best-coupled at 0.403).
  The dramatic Step-4 reconstruction ranking (HSC_MK 0.92, DC1 0.78) was an abundance
  artifact — all n=1–42 cells. Key nuance: the linear measure does NOT fully escape
  abundance either — both track n_cells (recon ρ=−0.51 via rare-cell inflation; linear
  ρ=−0.40 via the global Ridge being dominated by majority lymphocytes). So we cannot
  claim large *biological* coupling differences by cell type. Correct statement:
  identity is cell-type-specific; decoupling magnitude is not.
  Within-cell-type × tier: no reliable severity trend (weak non-monotonic "Healthy
  slightly higher" ~0.02–0.04 = one-patient noise, not reportable without CV0178 check).
- **Relationship structure:** coupling is not 1:1 — protein abundance reflects a
  regulatory module, not only its cognate transcript (CD86 → myeloid program). Known
  limits: low-expression proteins give artifactual candidates; denoised-vs-denoised
  correlation is partly model-induced (both decode from the same 20-d latent).

## Pitch relevance
nb04 is a clean in-miniature demo of the confound-audit thesis: a model can look like
it "captures severity biology" while actually encoding cell-type identity and patient
identity. Framing for the trust platform: severity-invariant + severity-confounded-
with-patient + cell-type-*identity*-dominant (but coupling magnitude uniform across
types). Bonus meta-point: even the "correction" measure (`deviation_score`) had its own
abundance bias — a confound audit must audit its own diagnostics.

## Technical notes this session
- `coupling.deviation_score` wired in via importlib (`COUPLING_PATH`, BASE_PATH-relative),
  same mechanism as nb02.
- `layers['counts']` is sparse — needs `_dense()` (`.todense()` guard) before numpy
  normalization; `protein_expression`/`protein_denoised`/`totalVI_normalized` are dense.

## Key files
- `nb04_explainability.ipynb`
- `data/processed/covid_totalvi.h5ad`
- `results/tables/nb04_pathway_candidates.csv`
- `src/analysis/coupling.py`
- `results/tables/nb02_covid_adt_gene_mapping.csv`

## Known confounds to keep checking
- CV0178: real Severe-tier coupling outlier from nb02; not excluded. Did NOT recur in
  nb04 reconstruction — but re-check on any new tier-level result.
- CV0198: excluded (B_malignant), replaced with CV0201.
