# Next Session

## Goal
Explainability analysis of the trained totalVI model — understand what the joint
latent space captured, validate it against known confounds, and use it to probe
the RNA-protein-severity relationship (not just predict severity).

## Context
- `nb03_totalvi.ipynb` is complete: totalVI trained on 62,420 QC-filtered cells
  (after fixing the missed-filter bug), 2,092 genes (163 matched coupling genes
  union top-2000 HVGs), 163 matched proteins, `batch_key='patient_id'`,
  `n_latent=20`. Train/validation ELBO curves both converged cleanly to ~940-960,
  no overfitting.
- Trained model saved to `MODELS_DIR / 'totalvi_covid'`.
- Full adata with latent + denoised outputs saved to
  `data/processed/covid_totalvi.h5ad` — has `obsm['X_totalVI']`,
  `obsm['protein_denoised']`, `layers['totalVI_normalized']`.
- Known confounds from nb02 that any downstream result must be checked against:
  `CV0198` (was 97.5% B_malignant, already excluded from the checkpoint) and
  `CV0178` (real patient-level coupling outlier in the Severe tier, not excluded
  — any severity signal here needs to be checked it isn't just this one patient).
- Objective reminder: DL here is to understand the RNA-protein relationship
  itself (severity is a probe/lens on that relationship, not the end goal).

## Step-by-step
1. **Latent sanity check** — UMAP on `X_totalVI`, colored separately by cell
   type, patient, and severity tier. Must show clean cell-type separation before
   trusting anything downstream.
2. **Per-latent-dimension correlation** — for each of the 20 dims, ANOVA/correlation
   against cell type, patient, and severity tier. Flag any dim dominated by
   patient identity rather than biology.
3. **Linear probe for severity, patient-held-out** — logistic regression from
   `z` to severity tier, `GroupKFold` by `patient_id` (never random CV — leaks
   patient identity). Compare against majority-class baseline and against the
   linear `deviation_score` baseline from `coupling.py` (unused so far).
4. **Reconstruction-based deviation score** — per-cell protein reconstruction
   error (raw vs `protein_denoised`) as a nonlinear analog of `deviation_score`.
   Map against severity tier and cell type, same structure as nb02 Steps 4/7.
5. **Pathway discovery** — correlate each protein's denoised expression against
   all 2,092 genes in the union (not just its matched 1:1 gene) across cells.
   Genes beyond the matched pair that still correlate are pathway-member
   candidates — the actual answer to "what else drives this protein."

Start with Step 1 (already confirmed, not yet written).

## What NOT to do
- Don't do patient-level severity classification (n=18, uninformative).
- Don't use random-split cross-validation anywhere severity is the target —
  always `GroupKFold` by `patient_id`.
- Don't restrict pathway discovery (Step 5) to the 163 matched genes — the
  whole point is finding genes outside that set.
- Don't treat Step 3's probe accuracy as the deliverable — it's a validation
  check on the relationship model, not the study's goal.
- If a severity trend shows up in Steps 3/4, don't report it before checking
  whether it's driven by one patient (repeat the `CV0178`-style diagnostic:
  per-patient breakdown before trusting a tier-level pattern).

## Open decision
New notebook (`nb04_explainability.ipynb`) vs. continuing in `nb03_totalvi.ipynb`
— recommend a new notebook since nb03 is already a full phase (data prep +
training). Confirm with user at start of next session.