# Next Session

## Goal
First real look at the downloaded data. Establish structure before deciding anything about
modeling. Mirrors the nb01 QC pattern from the multiomics-relationship-modeling project —
confirm what we actually have before writing any analysis code.

## Critical checkpoint — resolve this first
We have not yet confirmed that `covid_portal_210320_with_raw.h5ad` actually contains the
ADT (protein) layer, not just RNA. The filename and "with_raw" suggest raw RNA counts are
included, but nothing confirms protein data is in the same object. This is the single most
important thing to check before anything else — the whole project depends on RNA+protein
being paired here. If ADT is missing from this file, we need to find where it actually is
before proceeding (check `.obsm`, `.layers`, `.var['feature_types']`, or whether ADT was a
separate file we didn't download).

## Step-by-step

1. **Load `annotation_200112.csv` first** (small, fast). Confirm columns present:
   - patient/donor ID
   - severity / disease status (healthy, asymptomatic, mild, moderate, severe, critical)
   - cell type
   - any other covariates (batch, site, age, sex) — relevant later for confound-checking,
     which is the whole point of this project

2. **Load the h5ad in backed mode** — do not load fully into memory first:
   ```python
   adata = sc.read_h5ad('covid_portal_210320_with_raw.h5ad', backed='r')
   ```
   Inspect without loading data into memory:
   - `adata.shape` — confirm ~750,000 cells
   - `adata.obs.columns` — cross-check against annotation_200112.csv; are they the same
     info, or does the csv need merging in separately?
   - `adata.var` — **check `feature_types` or similar column for a Gene Expression vs.
     Antibody Capture split** — this answers the critical checkpoint above
   - `adata.obsm.keys()` and `adata.layers.keys()` — protein data sometimes lives here
     instead of `.var`

3. **If ADT/protein confirmed present**: note how it's structured (separate `.var` block,
   separate `.obsm` matrix, or something else) — this determines how nb01 needs to split
   RNA vs. protein into separate matrices, same as was done in nb02 previously.

   **If ADT is NOT present**: stop and report back before writing more code. Don't assume
   a workaround — this changes the project's viability with this specific dataset.

4. **Establish structure** (the actual nb01-equivalent QC output):
   - cell counts per severity tier and per patient
   - cell-type annotation completeness (any unlabeled cells?)
   - basic QC distributions (n_genes, total_counts, pct_mito) before deciding filtering
     thresholds

5. **Decide subsample strategy**: pick 3-4 patients per severity tier (healthy through
   critical) rather than random cell sampling, to keep patient-level biology intact. Save
   a subsampled `.h5ad` checkpoint so nb01 proper doesn't need to reload the full 7.19 GB
   object every session.

## What NOT to do this session
- Don't start building the coupling-comparison analysis yet — that's premature until the
  critical checkpoint above is resolved and structure is confirmed
- Don't decide on severity-tier groupings (binary healthy/disease vs. multi-tier) until
  cell counts per tier are known — some tiers may be too small to use
