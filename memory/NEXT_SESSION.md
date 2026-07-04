# Next Session

## Goal
One diagnostic left on nb04 (Step 5 pathway trust gate), then the pitch-readiness
call. Short session. The cell-type question is already answered — do NOT reopen it.

## Context
- nb04 is built and run: Steps 1–5 + Step 6 (deviation by cell type). See
  SESSION_MEMORY.md for full results.
- Settled findings: relationship is severity-invariant; apparent severity is
  confounded with patient identity (collapses under patient-held-out CV, n=18);
  cell type dominates latent *identity* but RNA-protein *decoupling magnitude* is
  roughly uniform across well-powered cell types (~0.40–0.43).
- Important caveat carried forward: BOTH deviation measures have an abundance
  dependence — reconstruction inflates rare types (posterior→prior), the global-Ridge
  linear score deflates abundant types (fit dominated by majority lymphocytes). Neither
  is a fully confound-free coupling measure. Don't treat either ranking as pure biology.

## Step-by-step
1. **Step 5 trust gate (the only open analysis).** Add `mean_denoised` and
   `detection_rate` (fraction of cells with raw ADT > 0) per protein to
   `nb04_pathway_candidates.csv`. For low-detection proteins (CD80, CD274), recompute
   top candidates from **raw CLR protein vs log-norm RNA** (bypass totalVI). Expect:
   CD86's myeloid module (PLAUR, CLEC7A, IL1RN, IFI30, IGSF6) survives raw;
   CD80/CD274 keratin/clone-ID lists collapse. That confirms both the genuine hit and
   the low-expression artifact, and gives a defensible per-protein trust filter.
2. **Pitch-readiness call.** nb04 now supports the confound-audit thesis end to end.
   Write the one-paragraph framing: model appears to encode severity biology; audit
   shows it encodes cell-type + patient identity; severity is invariant; and even the
   correction diagnostics carry their own confound (abundance) — the platform must
   audit its own tools.

## What NOT to do
- Don't reopen the cell-type coupling question — answered (uniform magnitude,
  identity ≠ decoupling). Don't re-run Step 6.
- Don't reopen severity prediction — settled negative. Any new tier result gets the
  CV0178 per-patient breakdown first.
- Don't trust any pathway list for a protein until it passes the detection-rate gate.
- Don't describe `deviation_score` as a clean/confound-free coupling measure — it has
  its own abundance bias (Step 6).

## After this
- Return to NeurIPS 2021 project (GSE194122; nb04 CITE-seq VAE was mid-session,
  pending RNA→protein prediction lit review), OR
- Start TCGA-BRCA (UCSC Xena / LinkedOmics) as the next dataset.
