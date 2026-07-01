# Next Session

1. Confirm annotation_200112.csv columns (patient ID, severity, cell type)
2. Load h5ad in backed mode (`sc.read_h5ad(..., backed='r')`), cross-check against annotation
3. Subsample by patient (3-4 per severity tier) and save a working checkpoint
4. Write nb01: QC pass on the subsample
