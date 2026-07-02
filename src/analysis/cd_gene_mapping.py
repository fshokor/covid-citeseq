"""
cd_gene_mapping.py
-------------------
CD nomenclature -> HGNC gene symbol mapping for matching ADT (CITE-seq protein)
names to GEX (RNA) gene symbols. Built from the unmatched list after the naive
regex-strip approach (which only matches cases where CD name == gene symbol).

Ambiguous cases are deliberately EXCLUDED (mapped to None) rather than guessed:
- TCR, HLA: multi-gene families, no single correct gene symbol
- CD45RA / CD45RO: both are PTPRC isoforms (alternative splicing, not separate
  genes) -- mapping both to PTPRC would let them "double count" against the
  same RNA signal, which is misleading for a coupling analysis.
- CD158 family (CD158, CD158b, CD158e1): different KIR genes per clone/epitope,
  not reliably resolvable without the antibody clone ID.
"""

CD_TO_GENE = {
    # Clear 1:1 CD -> gene symbol mappings
    "CD103": "ITGAE",
    "CD105": "ENG",
    "CD107a": "LAMP1",
    "CD112": "NECTIN2",
    "CD115": "CSF1R",
    "CD119": "IFNGR1",
    "CD11a": "ITGAL",
    "CD11b": "ITGAM",
    "CD11c": "ITGAX",
    "CD122": "IL2RB",
    "CD123": "IL3RA",
    "CD124": "IL4R",
    "CD127": "IL7R",
    "CD13": "ANPEP",
    "CD134": "TNFRSF4",
    "CD137": "TNFRSF9",
    "CD141": "THBD",
    "CD142": "F3",
    "CD146": "MCAM",
    "CD152": "CTLA4",
    "CD154": "CD40LG",
    "CD155": "PVR",
    "CD16": "FCGR3A",
    "CD161": "KLRB1",
    "CD162": "SELPLG",
    "CD169": "SIGLEC1",
    "CD172a": "SIRPA",
    "CD18": "ITGB2",
    "CD185": "CXCR5",
    "CD192": "CCR2",
    "CD194": "CCR4",
    "CD195": "CCR5",
    "CD196": "CCR6",
    "CD1c": "CD1C",
    "CD1d": "CD1D",
    "CD20": "MS4A1",
    "CD21": "CR2",
    "CD223": "LAG3",
    "CD224": "GGT1",
    "CD23": "FCER2",
    "CD25": "IL2RA",
    "CD26": "DPP4",
    "CD268": "TNFRSF13C",
    "CD270": "TNFRSF14",
    "CD272": "BTLA",
    "CD274": "CD274",   # PD-L1, gene symbol matches
    "CD278": "ICOS",
    "CD279": "PDCD1",
    "CD29": "ITGB1",
    "CD3": "CD3D",       # CD3 complex -- using CD3D as representative chain
    "CD303": "CLEC4C",
    "CD304": "NRP1",
    "CD31": "PECAM1",
    "CD314": "KLRK1",
    "CD319": "SLAMF7",
    "CD32": "FCGR2A",
    "CD328": "SIGLEC7",
    "CD335": "NCR1",
    "CD35": "CR1",
    "CD352": "SLAMF6",
    "CD39": "ENTPD1",
    "CD41": "ITGA2B",
    "CD42b": "GP1BA",
    "CD45": "PTPRC",
    "CD49a": "ITGA1",
    "CD49b": "ITGA2",
    "CD49d": "ITGA4",
    "CD49f": "ITGA6",
    "CD54": "ICAM1",
    "CD56": "NCAM1",
    "CD57": "B3GAT1",
    "CD62L": "SELL",
    "CD62P": "SELP",
    "CD64": "FCGR1A",
    "CD71": "TFRC",
    "CD73": "NT5E",
    "CD79b": "CD79B",
    "CD8": "CD8A",
    "CD85j": "LILRB1",
    "CD88": "C5AR1",
    "CD94": "KLRD1",
    "CD95": "FAS",
    "FceRIa": "FCER1A",
    "IgD": "IGHD",
    "IgM": "IGHM",
    "LOX": "OLR1",         # LOX-1 / OLR1, not the lysyl oxidase LOX gene
    "LOX-1": "OLR1",       # alias used in this dataset's ADT naming
    "Podoplanin": "PDPN",
    "TCRVa7.2": "TRAV1-2",
    "TCRVd2": "TRDV2",
    "integrinB7": "ITGB7",

    # Deliberately excluded -- ambiguous, do not guess
    "CD45RA": None,   # PTPRC isoform, not a separate gene
    "CD45RO": None,   # PTPRC isoform, not a separate gene
    "CD158": None,    # KIR family, clone-dependent
    "CD158b": None,   # KIR family, clone-dependent
    "CD158e1": None,  # KIR family, clone-dependent
    "TCR": None,      # multi-gene complex
    "HLA": None,      # multi-gene family (HLA-A/B/C/DR/DQ...)
}
