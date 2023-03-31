AA_TO_SMILES = {
    "A": "C[C@@H](C(=O)O)N",  # Alanine
    "R": "C(C[C@@H](C(=O)O)N)CN=C(N)N",  # Arginine
    "N": "C([C@@H](C(=O)O)N)C(=O)N",  # Asparagine
    "D": "C([C@@H](C(=O)O)N)C(=O)O",  # Aspartic acid
    "B": "X",  # Aspartate or Asparagine
    "C": "C([C@@H](C(=O)O)N)S",  # Cysteine
    "E": "C(CC(=O)O)[C@@H](C(=O)O)N",  # Glutamic acid
    "Q": "C(CC(=O)N)[C@@H](C(=O)O)N",  # Glutamine
    "Z": "X",  # Glutamate or Glutamine
    "G": "C(C(=O)O)N",  # Glycine
    "H": "C([C@@H](C(=O)O)N)C(=O)N",  # Histidine
    "I": "CC[C@H](C)[C@@H](C(=O)O)N",  # Isoleucine
    "L": "CC(C)C[C@@H](C(=O)O)N",  # Leucine
    "K": "C(CCN)C[C@@H](C(=O)O)N",  # Lysine
    "M": "CSCC[C@@H](C(=O)O)N",  # Methionine
    "F": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",  # Phenylalanine
    "P": "C1C[C@H](NC1)C(=O)O",  # Proline
    "S": "C([C@@H](C(=O)O)N)O",  # Serine
    "T": "C[C@H]([C@@H](C(=O)O)N)O",  # Threonine
    "W": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",  # Tryptophan
    "Y": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",  # Tyrosine
    "V": "CC(C)[C@@H](C(=O)O)N",  # Valine
    "X": "X",
    "U": "C([C@@H](C(=O)O)N)[SeH]",  # Selenocysteine
}

AA_TO_TRIPLET = {
    "A": "ala",  # Alanine
    "R": "arg",  # Arginine
    "N": "asn",  # Asparagine
    "D": "asp",  # Aspartic acid
    "B": ["asp", "asn"],  # Aspartate or Asparagine
    "C": "cys",  # Cysteine
    "E": "glu",  # Glutamic acid
    "Q": "gln",  # Glutamine
    "Z": ["glu", "gln"],  # Glutamate or Glutamine
    "G": "gly",  # Glycine
    "H": "his",  # Histidine
    "I": "ile",  # Isoleucine
    "L": "leu",  # Leucine
    "K": "lys",  # Lysine
    "M": "met",  # Methionine
    "F": "phe",  # Phenylalanine
    "P": "pro",  # Proline
    "S": "ser",  # Serine
    "T": "thr",  # Threonine
    "W": "trp",  # Tryptophan
    "Y": "tyr",  # Tyrosine
    "V": "val",  # Valine
    "X": "x",
    "U": "sec",  # Selenocysteine
}
