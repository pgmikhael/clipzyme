import argparse
from collections import defaultdict, Counter
from typing import List
from nox.utils.registry import register_object
from nox.utils.rdkit import smi_tokenizer
from nox.datasets.abstract import AbstractDataset
from tqdm import tqdm


class USPTO(AbstractDataset):
    def __init__():
        
        
    def __getitem__(self, index):
        try:
            return self.dataset[index]

        except Exception:
            warnings.warn("Could not load sample")
    
    # augment: permute; random
