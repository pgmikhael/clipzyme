# From: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
from typing import List, Tuple, Dict
import string
from Bio import SeqIO
import numpy as np
from scipy.spatial.distance import cdist


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]


def sample_msa(
    msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max"
) -> List[Tuple[str, str]]:
    """
    Sample sequences from MSA either randomly or those with highest similarity.
    First (query) sequence is always kept.

    Args:
        msa (List[Tuple[str, str]]): msa as (header, sequence) pairs
        num_seqs (int): number of sequences to sample
        mode (str, optional): sampling strategy. Defaults to "max".

    Raises:
        NotImplementedError

    Returns:
        List[Tuple[str, str]]: list as (header, sequence) pairs to keep
    """
    assert mode in ("max", "random")
    if len(msa) <= num_seqs:
        return msa

    if mode == "random":
        indices = np.random.choice(list(range(1, len(msa), 1)), num_seqs - 1)
        return [msa[0]] + [msa[idx] for idx in indices]
    elif mode == "max":
        # integers for seq
        array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

        optfunc = np.argmax
        all_indices = np.arange(len(msa))  # number of sequences in msa
        indices = [0]
        pairwise_distances = np.zeros((0, len(msa)))
        for _ in range(num_seqs - 1):
            dist = cdist(array[indices[-1:]], array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return [msa[idx] for idx in indices]
    else:
        raise NotImplementedError
