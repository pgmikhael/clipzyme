import json, os
from typing import Union, NamedTuple
from pathlib import Path
import torch
import pickle
from p_tqdm import p_map
from functools import partial
import pandas as pd
from clipzyme.utils.wln_processing import get_bond_changes
from clipzyme.utils.pyg import from_mapped_smiles


class ScreeningResults(NamedTuple):
    predictions: pd.DataFrame
    protein_hiddens: torch.Tensor
    reaction_hiddens: torch.Tensor
    protein_ids: list
    reaction_ids: list


def load_hidden(hiddens_dir: Path, file: str) -> Union[torch.Tensor, None]:
    """
    Load hidden representation from a file

    Parameters
    ----------
    hiddens_dir : Path
        directory where hidden representations are saved
    file : str
        file name of hidden representation

    Returns
    -------
    Union[torch.Tensor, None]
        hidden representation tensor or None if file does not exist
    """
    try:
        return torch.load(hiddens_dir.joinpath(file))
    except FileNotFoundError:
        return


def collect_screening_results(config_path: str) -> ScreeningResults:
    """
    Collect screening results from a screening experiment

    Parameters
    ----------
    config_path : str
        path to screening experiment configuration file

    Returns
    -------
    ScreeningResults
        screening results, including predictions, protein hiddens, reaction hiddens, protein ids, and reaction ids
    """
    config = json.load(open(config_path, "r"))
    inference_dir = config["inference_dir"]
    save_hiddens = config["save_hiddens"]
    save_predictions = config["save_predictions"]
    checkpoint_path = Path(config["checkpoint_path"])
    dataset_path = Path(config["dataset_file_path"])
    # get directory where hiddens are saved
    hiddens_dir = os.path.join(inference_dir, checkpoint_path.stem, dataset_path.stem)

    predictions = None
    if save_predictions:
        # get path to predictions (screening scores)
        predictions_path = os.path.join(hiddens_dir, "predictions.csv")
        predictions = pd.read_csv(predictions_path)

    hiddens_dir = Path(hiddens_dir)
    assert hiddens_dir.is_dir(), f"{hiddens_dir} is not a directory"

    protein_hiddens, reaction_hiddens = None, None
    protein_ids, reaction_ids = None, None
    if save_hiddens:
        load_hidden_func = partial(load_hidden, path=hiddens_dir)
        protein_files = list(hiddens_dir.glob("sample_*.protein.pt"))
        reaction_files = list(hiddens_dir.glob("sample_*.reaction.pt"))

        load_hidden_func = partial(load_hidden, path=hiddens_dir)
        protein_hiddens = p_map(load_hidden_func, protein_files)
        reaction_hiddens = p_map(load_hidden_func, reaction_files)

        protein_ids = [f.stem.split("_")[-1] for f in protein_files]
        reaction_ids = [f.stem.split("_")[-1] for f in reaction_files]

        protein_ids = [u for u, h in zip(protein_ids, protein_hiddens) if h is not None]
        reaction_ids = [
            u for u, h in zip(reaction_ids, reaction_hiddens) if h is not None
        ]

        protein_hiddens = [h for h in protein_hiddens if h is not None]
        reaction_hiddens = [h for h in reaction_hiddens if h is not None]

        protein_hiddens = torch.stack(protein_hiddens)
        reaction_hiddens = torch.stack(reaction_hiddens)

    # collect
    output = ScreeningResults(
        predictions=predictions,
        protein_hiddens=protein_hiddens,
        reaction_hiddens=reaction_hiddens,
        protein_ids=protein_ids,
        reaction_ids=reaction_ids,
    )

    return output


def process_mapped_reaction(
    reaction: str, bond_changes=None, use_one_hot_mol_features: bool = False
):
    """
    Process a mapped reaction string into a PyG data object

    Parameters
    ----------
    reaction : str
        mapped reaction string
    bond_changes : list, optional
        list of bond change t (i,j,t) associated with edges (i,j), by default None
    use_one_hot_mol_features : bool, optional
        whether to use one-hot features for molecules , by default False

    Returns
    -------
    tuple of PyG data objects
        reactants and products as PyG data objects
    """
    reactants, products = reaction.split(">>")

    reactants, atom_map2new_index = from_mapped_smiles(
        reactants,
        encode_no_edge=True,
        use_one_hot_encoding=use_one_hot_mol_features,
    )
    products, _ = from_mapped_smiles(
        products,
        encode_no_edge=True,
        use_one_hot_encoding=use_one_hot_mol_features,
    )

    if bond_changes is None:
        bond_changes = get_bond_changes(reaction)

    bond_changes = [
        (atom_map2new_index[int(u)], atom_map2new_index[int(v)], btype)
        for u, v, btype in bond_changes
    ]
    bond_changes = [(min(x, y), max(x, y), t) for x, y, t in bond_changes]
    reactants.bond_changes = bond_changes
    return reactants, products
