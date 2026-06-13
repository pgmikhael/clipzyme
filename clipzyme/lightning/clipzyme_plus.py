"""CLIPZyme+ : cofactor prediction on top of CLIPZyme protein embeddings.

``CLIPZymePlus`` is a standalone, inference-only extension of :class:`CLIPZyme`.
It reuses CLIPZyme's protein encoder to produce a 1280-dim embedding per enzyme,
then runs an ensemble of small MLP classifiers to predict the enzyme's cofactor.

The cofactor ensemble ships as a single self-contained checkpoint file (download
from Zenodo, then point to it). It is used exactly like ``CLIPZyme`` with a
``ReactionDataset``::

    from torch.utils.data import DataLoader
    from clipzyme import CLIPZymePlus, ReactionDataset
    from clipzyme.utils.loading import ignore_None_collate

    loader = DataLoader(
        ReactionDataset(
            dataset_file_path="files/new_data.csv",
            esm_dir="/path/to/esm2_dir",
            use_as_protein_encoder=True,
        ),
        batch_size=1,
        collate_fn=ignore_None_collate,
    )

    model = CLIPZymePlus(
        checkpoint_path="files/clipzyme_model.ckpt",
        cofactor_checkpoint_path="files/clipzyme_plus_cofactor_ensemble.pt",
    )
    model = model.eval()

    for batch in loader:
        output = model(batch)
        print(output.sample_ids, output.predicted_cofactor, output.predicted_probability)

You can also predict directly from structure files or precomputed embeddings::

    output = model.predict_from_structures(["1a0s.cif"], esm_dir="/path/to/esm2_dir")
    output = model.predict_from_embeddings(screen_hiddens)  # (N, 1280) array/tensor
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from clipzyme.lightning.clipzyme import CLIPZyme

# ---------------------------------------------------------------------------
# Cofactor ensemble checkpoint.
#
# The ensemble is distributed as a single ``.pt`` file (see the module docstring
# and ``_load_cofactor_ensemble`` for the format). Download it from Zenodo and
# pass its path via ``cofactor_checkpoint_path``. ``COFACTOR_CHECKPOINT_URL`` may
# be set to enable automatic download when the file is missing.
# ---------------------------------------------------------------------------
DEFAULT_COFACTOR_CHECKPOINT_PATH = "files/clipzyme_plus_cofactor_ensemble.pt"
COFACTOR_CHECKPOINT_URL: Optional[str] = (
    "https://zenodo.org/records/20673359/files/clipzyme_plus_cofactor_ensemble.pt?download=1"
)
COFACTOR_CHECKPOINT_FORMAT_VERSION = 1


class CofactorOutput(NamedTuple):
    """Per-batch cofactor predictions.

    Attributes
    ----------
    sample_ids: List[str]
        Sample/protein ids, parallel to the batch dimension.
    predicted_cofactor: List[str]
        Top-1 predicted cofactor class name per sample.
    predicted_probability: List[float]
        Ensemble-mean probability of the top-1 class per sample.
    probabilities: np.ndarray
        ``(B, n_classes)`` ensemble-mean probability per class.
    probabilities_std: np.ndarray
        ``(B, n_classes)`` ensemble standard deviation per class.
    run_support: np.ndarray
        ``(B, n_classes)`` number of ensemble members that scored each class.
    class_names: List[str]
        Class names, indexing the columns of the probability arrays.
    protein_hiddens: torch.Tensor
        ``(B, 1280)`` CLIPZyme protein embeddings used for the prediction.
    """

    sample_ids: List[str]
    predicted_cofactor: List[str]
    predicted_probability: List[float]
    probabilities: np.ndarray
    probabilities_std: np.ndarray
    run_support: np.ndarray
    class_names: List[str]
    protein_hiddens: torch.Tensor


class _CofactorMLP(nn.Module):
    """Inference replica of the trained cofactor MLP.

    The architecture (Linear / BatchNorm1d / ReLU / Dropout blocks) must match
    the training definition exactly so that ``net.<i>.*`` state-dict keys align.
    """

    def __init__(
        self, input_dim: int, num_classes: int, hidden_dims: Sequence[int], dropout: float
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _EnsembleMember:
    seed: Optional[int]
    model_name: str
    class_names: List[str]
    model: _CofactorMLP


def _download_cofactor_checkpoint(url: str, dest: Union[str, Path]) -> str:
    """Download the cofactor ensemble checkpoint to ``dest``."""
    import wget

    dest = Path(dest)
    os.makedirs(dest.parent, exist_ok=True)
    print(f"Downloading cofactor ensemble checkpoint to {dest}")
    wget.download(url, out=str(dest))
    print(f"\nCofactor ensemble checkpoint saved to {dest}")
    return str(dest)


def _build_member(member: dict, device: torch.device) -> _EnsembleMember:
    """Instantiate one ensemble member from its serialized record."""
    class_names = [str(c) for c in member["class_names"]]
    state_dict = member["state_dict"]
    input_dim = int(member.get("input_dim") or state_dict["net.0.weight"].shape[1])
    model = _CofactorMLP(
        input_dim=input_dim,
        num_classes=len(class_names),
        hidden_dims=[int(x) for x in member["hidden_dims"]],
        dropout=float(member["dropout"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return _EnsembleMember(
        seed=member.get("seed"),
        model_name=str(member.get("model_name", "cofactor_mlp")),
        class_names=class_names,
        model=model,
    )


class CLIPZymePlus(CLIPZyme):
    def __init__(
        self,
        args: argparse.Namespace = None,
        checkpoint_path: str = None,
        device: Optional[str] = None,
        cofactor_checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize a trained CLIPZyme+ model for cofactor inference.

        Parameters
        ----------
        args: argparse.Namespace
            Arguments from command line. If provided, ``args.checkpoint_path`` is
            used and protein-encoder mode is forced on. ``args.cofactor_checkpoint_path``
            is honored if set.
        checkpoint_path: str
            Path to a CLIPZyme checkpoint.
        device: str
            If provided, run inference on this device. Defaults to GPU if available.
        cofactor_checkpoint_path: str or Path, optional
            Path to the cofactor MLP ensemble checkpoint (a single ``.pt`` file
            downloaded from Zenodo). Defaults to ``args.cofactor_checkpoint_path``
            if given, else ``"files/clipzyme_plus_cofactor_ensemble.pt"``. If the
            file is missing and ``COFACTOR_CHECKPOINT_URL`` is set, it is downloaded.
        """
        if args is None:
            args = argparse.Namespace(
                checkpoint_path=checkpoint_path,
                use_as_protein_encoder=True,
                use_as_reaction_encoder=False,
                save_hiddens=False,
                save_predictions=False,
            )
        else:
            # CLIPZyme+ uses CLIPZyme purely as a protein encoder.
            args.use_as_protein_encoder = True
            args.use_as_reaction_encoder = False

        if cofactor_checkpoint_path is None:
            cofactor_checkpoint_path = getattr(
                args, "cofactor_checkpoint_path", None
            ) or DEFAULT_COFACTOR_CHECKPOINT_PATH

        super().__init__(args=args, checkpoint_path=checkpoint_path, device=device)

        # Device the underlying CLIPZyme model lives on; mirror it for the MLPs.
        try:
            self._infer_device = next(self.model.parameters()).device
        except StopIteration:
            self._infer_device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

        self._load_cofactor_ensemble(cofactor_checkpoint_path)

    # ------------------------------------------------------------------ setup
    def _load_cofactor_ensemble(
        self, cofactor_checkpoint_path: Union[str, Path]
    ) -> None:
        """Load the cofactor MLP ensemble from a single checkpoint file.

        The checkpoint is a dict with::

            {
              "format_version": 1,
              "model_name": str,
              "class_names": [str, ...],          # union across members
              "members": [
                  {"seed", "model_name", "input_dim", "hidden_dims",
                   "dropout", "class_names", "state_dict"},
                  ...
              ],
            }
        """
        path = Path(cofactor_checkpoint_path)
        if not path.exists():
            if COFACTOR_CHECKPOINT_URL:
                path = Path(
                    _download_cofactor_checkpoint(COFACTOR_CHECKPOINT_URL, path)
                )
            else:
                raise FileNotFoundError(
                    f"Cofactor ensemble checkpoint not found at {path}. "
                    "Download it from Zenodo and pass its path via "
                    "`cofactor_checkpoint_path`."
                )

        bundle = torch.load(path, map_location=self._infer_device)
        if not isinstance(bundle, dict) or "members" not in bundle:
            raise ValueError(
                f"Unexpected cofactor checkpoint format in {path}; expected a dict "
                "with a 'members' list."
            )
        version = int(bundle.get("format_version", COFACTOR_CHECKPOINT_FORMAT_VERSION))
        if version != COFACTOR_CHECKPOINT_FORMAT_VERSION:
            warnings.warn(
                f"Cofactor checkpoint format_version={version} differs from "
                f"expected {COFACTOR_CHECKPOINT_FORMAT_VERSION}; attempting to load anyway."
            )

        self.cofactor_members: List[_EnsembleMember] = [
            _build_member(m, device=self._infer_device) for m in bundle["members"]
        ]
        if not self.cofactor_members:
            raise ValueError(f"Cofactor checkpoint {path} contains no ensemble members.")

        self.cofactor_model_name: str = str(
            bundle.get("model_name", self.cofactor_members[0].model_name)
        )

        # Class index uses the union across members; members may cover a subset.
        merged: List[str] = list(bundle.get("class_names", []))
        for member in self.cofactor_members:
            for name in member.class_names:
                if name not in merged:
                    merged.append(name)
        self.cofactor_class_names: List[str] = merged
        self._class_to_idx = {name: i for i, name in enumerate(merged)}

    # --------------------------------------------------------------- inference
    def _ensemble_probs(self, embeddings: torch.Tensor):
        """Average softmax probabilities over the MLP ensemble.

        Parameters
        ----------
        embeddings: torch.Tensor
            ``(B, D)`` protein embeddings (CLIPZyme protein hiddens).

        Returns
        -------
        (mean, std, support): tuple of np.ndarray
            Each of shape ``(B, n_classes)``; ``support`` counts the ensemble
            members that scored each class.
        """
        embeddings = embeddings.to(self._infer_device)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        batch_size = embeddings.shape[0]
        n_classes = len(self.cofactor_class_names)

        probs = np.full((len(self.cofactor_members), batch_size, n_classes), np.nan)
        with torch.no_grad():
            for run_idx, member in enumerate(self.cofactor_members):
                logits = member.model(embeddings)
                member_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for local_idx, name in enumerate(member.class_names):
                    probs[run_idx, :, self._class_to_idx[name]] = member_probs[:, local_idx]

        support = np.sum(~np.isnan(probs), axis=0).astype(np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nan_to_num(np.nanmean(probs, axis=0))
            std = np.nan_to_num(np.nanstd(probs, axis=0))
        return mean, std, support

    def _build_output(
        self, embeddings: torch.Tensor, sample_ids: List[str]
    ) -> CofactorOutput:
        mean, std, support = self._ensemble_probs(embeddings)
        winners = np.argmax(mean, axis=1)
        class_names = self.cofactor_class_names
        return CofactorOutput(
            sample_ids=list(sample_ids),
            predicted_cofactor=[class_names[i] for i in winners],
            predicted_probability=[float(mean[row, i]) for row, i in enumerate(winners)],
            probabilities=mean,
            probabilities_std=std,
            run_support=support,
            class_names=class_names,
            protein_hiddens=embeddings.detach().cpu(),
        )

    def forward(self, batch, batch_idx: int = 0) -> CofactorOutput:
        """Predict cofactors for a ``ReactionDataset`` batch.

        Parameters
        ----------
        batch : dict
            A batch from ``ReactionDataset`` (must contain protein graphs).
        batch_idx : int, optional
            Unused; kept for signature compatibility with Lightning.

        Returns
        -------
        CofactorOutput
            Per-sample cofactor predictions and class probabilities.
        """
        embeddings = self.extract_protein_features(batch)
        sample_ids = batch.get("sample_id") or batch.get("protein_id")
        if sample_ids is None:
            sample_ids = [str(i) for i in range(embeddings.shape[0])]
        return self._build_output(embeddings, list(sample_ids))

    def predict_from_structures(
        self,
        cif_path: Union[str, List[str]],
        esm_dir: str,
        sample_ids: Optional[List[str]] = None,
    ) -> CofactorOutput:
        """Predict cofactors directly from one or more structure (CIF) files.

        Parameters
        ----------
        cif_path : str or list of str
            Path(s) to CIF file(s).
        esm_dir : str
            Path to the ESM-2 (``esm2_t33_650M_UR50D``) directory.
        sample_ids : list of str, optional
            Ids parallel to ``cif_path``; defaults to the file stems.

        Returns
        -------
        CofactorOutput
        """
        paths = [cif_path] if isinstance(cif_path, str) else list(cif_path)
        embeddings = self.extract_protein_features(cif_path=paths, esm_dir=esm_dir)
        if sample_ids is None:
            sample_ids = [Path(p).stem for p in paths]
        return self._build_output(embeddings, sample_ids)

    def predict_from_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        sample_ids: Optional[List[str]] = None,
    ) -> CofactorOutput:
        """Predict cofactors from precomputed CLIPZyme protein embeddings.

        Useful for re-scoring the CLIPZyme screening set (its ``hiddens`` are the
        same normalized 1280-dim protein features this ensemble was trained on).

        Parameters
        ----------
        embeddings : np.ndarray or torch.Tensor
            ``(N, 1280)`` (or ``(1280,)``) protein embeddings.
        sample_ids : list of str, optional
            Ids parallel to ``embeddings``; defaults to ``["0", "1", ...]``.

        Returns
        -------
        CofactorOutput
        """
        if isinstance(embeddings, np.ndarray):
            tensor = torch.from_numpy(embeddings.astype(np.float32))
        else:
            tensor = embeddings.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if sample_ids is None:
            sample_ids = [str(i) for i in range(tensor.shape[0])]
        return self._build_output(tensor, sample_ids)

    def top_k(
        self, output: CofactorOutput, k: int = 5, exclude_no_cofactor: bool = False
    ) -> List[List[dict]]:
        """Rank predictions per sample.

        Parameters
        ----------
        output : CofactorOutput
            A result from :meth:`forward` / ``predict_*``.
        k : int
            Number of top classes to return per sample.
        exclude_no_cofactor : bool
            Drop the ``"no_cofactor"`` class from the ranking.

        Returns
        -------
        list of list of dict
            For each sample, a list of ``{rank, class_name, probability_mean,
            probability_std, run_support_count}`` dicts, highest probability first.
        """
        class_names = output.class_names
        results: List[List[dict]] = []
        for row in range(output.probabilities.shape[0]):
            mean = output.probabilities[row]
            std = output.probabilities_std[row]
            support = output.run_support[row]
            ranking = np.argsort(-mean).tolist()
            if exclude_no_cofactor:
                ranking = [i for i in ranking if class_names[i] != "no_cofactor"]
            top = max(1, min(int(k), len(ranking)))
            results.append(
                [
                    {
                        "rank": rank,
                        "class_index": int(idx),
                        "class_name": class_names[idx],
                        "probability_mean": float(mean[idx]),
                        "probability_std": float(std[idx]),
                        "run_support_count": int(support[idx]),
                    }
                    for rank, idx in enumerate(ranking[:top], start=1)
                ]
            )
        return results
