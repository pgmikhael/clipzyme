from typing import Optional, List, NamedTuple
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from pathlib import Path
import os
import torch
from rich import print
import pytorch_lightning as pl
import argparse
from clipzyme.models.protmol import EnzymeReactionCLIP

CHECKPOINT_URL = "https://github.com/pgmikhael/clipzyme/releases/download/v1.0.0/clipzyme_checkpoints.zip"  # TODO: Update this


def download_and_extract(remote_model_url: str, local_model_dir) -> List[str]:
    resp = urlopen(remote_model_url)
    os.makedirs(local_model_dir, exist_ok=True)
    with ZipFile(BytesIO(resp.read())) as zip_file:
        all_files_and_dirs = zip_file.namelist()
        zip_file.extractall(local_model_dir)
    assert len(all_files_and_dirs) == 1, "Expected only one file in the zip"
    return all_files_and_dirs[0]


def download_model(checkpoint_path) -> str:
    """
    Download trained clipzyme model.

    Parameters
    ----------
        checkpoint_path (str): path to where model should be located

    Returns
    -------
        str: local path to model
    """
    # Create cache folder if not exists
    cache = os.path.dirname(checkpoint_path)
    os.makedirs(cache, exist_ok=True)

    # Download model
    print(f"Downloading model to {cache}")
    filename = download_and_extract(CHECKPOINT_URL, cache)
    model_path = os.path.join(cache, filename)
    print(f"Model saved in {model_path}")
    return model_path


class CLIPZymeOutput(NamedTuple):
    scores: torch.Tensor
    protein_hiddens: torch.Tensor
    reaction_hiddens: torch.Tensor


class CLIPZyme(pl.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace = None,
        checkpoint_path: str = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize a trained clipzyme model for inference.

        Parameters
        ----------
        args: argparse.Namespace
            Arguments from command line.
        checkpoint_path: str
            Path to a clipzyme checkpoint.
        device: str
            If provided, will run inference using this device.
            By default uses GPU, if available.
        """
        if args is not None:
            checkpoint_path = args.checkpoint_path
        # Check if path exists
        if (checkpoint_path is None) or (not os.path.exists(checkpoint_path)):
            try:
                # download model
                print(f"Model checkpoint not found at {checkpoint_path}")
                checkpoint_path = download_model(checkpoint_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Model checkpoint not found at {checkpoint_path} and failed to download model. {e}"
                )

        # Set device
        if device is not None:
            self.device = device
        elif self.device is not None:  # set in lightning module
            pass
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = self.load_model(checkpoint_path)

        # prep directories
        self.prepare_directories(args)

    def prepare_directories(self, args: argparse.Namespace) -> None:
        """
        Prepare directories for saving hiddens and predictions.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments from command line.
        """
        self.save_hiddens = getattr(args, "save_hiddens", False)
        self.save_predictions = getattr(args, "save_predictions", False)
        if self.save_hiddens or self.save_predictions:
            checkpoint_path = Path(args.checkpoint_path)
            dataset_path = Path(args.dataset_file_path)
            hiddens_dir = os.path.join(
                args.inference_dir, checkpoint_path.stem, dataset_path.stem
            )
            os.makedirs(hiddens_dir, exist_ok=True)
            self.hiddens_dir = Path(hiddens_dir)

        if self.save_predictions:
            self.predictions_dir = os.path.join(hiddens_dir, "predictions")
            os.makedirs(self.predictions_dir, exist_ok=True)
            self.predictions_dir = Path(self.predictions_dir)
            self.predictions_path = os.path.join(self.hiddens_dir, "predictions.csv")

    def load_model(self, path: str) -> EnzymeReactionCLIP:
        """
        Load model from path.

        Parameters
        ----------
        path : str
            Path to a clipzyme model checkpoint.

        Returns
        -------
        model
            Pretrained clipzyme model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu")
        args = checkpoint["args"]
        model = EnzymeReactionCLIP(args)

        # Remove model from param names
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)  # type: ignore
        if self.device == "cuda":
            model.to("cuda")

        # Set eval
        model.eval()
        print(f"Loaded model from {path}")
        return model

    def forward(
        self,
        batch,
        batch_idx: int = 0,
    ) -> CLIPZymeOutput:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            Batch of data.
        batch_idx : int, optional
            batch id, by default 0
        extract_protein_features : bool, optional
            whether to return protein features, by default False
        extract_reaction_features : bool, optional
            whether to return reaction features, by default False

        Returns
        -------
        CLIPZymeOutput
            Output of the model.
        """
        reaction_scores = None
        protein_hiddens = None
        reaction_hiddens = None

        model_output = self.model(batch)

        predictions_dict = {}
        predictions_dict = self.store_in_predictions(predictions_dict, batch)
        predictions_dict = self.store_in_predictions(predictions_dict, model_output)

        # want paired scores
        if not (self.use_as_protein_encoder or self.use_as_reaction_encoder):
            reaction_hiddens = model_output["substrate_hiddens"]
            protein_hiddens = model_output["protein_hiddens"]
            reaction_scores = torch.einsum(
                "bx,bx->b", reaction_hiddens, protein_hiddens
            )

        if self.use_as_protein_encoder:
            protein_hiddens = model_output["hidden"]
        if self.use_as_reaction_encoder:
            reaction_hiddens = model_output["hidden"]

        output = CLIPZymeOutput(
            scores=reaction_scores,
            protein_hiddens=protein_hiddens,
            reaction_hiddens=reaction_hiddens,
            sample_ids=batch["sample_id"],
        )

        return output

    def test_step(self, batch, batch_idx) -> CLIPZymeOutput:
        """
        Test step for the model.

        Parameters
        ----------
        batch : dict
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        ClipZymeOutput
            Output of the model.
        """
        output = self.forward(batch, batch_idx)
        if self.save_hiddens or self.save_predictions:
            self.save_outputs(output)
        return output

    def on_test_epoch_end(self) -> None:
        self.clean_up()

    def save_outputs(self, outputs: CLIPZymeOutput) -> None:
        """
        Save outputs to disk.

        Parameters
        ----------
        outputs : ClipZymeOutput
            Output of the model.
        """
        if outputs.protein_hiddens is not None:
            protein_hiddens = outputs.protein_hiddens.cpu()
            for idx, protein_hidden in enumerate(protein_hiddens):
                predictions_filename = self.hiddens_dir.joinpath(
                    f"sample_{outputs.sample_ids[idx]}.protein.pt"
                )
                torch.save(protein_hidden, predictions_filename)

        if outputs.reaction_hidden is not None:
            reaction_hiddens = outputs.reaction_hidden.cpu()
            for idx, reaction_hidden in enumerate(reaction_hiddens):
                predictions_filename = self.hiddens_dir.joinpath(
                    f"sample_{outputs.sample_ids[idx]}.reaction.pt"
                )
                torch.save(reaction_hidden, predictions_filename)

        if outputs.scores is not None:
            scores = outputs.scores.cpu()
            for idx, score in enumerate(scores):
                predictions_filename = self.predictions_dir.joinpath(
                    f"sample_{outputs.sample_ids[idx]}.score.pt"
                )
                torch.save(score.item(), predictions_filename)

    def extract_protein_features(self, batch: dict) -> torch.Tensor:
        """
        Extract protein features from model.

        Parameters
        ----------
        batch : dict
            Batch of data.

        Returns
        -------
        torch.Tensor
            Protein features.
        """
        self.use_as_protein_encoder = True
        model_output = self.model(batch)
        return model_output["protein_hiddens"]

    def extract_reaction_features(self, batch: dict) -> torch.Tensor:
        """
        Extract reaction features from model.

        Parameters
        ----------
        batch : dict
            Batch of data.

        Returns
        -------
        torch.Tensor
            Reaction features.
        """
        self.use_as_reaction_encoder = True
        model_output = self.model(batch)
        return model_output["substrate_hiddens"]

    def clean_up(self):
        """
        Clean up directories after inference.
        """
        if self.save_predictions:
            print("Collecting predictions files")

            with open(self.predictions_path, "w") as f:
                f.write("sample_id,score\n")
                for score_file in self.predictions_dir.iterdir():
                    score = torch.load(self.predictions_dir.joinpath(score_file))
                    f.write(f"{score_file.stem},{score}\n")

            self.predictions_dir.rmdir()
        print("Done")
