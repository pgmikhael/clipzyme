import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from nox.lightning.base import Base


class DiscreteDenoisingDiffusion(Base):
    def __init__(self, args):
        super().__init__(args)

        self.log_every_steps = args.log_every_steps
        self.number_chain_steps = args.number_chain_steps  # 10
        self.best_val_nll = 1e8
        self.val_counter = 0

    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_epoch_end(self, outs) -> None:
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute(),
            self.val_E_kl.compute(),
            self.val_y_kl.compute(),
            self.val_X_logp.compute(),
            self.val_E_logp.compute(),
            self.val_y_logp.compute(),
        ]
        wandb.log(
            {
                "val/epoch_NLL": metrics[0],
                "val/X_kl": metrics[1],
                "val/E_kl": metrics[2],
                "val/y_kl": metrics[3],
                "val/X_logp": metrics[4],
                "val/E_logp": metrics[5],
                "val/y_logp": metrics[6],
            },
            commit=False,
        )

        print(
            f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
            f"Val Edge type KL: {metrics[2] :.2f} -- Val Global feat. KL {metrics[3] :.2f}\n",
        )

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print("Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(
                    self.model.sample_batch(
                        batch_id=ident,
                        batch_size=to_generate,
                        num_nodes=None,
                        save_final=to_save,
                        keep_chain=chains_save,
                        number_chain_steps=self.number_chain_steps,
                    )
                )
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            print("Computing sampling metrics...")
            self.sampling_metrics(
                samples, self.name, self.current_epoch, val_counter=-1, test=False
            )
            print(f"Done. Sampling took {time.time() - start:.2f} seconds\n")
            self.sampling_metrics.reset()

    def test_epoch_end(self, outs) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        metrics = [
            self.test_nll.compute(),
            self.test_X_kl.compute(),
            self.test_E_kl.compute(),
            self.test_y_kl.compute(),
            self.test_X_logp.compute(),
            self.test_E_logp.compute(),
            self.test_y_logp.compute(),
        ]
        wandb.log(
            {
                "test/epoch_NLL": metrics[0],
                "test/X_mse": metrics[1],
                "test/E_mse": metrics[2],
                "test/y_mse": metrics[3],
                "test/X_logp": metrics[4],
                "test/E_logp": metrics[5],
                "test/y_logp": metrics[6],
            },
            commit=False,
        )

        print(
            f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
            f"Test Edge type KL: {metrics[2] :.2f} -- Test Global feat. KL {metrics[3] :.2f}\n",
        )

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f"Test loss: {test_nll :.4f}")

        # ! TODO < ----------

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{self.cfg.general.final_model_samples_to_generate}",
                end="",
                flush=True,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(
                self.sample_batch(
                    id,
                    to_generate,
                    num_nodes=None,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                )
            )
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(
            samples, self.name, self.current_epoch, self.val_counter, test=True
        )
        self.sampling_metrics.reset()
        print("Done.")
