import torch
from nox.lightning.base import Base, gather_step_outputs
from nox.utils.registry import get_object, register_object


@register_object("diffusion", "lightning")
class DiscreteDenoisingDiffusion(Base):
    def __init__(self, args):
        super().__init__(args)
        self.sampling_metric_val = get_object(args.sampling_metric, "metric")(args)
        self.sampling_metric_test = get_object(args.sampling_metric, "metric")(args)

    def validation_epoch_end(self, outputs) -> None:

        self.phase = "val"
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        outputs["loss"] = outputs["loss"].mean()

        self.args.val_counter += 1
        if self.args.val_counter % self.args.sample_every_val == 0:

            samples_left_to_generate = self.args.samples_to_generate
            samples_left_to_save = self.args.samples_to_save
            chains_left_to_save = self.args.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.args.batch_size
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
                        number_chain_steps=self.args.number_chain_steps,
                    )
                )
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

        self.sampling_metric_val.update({"molecules": samples})
        outputs.update(self.sampling_metric_val.compute())
        self.log_outputs(outputs, "val")

        for metric_fn in self.metrics["val"]:
            metric_fn.reset()
        self.sampling_metric_val.reset()

    def test_epoch_end(self, outs) -> None:
        """Measure likelihood on a test set and compute stability metrics."""

        self.phase = "test"
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        if isinstance(outputs.get("loss", 0), torch.Tensor):
            outputs["loss"] = outputs["loss"].mean()

        samples_left_to_generate = self.args.final_model_samples_to_generate
        samples_left_to_save = self.args.final_model_samples_to_save
        chains_left_to_save = self.args.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{self.args.final_model_samples_to_generate}",
                end="",
                flush=True,
            )
            bs = 2 * self.args.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(
                self.model.sample_batch(
                    id,
                    to_generate,
                    num_nodes=None,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=self.args.number_chain_steps,
                )
            )
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if not self.args.predict:
            self.sampling_metric_test.update({"molecules": samples})
            outputs.update(self.sampling_metric_test.compute())
            self.log_outputs(outputs, "test")

        for metric_fn in self.metrics["test"]:
            metric_fn.reset()
        self.sampling_metric_test.reset()

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(DiscreteDenoisingDiffusion, DiscreteDenoisingDiffusion).add_args(parser)
        parser.add_argument(
            "--sampling_metric",
            type=str,
            default="molecule_sampling_metrics",
            help="metric to use for sampling",
        )
        parser.add_argument(
            "--number_chain_steps",
            type=int,
            default=10,
            help="number of chain steps to take for sampling.",
        )
        parser.add_argument(
            "--sample_every_val",
            type=int,
            default=10,
            help="how often to sample.",
        )
        parser.add_argument(
            "--samples_to_generate",
            type=int,
            default=10,
            help="number of samples to generate.",
        )
        parser.add_argument(
            "--samples_to_save",
            type=int,
            default=10,
            help="number of samples to save.",
        )
        parser.add_argument(
            "--chains_to_save", type=int, default=10, help="number of chains to save."
        )
        parser.add_argument(
            "--val_counter",
            type=int,
            default=0,
            help="counter for how often to sample",
        )
        parser.add_argument(
            "--final_model_samples_to_generate",
            type=int,
            default=10,
            help="number of samples to generate at inference.",
        )
        parser.add_argument(
            "--final_model_samples_to_save",
            type=int,
            default=10,
            help="number of samples to save at inference.",
        )
        parser.add_argument(
            "--final_model_chains_to_save",
            type=int,
            default=10,
            help="number of chains to save at inference.",
        )
