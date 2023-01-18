from nox.utils.registry import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from nox.utils.classes import Nox
from nox.utils.digress import diffusion_utils


@register_object("digress_cross_entropy", "loss")
class DigressLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    # def __call__(self, model_output, batch, model, args):
    def forward(
        self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, args
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean."""
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(
            masked_pred_X, (-1, masked_pred_X.size(-1))
        )  # (bs * n, dx)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        if true_X.numel() > 0:
            target = torch.argmax(flat_true_X, dim=-1)
            loss_X = F.cross_entropy(
                flat_pred_X, target, reduction="sum"
            ) / flat_true_X.size(0)
        else:
            loss_X = 0.0

        if true_X.numel() > 0:
            target = torch.argmax(flat_true_E, dim=-1)
            loss_E = F.cross_entropy(
                flat_pred_E, target, reduction="sum"
            ) / flat_true_X.size(0)
        else:
            loss_E = 0.0

        if true_X.numel() > 0:
            target = torch.argmax(true_y, dim=-1)
            loss_y = F.cross_entropy(
                pred_y, target, reduction="sum"
            ) / flat_true_X.size(0)
        else:
            loss_y = 0.0

        loss = loss_X + args.lambda_edge_loss * loss_E + args.lambda_y_loss[1] * loss_y

        logging_dict = {
            "loss": loss,
            "node_loss": loss_X,
            "edge_loss": loss_E,
            "y_loss": loss_y,
        }
        predictions = {}
        return loss, logging_dict, predictions

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        parser.add_argument(
            "--lambda_edge_loss",
            type=float,
            default=5.0,
            help="Lambda to weigh the cross-entropy loss.",
        )
        parser.add_argument(
            "--lambda_y_loss",
            type=float,
            default=0.0,
            help="Lambda to weigh the cross-entropy loss.",
        )


@register_object("digress_simple_vlb", "loss")
class DigressSimpleVLBLoss(Nox):
    def __init__(self) -> None:
        super().__init__()

    # def __call__(self, model_output, batch, model, args):
    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        test = model.phase == "test"

        t = noisy_data["t"]

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, y, node_mask, model)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test, model)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask, model)

        x_samples = X * prob0.X.log()
        e_samples = E * prob0.E.log()
        y_samples = y * prob0.y.log()
        loss_term_0 = (
            x_samples.sum() / len(x_samples)
            + e_samples.sum() / len(e_samples)
            + y_samples.sum() / len(y_samples)
        )

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

        # Update NLL metric object and return batch nll
        nll = torch.sum(nlls) / nlls.numel()  # Average over the batch

        return {
            "nll": nll,
            "kl prior": kl_prior.mean(),
            "Estimator loss terms": loss_all_t.mean(),
            "log_pn": log_pN.mean(),
            "loss_term_0": loss_term_0,
            "test_nll" if test else "val_nll": nll,
        }

    def kl_prior(self, X, E, y, node_mask, model):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = model.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = model.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        proby = y @ Qtb.y if y.numel() > 0 else y
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = model.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = (
            model.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        )
        uniform_dist_y = torch.ones_like(proby) / model.ydim_output

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
        )

        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction="none"
        )
        kl_distance_y = F.kl_div(
            input=proby.log(), target=uniform_dist_y, reduction="none"
        )

        return (
            diffusion_utils.sum_except_batch(kl_distance_X)
            + diffusion_utils.sum_except_batch(kl_distance_E)
            + diffusion_utils.sum_except_batch(kl_distance_y)
        )

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test, model):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = model.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], model.device)
        Qsb = model.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], model.device)
        Qt = model.transition_model.get_Qt(noisy_data["beta_t"], model.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(
            X=X,
            E=E,
            y=y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(
            X=pred_probs_X,
            E=pred_probs_E,
            y=pred_probs_y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        (
            prob_true_X,
            prob_true_E,
            prob_pred.X,
            prob_pred.E,
        ) = diffusion_utils.mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask,
        )
        kl_x = F.kl_div(prob_true.X, torch.log(prob_pred.X))
        kl_e = F.kl_div(prob_true.E, torch.log(prob_pred.E))
        kl_y = (
            F.kl_div(prob_true.y, torch.log(prob_pred.y))
            if pred_probs_y.numel() != 0
            else 0
        )

        return kl_x + kl_e + kl_y

    def reconstruction_logp(self, t, X, E, y, node_mask, model):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = model.noise_schedule(t_zeros)
        Q0 = model.transition_model.get_Qt(beta_t=beta_0, device=model.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(
            probX=probX0, probE=probE0, node_mask=node_mask
        )

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = diffusion_utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": torch.zeros(X0.shape[0], 1).type_as(y0),
        }
        extra_data = model.model.compute_extra_data(noisy_data)

        noisyX = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        noisyE = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        noisyy = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        pred0 = model.model.model(noisyX, noisyE, noisyy, node_mask)  # graph denoiser

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(
            self.Edim_output
        ).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return diffusion_utils.PlaceHolder(X=probX0, E=probE0, y=proby0)
