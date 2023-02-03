import torch
import torch.nn as nn
import torch.nn.functional as F
from nox.utils.registry import register_object, get_object
from nox.models.abstract import AbstractModel
from nox.utils.digress.noise_schedule import (
    DiscreteUniformTransition,
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
)
from nox.utils.digress import diffusion_utils
import os
from nox.utils.classes import set_nox_type
from rich import print 
import copy 

@register_object("digress", "model")
class Digress(AbstractModel):
    def __init__(self, args):
        super(Digress, self).__init__()

        self.args = args

        self.model_dtype = torch.float32
        self.T = args.diffusion_steps  
        self.model = get_object(args.digress_graph_denoiser, "model")(args)  

        # cosine diffusion schedule
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            args.diffusion_noise_schedule, timesteps=args.diffusion_steps  
        )

        self.dataset_info = args.dataset_statistics

        self.node_dist = self.dataset_info.nodes_dist

        input_dims = args.dataset_statistics.input_dims
        output_dims = args.dataset_statistics.output_dims

        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]

        self.register_buffer("devicevar", torch.zeros(1, dtype=torch.int8))
        
        # TODO: move to dataset 
        self.extra_features = args.extra_features
        self.domain_features = args.domain_features

        if args.digress_transition == "uniform":  
            self.transition_model = DiscreteUniformTransition(  
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = diffusion_utils.PlaceHolder(  
                X=x_limit, E=e_limit, y=y_limit
            )
        elif args.digress_transition == "marginal":  

            node_types = self.dataset_info.node_types.float()  
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()  
            e_marginals = edge_types / torch.sum(edge_types)  

            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges"
            )
            self.transition_model = MarginalUniformTransition(  
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = diffusion_utils.PlaceHolder(  
                X=x_marginals,
                E=e_marginals,
                y=torch.ones(self.ydim_output) / self.ydim_output,
            )

    def forward(self, data):
        dense_data, node_mask = diffusion_utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()

        denoiser_input = {
            "X": X,
            "E": E, 
            "y": y, 
            "node_mask":node_mask
        }

        pred = self.model(denoiser_input)

        output = {
            "masked_pred_X": pred.X, 
            "masked_pred_E": pred.E,  
            "pred_y": pred.y,  
            "true_X": dense_data.X, 
            "true_E": dense_data.E, 
            "true_y": data.y, 
            "noisy_data": noisy_data, 
            "dense_data": dense_data, 
            "extra_data": extra_data, 
            "node_mask": node_mask,
        }

        return output

    def apply_noise(self, X, E, y, node_mask):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(X.size(0), 1), device=self.devicevar.device
        ).float()  # (bs, 1) # a timestep t for each sample
        s_int = t_int - 1  # one timestep before t (for going from t to t-1)

        t_float = t_int / self.T  # normalized timestep
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation

        # get the beta's associated with each timestep
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)

        # get alpha bar (for cosine schedule is (f(t)/f(0)) EQN 18
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        # get transition matrices for X, E, y (as placeholder obj)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=X.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = (
            diffusion_utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        )

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }
        return noisy_data

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        empty_placeholder = diffusion_utils.PlaceHolder(
            X=torch.zeros(0, device=self.devicevar.device),
            E=torch.zeros(0, device=self.devicevar.device),
            y=torch.zeros(0, device=self.devicevar.device)
        )

        if self.extra_features is not None:
            extra_features = self.extra_features(noisy_data)
        else:
            extra_features = empty_placeholder

        if self.domain_features is not None:
            extra_molecular_features = self.domain_features(noisy_data)
        else:
            extra_molecular_features = empty_placeholder

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return diffusion_utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        num_nodes=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.devicevar.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.devicevar.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.devicevar.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )

        chain_X = torch.zeros(chain_X_size) # chain length, batch size, max num nodes 
        chain_E = torch.zeros(chain_E_size) # chain length, batch size, max num nodes x max num nodes

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return {
            "chain_X": chain_X,
            "chain_E": chain_E,
            "molecules": molecule_list
        }

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1) the noise level associated with the timestep
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.devicevar.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.devicevar.device)
        Qt = self.transition_model.get_Qt(beta_t, self.devicevar.device)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data)
        
        denoiser_input = {
            "X": torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float(),
            "E": torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float(), 
            "y": torch.hstack((noisy_data["y_t"], extra_data.y)).float(), 
            "node_mask":node_mask
        }

        pred = self.model(denoiser_input)

        # Normalize predictions to get p(x) and p(e) (prior in EQN 6)
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        # Compute posterior q(s | t, x) EQN 1
        p_s_and_t_given_0_X = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
        )

        p_s_and_t_given_0_E = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )
        )

        # Compute EQN 6
        # Dim of these two tensors: bs, N, d0, d_t-1 
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1  
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = diffusion_utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0)
        )
        out_discrete = diffusion_utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0)
        )

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(
            node_mask, collapse=True
        ).type_as(y_t)

    @staticmethod
    def add_args(parser) -> None:
        """Add class specific args

        Args:
            parser (argparse.ArgumentParser): argument parser
        """
        super(Digress, Digress).add_args(parser)
        parser.add_argument(
            "--diffusion_steps",
            type=int,
            default=100,
            help="number of steps in diffusion process",
        )
        parser.add_argument(
            "--digress_graph_denoiser",
            type=str,
            action=set_nox_type("model"),
            default="graph_transformer",
            help="denoiser model",
        )
        parser.add_argument(
            "--diffusion_noise_schedule",
            type=str,
            default="cosine",
            choices=["cosine", "custom"],
            help="noise schedule",
        )
        parser.add_argument(
            "--digress_transition",
            type=str,
            default="uniform",
            choices=["uniform", "marginal"],
            help="transition matrix distribution",
        )


@register_object("digress_reactions", "model")
class DigressReactants(Digress):
    def __init__(self, args):
        super(DigressReactants, self).__init__(args)
        in_dim = self.Xdim_output + self.Edim_output 
        max_num_nodes = self.dataset_info.max_n_nodes  

        self.product_size_predictor = nn.Linear(in_dim, max_num_nodes)


    def forward(self, data):
        # load reactant / products as PyG batches
        reactants = data["reactants"]
        products = data["products"]

        # reactants 
        reactant_dense_data, reactant_node_mask = diffusion_utils.to_dense(
            reactants.x, reactants.edge_index, reactants.edge_attr, reactants.batch
        )
        reactant_dense_data = reactant_dense_data.mask(reactant_node_mask)
        
        # products
        products_dense_data, products_node_mask = diffusion_utils.to_dense(
            products.x, products.edge_index, products.edge_attr, products.batch
        )
        products_dense_data = products_dense_data.mask(products_node_mask)
        noisy_data = self.apply_noise(products_dense_data.X, products_dense_data.E, products.y, products_node_mask)
        
        # return only product-related data
        product_noisy_data = copy.deepcopy(noisy_data) 
        products_extra_data = self.compute_extra_data(noisy_data)
        
        # concatenate nodes from reactants and products
        noisy_data["X_t"] = torch.cat((noisy_data["X_t"], reactant_dense_data.X), dim=1).float() 

        #concatenate nodes from reactants and products
        num_reactant_nodes = reactant_dense_data.E.shape[1]
        num_product_nodes = products_dense_data.E.shape[1]

        # padding begins from last dimension
        # leave last as is (edge features), right pad products (upper left block), left pad reactants and combine (lower right block)
        product_pad = (0,0, 0,num_reactant_nodes, 0,num_reactant_nodes ) 
        reactant_pad =  (0,0, num_product_nodes,0 , num_product_nodes,0)

        E_products = torch.nn.functional.pad(noisy_data["E_t"],  product_pad, "constant", 0)
        E_reactants = torch.nn.functional.pad(reactant_dense_data.E,  reactant_pad, "constant", 0)

        noisy_data["E_t"] = E_products + E_reactants

        # concatenate node_mask
        node_mask = torch.cat((products_node_mask, reactant_node_mask), dim=1)
        noisy_data['node_mask'] = node_mask

        # add extra features
        extra_data = self.compute_extra_data(noisy_data)

        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()

        denoiser_input = {
            "X": X,
            "E": E, 
            "y": y, 
            "node_mask":noisy_data['node_mask']
        }

        pred = self.model(denoiser_input)

        output = {
            "masked_pred_X": pred.X[:, :num_product_nodes], # keep only product denoising
            "masked_pred_E": pred.E[:, :num_product_nodes, :num_product_nodes], # keep only product denoising
            "pred_y": pred.y,          
            "true_X": products_dense_data.X, 
            "true_E": products_dense_data.E, 
            "true_y": products.y, 
            "noisy_data": product_noisy_data, 
            "dense_data": products_dense_data, 
            "extra_data": products_extra_data, 
            "node_mask": products_node_mask,
        }


        # use reaction embeddings to predict number of product nodes
        reactant_embeds = torch.cat([
            (pred.X[:, num_product_nodes:]).sum(1),
            (pred.E[:, num_product_nodes:, num_product_nodes:]).sum((1,2)),
        ], -1)

        output["logit"] = self.product_size_predictor(reactant_embeds)
        output["y"] = products_node_mask.sum(1) - 1 # subtract 1 so that 1st class corresponds to size of 1

        return output
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        num_nodes=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.devicevar.device) 
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.devicevar.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.devicevar.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )

        chain_X = torch.zeros(chain_X_size) # chain length, batch size, max num nodes 
        chain_E = torch.zeros(chain_E_size) # chain length, batch size, max num nodes x max num nodes

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, y, node_mask
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        return {
            "chain_X": chain_X,
            "chain_E": chain_E,
            "molecules": molecule_list
        }