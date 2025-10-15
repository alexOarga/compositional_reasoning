import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter
from torch.distributions import Normal
from comp_reasoning.utils import partition_graph_diameter_N


def timestep_embedding(timesteps, dim, max_period=1000):
    """
    Create sinusoidal timestep embeddings.
    timesteps: (batch,) tensor of timesteps
    dim: embedding dimension
    """
    half = dim // 2
    # compute frequencies
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    # outer product: batch x half
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    # concat sin and cos
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # if dim is odd, pad one column
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPModel(nn.Module):
    def __init__(self,
                 n_steps,
                 n_steps_annealed,
                 n_layers,
                 x_dim,
                 h_dim,
                 out_dim,
                 emb_dim,
                 widen=2,
                 emb_type='learned',
                 annealed_landscape=False):
        super(MLPModel, self).__init__()

        assert emb_type in ('learned', 'sinusoidal', 'none'), "emb_type must be 'learned' or 'sinusoidal'"

        self.n_layers = n_layers
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.widen = widen
        self.emb_type = emb_type
        self.annealed_landscape = annealed_landscape

        if emb_type == 'learned':
            self.embedding = nn.Embedding(n_steps, emb_dim)
            if n_steps_annealed is not None:
                self.embedding_annealed = nn.Embedding(n_steps_annealed, emb_dim)
            else:
                self.embedding_annealed = None
        elif emb_type == 'sinusoidal':
            # embeddings are generated on-the-fly
            self.embedding = None
            self.embedding_annealed = None
        elif emb_type == 'none':
            self.embedding = None
            self.embedding_annealed = None
        else:
            raise ValueError("emb_type must be 'learned', 'sinusoidal', or 'none'")

        self.initial_linear = nn.Linear(x_dim, h_dim)
        self.layers_h = nn.ModuleList([nn.Linear(h_dim, h_dim * widen) for _ in range(n_layers)])
        self.layers_emb = nn.ModuleList([nn.Linear(emb_dim, h_dim * widen) for _ in range(n_layers)])
        self.layers_emb_ann = nn.ModuleList([nn.Linear(emb_dim, h_dim * widen) for _ in range(n_layers)])
        self.layers_int = nn.ModuleList([nn.Linear(h_dim * widen, h_dim * widen) for _ in range(n_layers)])
        self.layers_out = nn.ModuleList([nn.Linear(h_dim * widen, h_dim) for _ in range(n_layers)])
        self.final_linear = nn.Linear(h_dim, out_dim)

    def forward(self, x, t=None, t_annealed=None, x_initial=None):
        if x_initial is not None:
            x = torch.cat((x, x_initial), dim=-1)

        # --- Embedding selection ---
        if t is not None:
            if self.emb_type == 'learned':
                emb = self.embedding(t)
                emb_ann = self.embedding_annealed(t_annealed) if (self.annealed_landscape and t_annealed is not None) else None
            elif self.emb_type == 'sinusoidal':
                emb = timestep_embedding(t, self.emb_dim)
                emb_ann = timestep_embedding(t_annealed, self.emb_dim) if (self.annealed_landscape and t_annealed is not None) else None
        else:
            emb, emb_ann = None, None

        # --- Network forward ---
        x = self.initial_linear(x)

        for i in range(self.n_layers):
            h = x
            h = F.layer_norm(x, normalized_shape=(self.h_dim,), eps=1e-5)
            h = F.relu(h)
            h = self.layers_h[i](h)

            if emb is not None:
                if self.annealed_landscape and emb_ann is not None:
                    h = h + self.layers_emb[i](emb) + self.layers_emb_ann[i](emb_ann)
                else:
                    h = h + self.layers_emb[i](emb)

            h = F.relu(h)
            h = self.layers_int[i](h)
            h = F.relu(h)
            h = self.layers_out[i](h)
            x = x + h  # residual

        x = self.final_linear(x)
        return x


class ColorGNN(nn.Module):
    def __init__(self, 
                 n_steps,
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers=2, 
                 pre_mlp_layers=1,
                 post_mlp_layers=2,
                 conv_type='custom', 
                 aggr='mean'):
        super(ColorGNN, self).__init__()

        assert conv_type in ['GCN', 'SAGE', 'GAT'], "conv_type must be one of: GCN, SAGE, GAT"
        assert aggr in ['mean', 'sum', 'max'], "aggr must be one of: mean, sum, max"
        assert pre_mlp_layers >= 0 and post_mlp_layers >= 1

        self.num_layers = num_layers
        self.embedding = nn.Embedding(n_steps, hidden_channels)

        Conv = {
            'GCN': GCNConv,
            'SAGE': SAGEConv,
            'GAT': GATConv
        }[conv_type]

        #self.first_layer = CustomGNNLayer(in_dim=hidden_channels, hidden_dim=hidden_channels)
        self.first_layer = nn.Linear(in_channels, hidden_channels)

        self.pre_mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.post_mlps = nn.ModuleList()

        # First layer
        self.pre_mlps.append(self._build_mlp(hidden_channels, hidden_channels, pre_mlp_layers))
        self.convs.append(Conv(hidden_channels, hidden_channels))
        self.post_mlps.append(self._build_mlp(hidden_channels, hidden_channels, post_mlp_layers))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.pre_mlps.append(self._build_mlp(hidden_channels, hidden_channels, pre_mlp_layers))
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.post_mlps.append(self._build_mlp(hidden_channels, hidden_channels, post_mlp_layers))

        self.final_layer = nn.Linear(hidden_channels, out_channels)

        self.global_pool = {
            'mean': global_mean_pool,
            'sum': global_add_pool,
            'max': global_max_pool
        }[aggr]

    def _build_mlp(self, in_dim, out_dim, num_layers):
        if num_layers == 0:
            return nn.Identity()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, t, x_initial, batch=None):
        #x = self.feature_encoder(x, t, t_annealed=None, x_initial=None)
        edge_index = x_initial
        
        #x = self.first_layer(x, edge_index)
        x = self.first_layer(x)

        for pre_mlp, conv, post_mlp in zip(self.pre_mlps, self.convs, self.post_mlps):
            x_init = x.clone()
            x = pre_mlp(x) + self.embedding(t)
            x = conv(x, edge_index)
            #x = x + self.embedding(t)
            x = post_mlp(x) + x_init  # Residual connection

        x = self.final_layer(x)

        if batch is not None:
            x = self.global_pool(x, batch)

        return x


def build_mlp(input_dim, hidden_dims, output_dim,
              activation='relu', dropout=0.0,
              emb_dim=32, n_steps=100, n_steps_annealed=100,
              emb_type='learned', annealed_landscape=False,
              widen=2) -> nn.Module:

    # Derive h_dim from hidden_dims: if multiple, pick the last one
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must have at least one element.")
    h_dim = hidden_dims[-1]
    n_layers = len(hidden_dims)

    return MLPModel(
        n_steps=n_steps,
        n_steps_annealed=n_steps_annealed,
        n_layers=n_layers,
        x_dim=input_dim,
        h_dim=h_dim,
        out_dim=output_dim,
        emb_dim=emb_dim,
        widen=widen,
        emb_type=emb_type,
        annealed_landscape=annealed_landscape,
    )


class EBMDiffModel(nn.Module):
    def __init__(self,
                 net):
        super(EBMDiffModel, self).__init__()
        self.net = net

    def neg_logp_unnorm(self, x, t, x_initial=None):
        energy_score = self.net(x, t, x_initial=x_initial)
        res = torch.norm(energy_score, dim=-1)
        return res

    def forward(self, x, t, x_initial=None):
        torch.set_grad_enabled(True)

        if self.training:
            create_graph = True
        else:
            create_graph = False

        x.requires_grad_(requires_grad=True)
        out = self.neg_logp_unnorm(x, t, x_initial=x_initial)
        epsilon = torch.autograd.grad(outputs=out.sum(), inputs=x, create_graph=create_graph)
        return epsilon[0]
    

class EBMDiffReducedQueensModel(nn.Module):
    def __init__(self,
                 net,
                 base_dimension,
                 reduced_dimension):
        super(EBMDiffReducedQueensModel, self).__init__()
        self.net = net
        self.base_dimension = base_dimension
        self.reduced_dimension = reduced_dimension

    def neg_logp_unnorm(self, x, t, t_annealed, x_initial=None):
        x_clone = x.clone()
        # pad x with zeros on dim=1 until it reaches the base dimension
        if x_clone.shape[1] < self.base_dimension:
            x_clone = F.pad(x_clone, (0, self.base_dimension - x_clone.shape[1]), value=0)
        energy_score = self.net(x_clone, t, t_annealed, x_initial=x_initial)
        res = torch.norm(energy_score, dim=-1)
        return res

    def forward(self, x, t, t_annealed, x_initial=None):
        torch.set_grad_enabled(True)

        x_clone = x.clone()
        # pad x with zeros on dim=1 until it reaches the base dimension
        if x_clone.shape[1] < self.base_dimension:
            x_clone = F.pad(x_clone, (0, self.base_dimension - x_clone.shape[1]), value=0)

        if self.training:
            create_graph = True
        else:
            create_graph = False

        x_clone.requires_grad_(requires_grad=True)
        out = self.neg_logp_unnorm(x_clone, t, t_annealed, x_initial=x_initial)
        epsilon = torch.autograd.grad(outputs=out.sum(), inputs=x_clone, create_graph=create_graph)
        # reduce epsilon to the reduced dimension
        epsilon = epsilon[0][:, 0:self.reduced_dimension]
        return epsilon
    

class EBMProduct8queens(nn.Module):
    def __init__(self,
                 dimension,
                 net_rows,
                 net_diags,
                 weight_rows,
                 weight_diags):
        super(EBMProduct8queens, self).__init__()

        self.net_rows = net_rows
        self.net_diags = net_diags
        self.weight_rows = weight_rows
        self.weight_diags = weight_diags
        self.dimension = dimension

    def forward(self, x, t, t_annealed, x_initial):
        # NOTE: there is no need to write the forward function as below
        # but this is generally more memory efficient
        DIM = self.dimension
        x = x.reshape((-1, DIM, DIM))
        batch_size = x.shape[0]
        energies = torch.zeros(batch_size, DIM, DIM, device=x.device)
        count = torch.zeros(batch_size, DIM, DIM, dtype=torch.int32, device=x.device)

        N = DIM # rows
        M = DIM # columns
        for r in range(x.shape[1]):
            energies[:, r] += self.weight_rows * self.net_rows(x[:, r], t, t_annealed, None)
            count[:, r] += 1
        for c in range(x.shape[2]):
            energies[:, :, c] += self.weight_rows * self.net_rows(x[:, :, c], t, t_annealed, None)
            count[:, :, c] += 1
        for d in range(-(DIM - 1), DIM):
            diag_index = []
            for i in range(DIM):
                if 0 <= i + d < DIM:
                    diag_index.append((i, i + d))
            diags = []
            for rr, cc in diag_index:
                diags.append(x[:, rr, cc])
            diags = torch.stack(diags, dim=1)
            e = self.weight_diags * self.net_diags(F.pad(diags, (0, DIM - diags.shape[1])), t, t_annealed, None)
            for i, (rr, cc) in enumerate(diag_index):
                energies[:, rr, cc] += e[:, i]
                count[:, rr, cc] += 1
        # get fliped diagonal
        for d in range(-(DIM - 1), DIM):
            diag_index = []
            for i in range(DIM):
                if 0 <= i + d < DIM:
                    diag_index.append((i, (DIM-1) - i - d))
            diags = []
            for rr, cc in diag_index:
                diags.append(x[:, rr, cc])
            diags = torch.stack(diags, dim=1)
            e = self.weight_diags * self.net_diags(F.pad(diags, (0, DIM - diags.shape[1])), t, t_annealed, None)
            for i, (rr, cc) in enumerate(diag_index):
                energies[:, rr, cc] += e[:, i]
                count[:, rr, cc] += 1

        energies = energies / count # NOTE: empirically this usually works better than sum
        energies = energies.reshape((batch_size, N * M))
        return energies


    def neg_logp_unnorm(self, x, t, t_annealed, x_initial):
        DIM = self.dimension
        x = x.reshape((-1, DIM, DIM))
        batch_size = x.shape[0]
        # we accumulate energy in a DIM x DIM matrix, but there is no need to do this, it is just for debbugging
        energies = torch.zeros(batch_size, DIM, DIM, device=x.device) 
        count = torch.zeros(batch_size, DIM, DIM, dtype=torch.int32, device=x.device)

        N = DIM # rows
        M = DIM # columns
        for r in range(x.shape[1]):
            energies[:, r] += self.weight_rows * self.net_rows.neg_logp_unnorm(x[:, r], t, t_annealed, None).unsqueeze(-1)
            count[:, r] += 1
        for c in range(x.shape[2]):
            energies[:, :, c] += self.weight_rows * self.net_rows.neg_logp_unnorm(x[:, :, c], t, t_annealed, None).unsqueeze(-1)
            count[:, :, c] += 1
        for d in range(-(DIM-1), DIM):
            diag_index = []
            for i in range(DIM):
                if 0 <= i + d < DIM:
                    diag_index.append((i, i + d))
            diags = []
            for rr, cc in diag_index:
                diags.append(x[:, rr, cc])
            diags = torch.stack(diags, dim=1)
            diags = F.pad(diags, (0, DIM - diags.shape[1]))
            e = self.weight_diags * self.net_diags.neg_logp_unnorm(diags, t, t_annealed, None).unsqueeze(-1)
            for i, (rr, cc) in enumerate(diag_index):
                energies[:, rr, cc] += e[:, 0]
                count[:, rr, cc] += 1
        # get inverse diagonal
        for d in range(-(DIM - 1), DIM):
            diag_index = []
            for i in range(DIM):
                if 0 <= i + d < DIM:
                    diag_index.append((i, (DIM-1) - i - d))
            diags = []
            for rr, cc in diag_index:
                diags.append(x[:, rr, cc])
            diags = torch.stack(diags, dim=1)
            diags = F.pad(diags, (0, DIM - diags.shape[1]))
            e = self.weight_diags * self.net_diags.neg_logp_unnorm(diags, t, t_annealed, None).unsqueeze(-1)
            for i, (rr, cc) in enumerate(diag_index):
                energies[:, rr, cc] += e[:, 0]
                count[:, rr, cc] += 1

        energies = energies / count
        energies = energies.reshape((batch_size, N * M))

        return energies.mean(dim=-1)
    

class EBMSAT(nn.Module):
    def __init__(self,
                 net,
                 batch_clauses,
                 num_clauses=1, 
                 ):
        super(EBMSAT, self).__init__()
        self.net = net
        self.num_clauses = num_clauses
        self.batch_clauses = batch_clauses

    def neg_logp_unnorm(self, x, t, t_annealed, x_initial, mask_clause):
        energy = torch.zeros_like(x)
        count = torch.zeros_like(x)
        BC = self.batch_clauses
        batch_size = x.shape[0]

        # before proceeding, group the clauses according to models num_clauses
        grouped_x_initial = []
        for i in range(0, len(x_initial), self.num_clauses):
            list_clauses = x_initial[i: i + self.num_clauses]
            if len(list_clauses) < self.num_clauses:
                for _ in range(self.num_clauses - len(list_clauses)):
                    # pad with zero-clauses
                    list_clauses.append(torch.zeros_like(list_clauses[0]))
                    # add False column to mask_clause
                    mask_clause = torch.cat([mask_clause, torch.zeros_like(mask_clause[:, 0:1])], dim=1)
            grouped_x_initial.append(torch.cat(list_clauses, dim=1))

        for i in range(math.ceil(float(len(grouped_x_initial)) / BC)):
            clause = grouped_x_initial[i * BC: (i + 1) * BC]
            BC_iter = len(clause) # we compute this because last iteration might be less than BC
            # stack clauses in dim 0
            clause = torch.cat(clause, dim=0)

            # repeat mask_clause and x for each clause similarly in dim 0
            #mask_clause_rep = torch.repeat_interleave(mask_clause, BC_iter, dim=0)
            x_rep = x.repeat(BC_iter, 1)

            # NOTE: clauses start at 1. Entries with 0 values are padding values.
            # we set them to 1 to avoid index out of bounds. The 'mask_clause' later assures that 
            # these values are not considered in the energy calculation
            clause[clause == 0] = 1
            idx = torch.abs(clause) - 1
            #mask_clause_i = mask_clause_rep[:, i*BC: (i+1)*BC]
            #if idx[mask_clause_i].shape[0] == 0:
            #    continue

            ##x_e = torch.gather(x, dim=1, index=idx)
            batch_indices = torch.arange(x_rep.shape[0], device=x.device).unsqueeze(1)  # Row indices for batch
            x_e = x_rep[batch_indices, idx]  # Fetch values without using gather

            sign = torch.sign(clause)
            sign[sign == 1] = 0.0
            sign[sign == -1] = 1.0

            t_rep = t.repeat(BC_iter)
            t_annealed_rep = t_annealed.repeat(BC_iter)

            e = self.net.neg_logp_unnorm(x_e, t_rep, t_annealed_rep, x_initial=sign) # shape: (Batch,)

            if not self.training: # This is just done for memory efficiency
                e = e.detach()
            e = e[:, None].repeat(1, clause.shape[1])
            assert e.shape == clause.shape

            # reshape e and idx to (Batch, 3 * BC)
            e = torch.split(e, batch_size, dim=0)
            e = torch.cat(e, dim=1)
            idx = torch.split(idx, batch_size, dim=0)
            idx = torch.cat(idx, dim=1)
            mask_clause_i = mask_clause[:, i*BC: (i+1)*BC]
            mask_clause_i = torch.repeat_interleave(mask_clause_i, 3, dim=1)

            update = e
            count_update = torch.ones_like(idx).float()

            update[~mask_clause_i] = 0.0
            count_update[~mask_clause_i] = 0.0

            batch_indices = torch.arange(x.shape[0], device=x.device)
            #for k in range(3): # 3-sat has 3 entries per clause
            #    energy[batch_indices, idx[:, k]] += update[:, k]
            #    count[batch_indices, idx[:, k]] += count_update[:, k]
            energy.scatter_add_(dim=1, index=idx, src=update)
            count.scatter_add_(dim=1, index=idx, src=count_update)

        # if count is all zero then its probably a bug, raise an error
        assert count.sum() > 0, "Count of different variables cannot be zero in EBMSAT"

        # filter zero entries from count (and energy)
        idx = count == 0
        count[idx] = 1.0
        energy[idx] = 0.0

        energy = energy / count
        return energy.sum(dim=-1)                        

    def forward(self, x, t, t_annealed, x_initial, mask_clause):
        energy = torch.zeros_like(x)
        count = torch.zeros_like(x)
        
        BC = self.batch_clauses
        batch_size = x.shape[0]

        # before proceeding, group the clauses according to models num_clauses
        grouped_x_initial = []
        for i in range(0, len(x_initial), self.num_clauses):
            list_clauses = x_initial[i: i + self.num_clauses]
            if len(list_clauses) < self.num_clauses:
                for _ in range(self.num_clauses - len(list_clauses)):
                    # pad with zero-clauses
                    list_clauses.append(torch.zeros_like(list_clauses[0]))
                    # add False column to mask_clause
                    mask_clause = torch.cat([mask_clause, torch.zeros_like(mask_clause[:, 0:1])], dim=1)
            grouped_x_initial.append(torch.cat(list_clauses, dim=1))

        for i in range(math.ceil(float(len(grouped_x_initial)) / BC)):
            clause = grouped_x_initial[i * BC: (i + 1) * BC]
            BC_iter = len(clause) # we compute this because last iteration might be less than BC

            clause = torch.cat(clause, dim=0)

            x_rep = x.repeat(BC_iter, 1)

            # NOTE: clauses start at 1. Entries with 0 values are padding values.
            # we set them to 1 to avoid index out of bounds. The 'mask_clause' later assures that 
            # these values are not considered in the energy calculation
            clause[clause == 0] = 1
            idx = torch.abs(clause) - 1

            batch_indices = torch.arange(x_rep.shape[0], device=x.device).unsqueeze(1)  # Row indices for batch
            x_e = x_rep[batch_indices, idx]  # Fetch values without using gather

            sign = torch.sign(clause)
            sign[sign == 1] = 0.0
            sign[sign == -1] = 1.0

            t_rep = t.repeat(BC_iter)
            t_annealed_rep = t_annealed.repeat(BC_iter)

            e = self.net(x_e, t_rep, t_annealed_rep, x_initial=sign) # shape: (Batch,)
            if not self.training: # This is just done for memory efficiency
                e = e.detach()
            assert e.shape == clause.shape

            # reshape e and idx to (Batch, 3 * BC)
            e = torch.split(e, batch_size, dim=0)
            e = torch.cat(e, dim=1)
            idx = torch.split(idx, batch_size, dim=0)
            idx = torch.cat(idx, dim=1)
            mask_clause_i = mask_clause[:, i*BC: (i+1)*BC]
            mask_clause_i = torch.repeat_interleave(mask_clause_i, 3, dim=1)

            update = e
            count_update = torch.ones_like(idx).float()

            update[~mask_clause_i] = 0.0
            count_update[~mask_clause_i] = 0.0

            energy.scatter_add_(dim=1, index=idx, src=update)
            count.scatter_add_(dim=1, index=idx, src=count_update)

        idx = count == 0
        count[idx] = 1.0
        energy[idx] = 0.0

        energy = energy / count
        return energy
    

class EnergyMessagePassing(MessagePassing):
    def __init__(self, ebm):
        super().__init__(aggr='mean')
        self.ebm = ebm

    def forward(self, x, edge_index, t):
        row, col = edge_index  # source: row, target: col

        src_feats = x[row]  # [num_edges, in_channels]
        tgt_feats = x[col]  # [num_edges, in_channels]

        edge_input = torch.cat([src_feats, tgt_feats], dim=1)  # [num_edges, in_channels * 2]
        edge_output = self.ebm.neg_logp_unnorm(edge_input, t, x_initial=None)

        src_update = edge_output
        tgt_update = edge_output

        num_nodes = x.size(0)

        update = torch.cat([src_update, tgt_update], dim=0)
        idx = torch.cat([row, col], dim=0)

        node_updates = scatter_add(update, idx, dim=0, dim_size=num_nodes)

        return node_updates
    

class GradEnergyMessagePassing(nn.Module):
    def __init__(self, ebm):
        super().__init__()
        self.ebm = ebm

    def forward(self, x, edge_index, t):
        row, col = edge_index  # source: row, target: col

        src_feats = x[row]  # [num_edges, in_channels]
        tgt_feats = x[col]  # [num_edges, in_channels]

        edge_input = torch.cat([src_feats, tgt_feats], dim=1)  # [num_edges, in_channels * 2]
        edge_output = self.ebm(edge_input, t, x_initial=None)

        out_channels = edge_output.size(1) // 2
        src_update = edge_output[:, :out_channels]
        tgt_update = edge_output[:, out_channels:]

        num_nodes = x.size(0)

        update = torch.cat([src_update, tgt_update], dim=0)
        idx = torch.cat([row, col], dim=0)

        node_updates = scatter_add(update, idx, dim=0, dim_size=num_nodes)

        return node_updates


class EBMColorsGraph(nn.Module):
    def __init__(self,
                 net,
                 diffusion,
                 num_colors,
                 batch_graphs):
        super(EBMColorsGraph, self).__init__()
        self.net = net
        self.diffusion = diffusion
        self.num_colors = num_colors
        self.neg_logp_mp = EnergyMessagePassing(ebm=self.net)
        self.grad_mp = GradEnergyMessagePassing(ebm=self.net)
        self.batch_graphs = batch_graphs

    def neg_logp_unnorm(self, x, t, x_initial):
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_colors)
        x_initial = x_initial.reshape(batch_size, -1, 2)
        data_list = []
        for i in range(batch_size):
            data_list.append(
                Data(
                    x=x[i], 
                    edge_index=x_initial[i].t(),
                    t=t[i].repeat(x_initial[i].shape[0]),
                    i=i
                )
            )
        energy = torch.zeros(batch_size, device=x.device)
        for i in range(0, len(data_list), self.batch_graphs):
            chunk = data_list[i:i+self.batch_graphs]
            batch = Batch.from_data_list(chunk)
            batch.x = self.neg_logp_mp(batch.x, batch.edge_index, batch.t)
            updated_list = batch.to_data_list()

            for j, graph in enumerate(updated_list):
                energy[graph.i] = graph.x.sum().detach()
        return energy
                    
    
    def forward(self, x, t, x_initial):
        create_graph = True if self.training else False

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_colors)
        x_initial = x_initial.reshape(batch_size, -1, 2)
        data_list = []
        for i in range(batch_size):
            data_list.append(
                Data(
                    x=x[i], 
                    edge_index=x_initial[i].t(),
                    t=t[i].repeat(x_initial[i].shape[0]),
                    i=i
                )
            )
        grad = torch.zeros_like(x)
        for i in range(0, len(data_list), self.batch_graphs):
            chunk = data_list[i:i+self.batch_graphs]
            batch = Batch.from_data_list(chunk)
            batch.x.requires_grad_(requires_grad=True)
            e = self.neg_logp_mp(batch.x, batch.edge_index, batch.t)
            batch.x = torch.autograd.grad(outputs=e.sum(), inputs=batch.x, create_graph=create_graph)[0]
            batch.x = batch.x.detach()
            updated_list = batch.to_data_list()

            for j, graph in enumerate(updated_list):
                grad[graph.i] = graph.x
        grad = grad.reshape(batch_size, -1)

        tmp = torch.tensor(1.0 - self.diffusion.alphas_cumprod, dtype=torch.float, device=x.device)
        tmp = tmp[t]
        grad = grad * tmp[:, None]
        
        return grad
    

class EBMReducedColorsGraph(nn.Module):
    def __init__(self,
                 net,
                 base_dimension,
                 reduced_dimension):
        super(EBMReducedColorsGraph, self).__init__()
        self.net = net
        self.base_dimension = base_dimension
        self.reduced_dimension = reduced_dimension

    def neg_logp_unnorm(self, x, t, x_initial):
        return self.net.neg_logp_unnorm(x, t, x_initial)      
    
    def forward(self, x, t, x_initial):
        grad = self.net(x, t, x_initial)
        return grad


class EBMDiffReducedColorsModel(nn.Module):
    def __init__(self,
                 net,
                 base_dimension,
                 reduced_dimension):
        super(EBMDiffReducedColorsModel, self).__init__()
        self.net = net
        self.base_dimension = base_dimension
        self.reduced_dimension = reduced_dimension

    def neg_logp_unnorm(self, x, t, x_initial=None):
        x_clone = x.clone()
        # pad x with one hot solutions
        if x_clone.shape[1] < self.base_dimension:
            x_clone = x_clone.reshape(x_clone.shape[0], self.reduced_dimension)
            # pad with -1 until it reaches the base dimension
            ones = torch.zeros((x_clone.shape[0], self.base_dimension - self.reduced_dimension), device=x_clone.device)
            x_clone = torch.cat([x_clone, ones], dim=1)
            x_clone = x_clone.reshape(x_clone.shape[0], -1)
        energy_score = self.net(x_clone, t, x_initial=x_initial)
        res = torch.norm(energy_score, dim=-1)
        return res

    def forward(self, x, t, x_initial=None):
        torch.set_grad_enabled(True)

        x_clone = x.clone()
        # pad x with one hot solutions
        if x_clone.shape[1] < self.base_dimension:
            x_clone = x_clone.reshape(x_clone.shape[0], self.reduced_dimension)
            # pad with -1 until it reaches the base dimension
            ones = torch.zeros((x_clone.shape[0], self.base_dimension - self.reduced_dimension), device=x_clone.device)
            x_clone = torch.cat([x_clone, ones], dim=1)

        if self.training:
            create_graph = True
        else:
            create_graph = False

        x_clone.requires_grad_(requires_grad=True)
        out = self.neg_logp_unnorm(x_clone, t, x_initial=x_initial)
        epsilon = torch.autograd.grad(outputs=out.sum(), inputs=x_clone, create_graph=create_graph)
        # crop only the first reduced_dimension one hot solutions
        epsilon = epsilon[0]
        epsilon = epsilon.reshape(epsilon.shape[0], self.base_dimension)
        epsilon = epsilon[:, :self.reduced_dimension]

        return epsilon


class EdgeBased_EBMDiffReducedColorsModel(nn.Module):
    def __init__(self,
                 net,
                 base_dimension,
                 reduced_dimension):
        super(EdgeBased_EBMDiffReducedColorsModel, self).__init__()
        self.net = net
        self.base_dimension = base_dimension
        self.reduced_dimension = reduced_dimension

    def neg_logp_unnorm(self, x, t, x_initial=None):
        x_clone = x.clone()
        # pad x with one hot solutions
        if x_clone.shape[1] < (self.base_dimension * 2):
            x_clone = x_clone.reshape(x_clone.shape[0], -1, self.reduced_dimension)
            # pad with -1 until it reaches the base dimension
            ones = torch.ones((x_clone.shape[0], x_clone.shape[1], self.base_dimension - self.reduced_dimension), device=x_clone.device)
            ones = ones * 0.0
            x_clone = torch.cat([x_clone, ones], dim=2)
            x_clone = x_clone.reshape(x_clone.shape[0], -1)
        energy_score = self.net(x_clone, t, x_initial=x_initial)
        res = torch.norm(energy_score, dim=-1)
        return res

    def forward(self, x, t, x_initial=None):
        torch.set_grad_enabled(True)

        x_clone = x.clone()
        # pad x with one hot solutions
        if x_clone.shape[1] < (self.base_dimension * 2):
            x_clone = x_clone.reshape(x_clone.shape[0], -1, self.reduced_dimension)
            # pad with -1 until it reaches the base dimension
            ones = torch.ones((x_clone.shape[0], x_clone.shape[1], self.base_dimension - self.reduced_dimension), device=x_clone.device)
            ones = ones * 0.0
            x_clone = torch.cat([x_clone, ones], dim=2)
            x_clone = x_clone.reshape(x_clone.shape[0], -1)

        if self.training:
            create_graph = True
        else:
            create_graph = False

        x_clone.requires_grad_(requires_grad=True)
        out = self.neg_logp_unnorm(x_clone, t, x_initial=x_initial)
        epsilon = torch.autograd.grad(outputs=out.sum(), inputs=x_clone, create_graph=create_graph)
        # crop only the first reduced_dimension one hot solutions
        epsilon = epsilon[0]
        epsilon = epsilon.reshape(epsilon.shape[0], -1, self.base_dimension)
        epsilon = epsilon[:, :, :self.reduced_dimension]
        epsilon = epsilon.reshape(epsilon.shape[0], -1)

        return epsilon

        
class EBMCrossWorld(nn.Module):
    def __init__(self,
                 net,
                 dimension,
                 vocab_dim,
                 embedding_dim,
                 batch_size=1):
        super(EBMCrossWorld, self).__init__()
        self.net = net
        self.dimension = dimension
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def neg_logp_unnorm(self, x, t, x_initial=None):
        assert x.shape[0] % self.batch_size == 0, "Batch size must be divisible by the number of batches"

        def repeat_fn(t, num):
            return t.repeat(num, *[1] * (t.ndim - 1))
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.dimension, self.dimension, self.vocab_dim)
        e = torch.zeros((batch_size,), device=x.device)

        emb = x_initial['h_words']
        emb_b = repeat_fn(emb, self.batch_size)

        for b in range(0, batch_size, self.batch_size):
            x_b = x[b:b + self.batch_size]
            x_b = x_b.reshape(self.dimension * self.batch_size, -1)
            t_b = t[b:b + self.batch_size]
            t_b = torch.repeat_interleave(t_b, self.dimension, dim=0)
            g = self.net.neg_logp_unnorm(x_b, t_b, x_initial=emb_b)
            g = g.reshape(self.batch_size, self.dimension)
            e[b:b + self.batch_size] += g.sum(dim=-1)

        x = x.permute(0, 2, 1, 3)

        emb = x_initial['v_words']
        emb_b = repeat_fn(emb, self.batch_size)

        for b in range(0, batch_size, self.batch_size):
            x_b = x[b:b + self.batch_size]
            x_b = x_b.reshape(self.dimension * self.batch_size, -1)
            t_b = t[b:b + self.batch_size]
            t_b = torch.repeat_interleave(t_b, self.dimension, dim=0)
            g = self.net.neg_logp_unnorm(x_b, t_b, x_initial=emb_b)
            g = g.reshape(self.batch_size, self.dimension)
            e[b:b + self.batch_size] += g.sum(dim=-1)
            
        x = x.permute(0, 2, 1, 3)
        return e
    
    def forward(self, x, t, x_initial=None):
        assert x.shape[0] % self.batch_size == 0, "Batch size must be divisible by the number of batches"

        def repeat_fn(t, num):
            return t.repeat(num, *[1] * (t.ndim - 1))

        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.dimension, self.dimension, self.vocab_dim)
        grad = torch.zeros_like(x)

        emb = x_initial['h_words']
        emb_b = repeat_fn(emb, self.batch_size)

        for b in range(0, batch_size, self.batch_size):
            x_b = x[b:b + self.batch_size]
            x_b = x_b.reshape(self.dimension * self.batch_size, -1)
            t_b = t[b:b + self.batch_size]
            t_b = torch.repeat_interleave(t_b, self.dimension, dim=0)
            g = self.net(x_b, t_b, x_initial=emb_b)
            g = g.reshape(self.batch_size, self.dimension, self.dimension, self.vocab_dim)
            grad[b:b + self.batch_size] += g

        x = x.permute(0, 2, 1, 3)
        grad = grad.permute(0, 2, 1, 3)

        emb = x_initial['v_words']
        emb_b = repeat_fn(emb, self.batch_size)

        for b in range(0, batch_size, self.batch_size):
            x_b = x[b:b + self.batch_size]
            x_b = x_b.reshape(self.dimension * self.batch_size, -1)
            t_b = t[b:b + self.batch_size]
            t_b = torch.repeat_interleave(t_b, self.dimension, dim=0)
            g = self.net(x_b, t_b, x_initial=emb_b)
            g = g.reshape(self.batch_size, self.dimension, self.dimension, self.vocab_dim)
            grad[b:b + self.batch_size] += g

        x = x.permute(0, 2, 1, 3)
        grad = grad.permute(0, 2, 1, 3)
        grad = grad.reshape(batch_size, -1)
        return grad / 2.0
