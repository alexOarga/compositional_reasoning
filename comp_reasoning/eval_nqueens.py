import sys
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from comp_reasoning.data.nqueens import NQueensRowsDataset
from comp_reasoning.models import EBMProduct8queens, MLPModel, EBMDiffModel, EBMDiffReducedQueensModel
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.sampling import pem_sampling, p_sample_loop
from comp_reasoning.diffusion.anneal_samplers import (
    AnnealedMALASampler, AnnealedCHASampler,
    AnnealedUHASampler, AnnealedULASampler
)
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from comp_reasoning.diffusion.composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)
from composable_diffusion.sampler_gd import ModelVarType


def load_ebm_model(config, checkpoint_path, base_dim, reduced_dim, device, diffusion, annealed_diffusion):
    mlp = MLPModel(
        n_steps=config.num_steps,
        n_steps_annealed=config.annealed_steps,
        n_layers=config.model.n_layers,
        x_dim=base_dim,
        h_dim=config.model.h_dim,
        out_dim=1,
        emb_dim=config.model.emb_dim,
        widen=config.model.widen,
        emb_type='learned',
        annealed_landscape=config.annealed_landscape,
    )
    if base_dim == reduced_dim:
        ebm = EBMDiffModel(mlp)
    else:
        ebm = EBMDiffReducedQueensModel(mlp, base_dim, reduced_dim)

    model = PortableDiffusionModel(
        ebm, 
        diffusion, 
        annealed_diffusion,
        config.annealed_landscape, 
        config.contrastive
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def is_valid_queens(tensor, dim):
    if tensor.shape != (dim, dim): return False
    if not (tensor.sum(0) == 1).all() or not (tensor.sum(1) == 1).all(): return False
    main_diags = [tensor.diagonal(i).sum() for i in range(-(dim - 1), dim)]
    anti_diags = [tensor.flip(1).diagonal(i).sum() for i in range(-(dim - 1), dim)]
    return all(x <= 1 for x in main_diags + anti_diags)


def num_valid_constraints(tensor, dim):
    row_invalid = tensor.sum(0) > 1
    col_invalid = tensor.sum(1) > 1
    main_diag_invalid = [tensor.diagonal(i).sum() > 1 for i in range(-(dim - 1), dim)]
    anti_diag_invalid = [tensor.flip(1).diagonal(i).sum() > 1 for i in range(-(dim - 1), dim)]
    total = 2 * dim + len(main_diag_invalid) + len(anti_diag_invalid)
    return total - (row_invalid.sum() + col_invalid.sum() + sum(main_diag_invalid) + sum(anti_diag_invalid))


def greedy_decode(heatmap, dim):
    heatmap = heatmap.view(-1)
    solution = torch.zeros_like(heatmap)
    _, idx = torch.sort(heatmap, descending=True)
    idx = idx[:dim]
    for i in idx:
        solution[i] = 1
        if num_valid_constraints(solution.view(dim, dim), dim) == dim * 3:
            break
        solution[i] = 0
    return solution


def initialize_test_labels(file_path):
    with open(file_path, 'r') as f:
        return {line.strip().replace('.', '0').replace('Q', '1') for line in f}


def solution_to_str(solution):
    return ''.join(str(int(x)) for x in solution)



@hydra.main(version_base=None, config_path="./configs", config_name="nqueens")
def eval(cfg: DictConfig):
    device = cfg.device

    options = {
        "learn_sigma": False,
        "noise_schedule": "linear",
    }
    diffusion = Sampler_create_gaussian_diffusion(
        steps=cfg.num_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
        timestep_respacing='100',
    )
    annealed_diffusion = Sampler_create_gaussian_diffusion(
        steps=cfg.annealed_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
    )

    N = cfg.reduced_dimension
    test_set = initialize_test_labels(cfg.test_path)

    # Load row and diagonal models
    model_rows = load_ebm_model(
        OmegaConf.load(cfg.rows_model_config), 
        cfg.rows_model, 
        cfg.base_dimension, 
        N, 
        device, 
        diffusion=diffusion, 
        annealed_diffusion=annealed_diffusion
    )
    model_diags = load_ebm_model(
        OmegaConf.load(cfg.diags_model_config), 
        cfg.diags_model, 
        cfg.base_dimension, 
        N, 
        device, 
        diffusion, 
        annealed_diffusion
    )

    # Create product model
    ebm_product = EBMProduct8queens(
        N,
        model_rows.net, model_diags.net,
        cfg.rows_model_weight, cfg.diags_model_weight
    )
    model = PortableDiffusionModel(
        ebm_product, 
        diffusion, 
        annealed_diffusion,
        cfg.annealed_landscape, 
        cfg.contrastive
    ).to(device)
    model.eval()

    # Setup sampler
    samplers = {
        'ULA': AnnealedULASampler,
        'MALA': AnnealedMALASampler,
        'UHMC': AnnealedUHASampler,
        'HMC': AnnealedCHASampler,
    }

    def grad_fn(x, t, **kwargs): return -model.p_gradient(x, t, **kwargs).detach()
    def energy_fn(x, t, **kwargs):
        e = -model.p_energy(x, t, **kwargs).detach()
        g = -model.p_gradient(x, t, **kwargs).detach()
        return e, g

    sampler_cls = samplers.get(cfg.sampling)
    sampler = None
    if sampler_cls:
        steps = 100
        la_steps = 20
        step_sizes = diffusion.betas * (2 if 'ULA' in cfg.sampling or 'MALA' in cfg.sampling else 0.1)
        sampler = sampler_cls(steps, la_steps, step_sizes, energy_fn if 'MALA' in cfg.sampling or 'HMC' in cfg.sampling else grad_fn)

    solutions, solutions_max, solutions_greedy = [], [], []

    for _ in range((100 + cfg.batch_size - 1) // cfg.batch_size):
        shape = (cfg.batch_size, N * N)
        t_annealed = torch.zeros(cfg.batch_size, device=device).long()
        if cfg.sampling == 'pem':
            x_sampled = pem_sampling(
                diffusion=diffusion,
                sampler=None,
                model=model,
                shape=shape,
                annealed_timestep=t_annealed,
                pem_num_particles=cfg.pem_num_particles,
                model_kwargs={'x_initial': None, 't_annealed': t_annealed},
                device=device
            )
        else:
            x_sampled = p_sample_loop(
                diffusion, sampler, model, shape, t_annealed,
                model_kwargs={'x_initial': None, 't_annealed': t_annealed},
                device=device
            )

        # Post-processing samples
        for x in x_sampled:
            sol_bin = (x > 0.5).float()
            greedy = greedy_decode(x, N)
            sol_rank = torch.zeros_like(x)
            sol_rank[torch.topk(x, N).indices] = 1

            solutions.append(sol_bin.view(N, N))
            solutions_max.append(sol_rank.view(N, N))
            solutions_greedy.append(greedy.view(N, N))

    def summarize(sol_list, label):
        valid = sum(is_valid_queens(s, N) for s in sol_list)
        unique = len({solution_to_str(s.view(-1)) for s in sol_list if is_valid_queens(s, N)})
        in_test = sum(solution_to_str(s.view(-1)) in test_set for s in sol_list)
        counts = [s.sum().item() for s in sol_list]
        print(f"{label} - Valid: {valid}, Unique: {unique}, In Test: {in_test}")
        print(f"{label} - Avg Queens: {np.mean(counts):.2f} (std: {np.std(counts):.2f})")

    summarize(solutions, "Raw")
    summarize(solutions_max, "Top-K")
    summarize(solutions_greedy, "Greedy")

if __name__ == "__main__":
    eval()