import torch
import sys
import argparse
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

from comp_reasoning.models import ColorMLPModel, MLPModel, EBMDiffModel, EBMColorsGraph, EdgeBased_EBMDiffReducedColorsModel, EBMReducedColorsGraph
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.sampling import p_sample_loop, pem_sampling
from comp_reasoning.data.colors import parse_graph_file

from comp_reasoning.diffusion.anneal_samplers import AnnealedMALASampler, AnnealedCHASampler, AnnealedUHASampler,AnnealedULASampler
sys.path.append('comp_reasoning/diffusion')
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    create_gaussian_diffusion,
    Sampler_create_gaussian_diffusion,
    get_named_beta_schedule
)

options = {}
options["learn_sigma"] = False
options["noise_schedule"] = "linear" # linear

#@hydra.main(version_base=None, config_path="./configs", config_name="color")
#def experiment(params: DictConfig):
def experiment():

    config_path = "./comp_reasoning/configs/color.yaml"
    params = OmegaConf.load(config_path)

    # read path from command line --path
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', type=str, default=params.graph_file)
    parser.add_argument('--pem_num_particles', type=int, default=params.pem_num_particles)
    parser.add_argument('--device', type=str, default=params.device)
    parser.add_argument('--batch_graphs', type=int, default=params.batch_graphs)

    params.pem_num_particles = parser.parse_args().pem_num_particles
    device = parser.parse_args().device
    params.batch_graphs = parser.parse_args().batch_graphs
    params.graph_file = parser.parse_args().graph_file
    print(" >>> Graph file: ", params.graph_file)

    diffusion = Sampler_create_gaussian_diffusion(
        steps=params.num_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
        timestep_respacing=str(params.num_steps),
        )

    annealed_diffusion = Sampler_create_gaussian_diffusion(
        steps=params.annealed_steps,
        learn_sigma=options['learn_sigma'],
        noise_schedule=options['noise_schedule'],
        )

    # 2. Load Data
    chrom_number = params.num_colors

    # 1. Load Model
    device = params.device

    mlp = ColorMLPModel(
        params.num_steps,
        n_steps_annealed=params.annealed_steps,
        n_layers=params.model.n_layers,
        x_dim=params.num_colors * 2,
        h_dim=params.model.h_dim,
        out_dim=1,
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape)
    ebm_model = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm_model,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    model.load_state_dict(torch.load(params.pk_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Setup sampling
    #ULA
    la_steps = 20
    la_step_sizes = diffusion.betas * 2

    # UHMC
    num_steps = 100
    ha_steps = 10 #2, 10 # Hamiltonian steps to run
    num_leapfrog_steps = 3 # Steps to run in leapfrog
    damping_coeff = 0.9 #0.7, 0.9
    mass_diag_sqrt = diffusion.betas
    ha_step_sizes = (diffusion.betas) * 0.1 #0.1

    def gradient_function(x, t, **kwargs):
        return -1 * model.p_gradient(x, t, **kwargs).detach()

    def energy_function(x, t, **kwargs):
        energy = -1 * model.p_energy(x, t, **kwargs).detach()
        grad = -1 * model.p_gradient(x, t, **kwargs).detach()
        return energy, grad

    sampler_type = params.sampling
    sampler = None
    if sampler_type == 'ULA':
        sampler = AnnealedULASampler(num_steps, la_steps, la_step_sizes, gradient_function)
    if sampler_type == 'MALA':
        sampler = AnnealedMALASampler(num_steps, la_steps, la_step_sizes, energy_function)
    if sampler_type == 'UHMC':
        sampler = AnnealedUHASampler(num_steps,
                        ha_steps,
                        ha_step_sizes,
                        damping_coeff,
                        mass_diag_sqrt,
                        num_leapfrog_steps,
                        gradient_function)
    if sampler_type == 'HMC':
        sampler = AnnealedCHASampler(num_steps,
                        ha_steps,
                        ha_step_sizes,
                        damping_coeff,
                        mass_diag_sqrt,
                        num_leapfrog_steps,
                        energy_function)

    avg_energy = 0.0
    for test in range(1):
        t_annealed = torch.ones(params.batch_size, device=device).long() * 0
        if params.sampling in ['reverse_diffusion', 'ULA', 'MALA', 'UHMC', 'HMC']:
            x_sampled = p_sample_loop(
                diffusion,
                sampler=sampler,
                model=model,
                shape=(1, chrom_number * 2),
                annealed_timestep=t_annealed,
                model_kwargs={
                    'x_initial': None,
                    #'t_annealed': t_annealed,
                },
                device=torch.device(device),
            )
        elif params.sampling == 'pem':
            t_annealed = torch.ones(params.batch_size, device=device).long() * 0
            x_sampled = pem_sampling(
                diffusion,
                sampler=None,
                model=model,
                shape=(1, chrom_number * 2),
                annealed_timestep=t_annealed,
                pem_num_particles=params.pem_num_particles,
                model_kwargs={
                    'x_initial': None,
                    #'t_annealed': t_annealed,
                },
                device=torch.device(device),
            )
        else:
            raise ValueError(f"Unknown sampling method: {params.sampling}")

        print(x_sampled[0, 0:chrom_number])
        print(x_sampled[0, chrom_number:])
        print(torch.round(x_sampled[0, 0:chrom_number] + 0.01).int())
        print(torch.round(x_sampled[0, chrom_number:] + 0.01).int())
        x_sampled = x_sampled.view(x_sampled.shape[0], -1, chrom_number)
        x_sampled_copy = x_sampled.clone().reshape(x_sampled.shape[0], -1)
        x_sampled = torch.argmax(x_sampled, dim=2)
 
        batch_size = x_sampled.shape[0]
        timesteps = torch.zeros((batch_size, )).long()
        timesteps_annealed = torch.zeros((batch_size, )).long()
        timesteps = timesteps.to(device)
        timesteps_annealed = timesteps_annealed.to(device)

        e = model.net.neg_logp_unnorm(
            x_sampled_copy,
            timesteps,
            #timesteps_annealed,
            x_initial=edge_array,
        )
        avg_energy += torch.mean(e).item()

        print("avg energy: ", avg_energy)
        print("cost: ", inco_count, "/", edge_array.shape[0])

if __name__ == "__main__":
    experiment()