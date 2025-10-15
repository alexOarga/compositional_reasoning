import torch
import sys
import argparse
import random
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

from comp_reasoning.models import ColorMLPModel, EBMDiffModel, EBMColorsGraph, EdgeBased_EBMDiffReducedColorsModel, EBMReducedColorsGraph, MLPModel
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
    num_nodes, edge_array, chrom_number = parse_graph_file(params.graph_file)
    edge_array = torch.tensor(edge_array, device=device)
    print("Num nodes:", num_nodes, " Num edges:", edge_array.shape[0])

    # 1. Load Model
    device = params.device

    mlp = MLPModel(
        params.num_steps,
        n_steps_annealed=params.annealed_steps,
        n_layers=params.model.n_layers,
        x_dim=params.num_colors * 2,
        h_dim=params.model.h_dim,
        out_dim=1,
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape)
    if chrom_number == params.num_colors:
        ebm_model = EBMDiffModel(mlp)
    else:
        ebm_model = EdgeBased_EBMDiffReducedColorsModel(mlp, params.num_colors, chrom_number)
    model = PortableDiffusionModel(
        ebm_model,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    model.load_state_dict(torch.load(params.pk_path, map_location=device))
    
    ebm_ = EBMColorsGraph(ebm_model, diffusion, num_colors=chrom_number, batch_graphs=params.batch_graphs)
    if chrom_number != params.num_colors:
        ebm_ = EBMReducedColorsGraph(ebm_, params.num_colors, chrom_number)
    model = PortableDiffusionModel(
        ebm_,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    
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
        if params.sampling == 'random':
            x_sampled = torch.rand((1, num_nodes * chrom_number), device=device)
        elif params.sampling in ['reverse_diffusion', 'ULA', 'MALA', 'UHMC', 'HMC']:
            x_sampled = p_sample_loop(
                diffusion,
                sampler=sampler,
                model=model,
                shape=(1, num_nodes * chrom_number),
                annealed_timestep=t_annealed,
                model_kwargs={
                    'x_initial': edge_array,
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
                shape=(1, num_nodes * chrom_number),
                annealed_timestep=t_annealed,
                pem_num_particles=params.pem_num_particles,
                model_kwargs={
                    'x_initial': edge_array,
                    #'t_annealed': t_annealed,
                },
                device=torch.device(device),
            )
        else:
            raise ValueError(f"Unknown sampling method: {params.sampling}")

        x_sampled = x_sampled.view(x_sampled.shape[0], -1, chrom_number)
        print(x_sampled[0, 0])
        x_sampled_copy = x_sampled.clone().reshape(x_sampled.shape[0], -1)
        x_sampled = torch.argmax(x_sampled, dim=2)
        x_sampled = x_sampled.cpu().detach()
        edge_array = edge_array.cpu().detach()
        inco_count = 0
        f_edge = set([])
        for ee in edge_array:
            e1 = ee[0].item()
            e2 = ee[1].item()
            if (e1, e2) not in f_edge and (e2, e1) not in f_edge:
                f_edge.add((e1, e2))
        f_edge = list(f_edge)
    
        print("Counting edges.. ")
        edge_array = torch.tensor(f_edge, device=device)
        for edge in edge_array:
            e1, e2 = edge
            val1, val2 = x_sampled[0, e1], x_sampled[0, e2]
            if val1 == val2:
                #print(f">>> Incorrect: {e1} {val1} - {e2} {val2}")
                inco_count += 1
            else:
                #print(f"correct {e1} {val1} - {e2} {val2}")
                pass # just for debugging

        # Initialize
        x_colors = x_sampled[0].clone()  # shape: (num_nodes,)
        new_color = chrom_number + 1

        print("num conflicting edges: ", inco_count)
        print("Resolving conflicts..")
        x_colors = x_colors.to(device)
        edge_array = edge_array.to(x_colors.device)

        while True:
            e1s = edge_array[:, 0]
            e2s = edge_array[:, 1]

            # Find all conflicting edges
            same_color_mask = x_colors[e1s] == x_colors[e2s]
            conflict_edges = edge_array[same_color_mask]

            print(f"Conflicts remaining: {conflict_edges.shape[0]}", new_color)
            if conflict_edges.shape[0] == 0:
                break

            recolored_nodes = set()

            for e1, e2 in conflict_edges:
                e1, e2 = e1.item(), e2.item()

                # Avoid recoloring the same node multiple times
                if e1 in recolored_nodes or e2 in recolored_nodes:
                    continue

                # Pick the one with fewer neighbors using new_color (optional heuristic)
                chosen = random.choice([e1, e2])
                x_colors[chosen] = new_color
                recolored_nodes.add(chosen)

            # Always increment color after each full conflict scan
            new_color += 1

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