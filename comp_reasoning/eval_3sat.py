import torch
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW

from comp_reasoning.models import MLPModel, EBMDiffModel, EBMSAT
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.sampling import p_sample_loop, pem_sampling
from comp_reasoning.data.sat import DimacsSATDataset

from comp_reasoning.diffusion.anneal_samplers import AnnealedMALASampler, AnnealedCHASampler, AnnealedUHASampler,AnnealedULASampler
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)


INPUT_CLAUSES = 1

#@hydra.main(version_base=None, config_path="./configs", config_name="eval_sat")
def experiment():

    config_path = "./comp_reasoning/configs/sat.yaml"
    params = OmegaConf.load(config_path)

    # read path from command line --path
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./3sat/train')
    parser.add_argument('--pem_num_particles', type=int, default=params.pem_num_particles)
    parser.add_argument('--device', type=str, default=params.device)
    parser.add_argument('--batch_size', type=int, default=params.batch_size)

    PATH = parser.parse_args().path
    params.pem_num_particles = parser.parse_args().pem_num_particles
    device = parser.parse_args().device
    params.batch_size = parser.parse_args().batch_size

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    options = {}
    options["learn_sigma"] = False
    options["noise_schedule"] = "linear" # linear

    base_timestep_respacing = '100'
    annealed_diffusion_steps = 10

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

    dataset = DimacsSATDataset(PATH, is_train=False)
    print("Dataset size: ", len(dataset))
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    N = 3 # 3 SAT PROBLEM
    mlp = MLPModel(
        params.num_steps,
        n_steps_annealed=annealed_diffusion_steps,
        n_layers=params.model.n_layers,
        x_dim=(N * INPUT_CLAUSES) + (N * INPUT_CLAUSES),
        h_dim=params.model.h_dim,
        out_dim=1,
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape)
    ebm_ = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm_,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    state_dict = torch.load(params.chk_model, map_location=device)
    # if net.net.net in state_dict replace it with net.net
    for key in list(state_dict.keys()):
        if 'net.net.net' in key:
            new_key = key.replace('net.net.net', 'net.net')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

    ebm_model = EBMSAT(ebm_, batch_clauses=params.batch_clauses, num_clauses=INPUT_CLAUSES)
    model = PortableDiffusionModel(
        ebm_model,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )

    model.to(device)
    model.eval()

    def is_satisfied(clauses, values):
        """
        Check if the given SAT problem is satisfied by the provided variable values.
        
        :param clauses: List of clauses, where each clause is a list of integers representing literals.
        :param values: PyTorch tensor of shape (n,) containing 0 or 1 for each variable (1-based index).
        :return: True if the SAT formula is satisfied, False otherwise.
        """
        for clause in clauses:
            clause_satisfied = False
            for literal in clause:
                var_index = abs(literal) - 1  # Convert 1-based index to 0-based
                var_value = values[var_index].item()
                if (literal > 0 and var_value == 1) or (literal < 0 and var_value == 0):
                    clause_satisfied = True
                    break  
            if not clause_satisfied:
                return False
        return True  # All clauses are satisfied

    def number_of_satisfied_clauses(clauses, values, mask_clause):
        """
        Count the number of clauses satisfied by the provided variable values.
        
        :param clauses: List of clauses, where each clause is a list of integers representing literals.
        :param values: PyTorch tensor of shape (n,) containing 0 or 1 for each variable (1-based index).
        :return: Number of clauses satisfied by the variable values.
        """
        num_satisfied = 0
        num_total = 0
        for clause in clauses:
            if not mask_clause[num_total]:
                continue
            num_total += 1
            for literal in clause:
                var_index = abs(literal) - 1  # Convert 1-based index to 0-based
                var_value = values[var_index].item()
                if (literal > 0 and var_value == 1) or (literal < 0 and var_value == 0):
                    num_satisfied += 1
                    break 
        return num_satisfied, num_total

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
        #t = num_steps - (torch.ones((x.shape[0],), dtype=torch.int) * t) - 1
        return -1 * model.p_gradient(x, t, **kwargs).detach()

    def energy_function(x, t, **kwargs):
        #t = num_steps - (torch.ones((x.shape[0],), dtype=torch.int) * t) - 1
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

    correct = []
    avg_sat_clauses = []

    for i, (clause, mask_clause, solution, mask_var, _) in enumerate(loader):
        
        # clone because its modified later
        clauses_clone = []
        for j in range(len(clause)):
            clauses_clone.append(clause[j].clone())
        mask_clause_clone = mask_clause.clone()

        num_samples = mask_clause.shape[0]
        num_variables = mask_var.shape[1]

        # tensors to device
        for j in range(len(clause)):
            clause[j] = clause[j].to(device)
        mask_clause = mask_clause.to(device)
        for j in range(len(clauses_clone)):
            clauses_clone[j] = clauses_clone[j].to(device)
        mask_clause_clone = mask_clause_clone.to(device)
        mask_var = mask_var.to(device)
        
        batch_size = mask_clause.shape[0]
        timesteps = torch.zeros((batch_size, )).long()
        timesteps_annealed = torch.zeros((batch_size, )).long()
        timesteps = timesteps.to(device)
        timesteps_annealed = timesteps_annealed.to(device)
        
        if sampler_type in ('reverse_diffusion', 'ULA', 'MALA', 'UHMC', 'HMC'):
            x_sampled = p_sample_loop(
                    diffusion,
                    sampler=sampler,
                    model=model,
                    shape=(num_samples, num_variables),
                    annealed_timestep=timesteps_annealed,
                    model_kwargs={
                        'x_initial': clauses_clone,
                        't_annealed': timesteps_annealed,
                        'mask_clause': mask_clause,
                    },
                    device=torch.device(device),
                )
        
        elif sampler_type == 'pem':
            x_sampled = pem_sampling(
                    diffusion,
                    sampler=None,
                    model=model,
                    shape=(num_samples, num_variables),
                    annealed_timestep=timesteps_annealed,
                    pem_num_particles=params.pem_num_particles,
                    model_kwargs={
                        'x_initial': clauses_clone,
                        't_annealed': timesteps_annealed,
                        'mask_clause': mask_clause_clone,
                    },
                    device=torch.device(device),
                    interleave_repeat=True,
            )
        elif sampler_type == 'random':
            x_sampled = torch.rand((num_samples, num_variables)).to(device)
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")
        
        print("min max: ", x_sampled.min(), x_sampled.max())
        rounded_x_sampled = torch.round(x_sampled.clamp(0, 1)).int()

        batch_size = x_sampled.shape[0]
        for j in range(batch_size): # for each batch

            # check if instance j is correct
            correct_instance = is_satisfied(
                [s[j] for s in clause],
                rounded_x_sampled[j]
            )
            if correct_instance:
                correct.append(correct_instance)

            # compute number of satisfied clauses 
            num_satisfied, num_total = number_of_satisfied_clauses(
                [s[j] for s in clause],
                rounded_x_sampled[j],
                mask_clause[j]
            )
            avg_sat_clauses.append((num_satisfied, num_total))
        
        print(i, batch_size)
        print(" --> ", len(correct), "/", len(avg_sat_clauses), float(len(correct)) / len(avg_sat_clauses))
        avgs = [float(x1) / x2 for x1, x2 in avg_sat_clauses]
        print(" --> ", float(sum(avgs)) / len(avgs))

        e = model.net.neg_logp_unnorm(
            x_sampled,
            timesteps,
            timesteps_annealed,
            x_initial=clause,
            mask_clause=mask_clause,
        )
        print("Average solution energy: ", e.mean().item())
        print()

    print(" Final --> ", len(correct), "/", len(avg_sat_clauses), float(len(correct)) / len(avg_sat_clauses))
    print(" Final --> ", float(sum(avgs)) / len(avgs))
    print(" Final std: ", np.std(np.array(avgs)))



if __name__ == "__main__":
    experiment()