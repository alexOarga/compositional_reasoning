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
from comp_reasoning.data.crossword import CrosswordRowDataset, idxs_to_uppercase_string

from comp_reasoning.diffusion.anneal_samplers import AnnealedMALASampler, AnnealedCHASampler, AnnealedUHASampler,AnnealedULASampler
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)


#@hydra.main(version_base=None, config_path="./configs", config_name="crosswords")
def experiment():

    config_path = "./comp_reasoning/configs/crosswords_mini.yaml"
    params = OmegaConf.load(config_path)
    device = params.device

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

    train_dataset = CrosswordRowDataset(params.train_file)
    dataset = CrosswordRowDataset(params.test_file)
    train_vocab = train_dataset.vocabulary()
    test_vocab = dataset.vocabulary()
    print("Dataset size: ", len(dataset))
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

    mlp = MLPModel(
        params.num_steps,
        n_steps_annealed=params.annealed_steps,
        n_layers=params.model.n_layers,
        x_dim=params.embedding_dim + (params.vocab_size * 5),
        h_dim=params.model.h_dim,
        out_dim=params.embedding_dim + (params.vocab_size * 5),
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape
    )
    ebm_ = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm_,
        diffusion, 
        annealed_diffusion, 
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    model.load_state_dict(torch.load(params.chk_model, map_location=device))
    model.to(device)
    model.eval()


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

    for i, (target, embedding) in enumerate(loader):
        
        target = target.float().to(device)
        embedding = embedding.float().to(device)
        
        batch_size = target.shape[0]
        num_variables = target.shape[1]
        
        timesteps = torch.zeros((batch_size, )).long()
        timesteps_annealed = torch.zeros((batch_size, )).long()
        timesteps = timesteps.to(device)
        timesteps_annealed = timesteps_annealed.to(device)
        
        if sampler_type in ('reverse_diffusion', 'ULA', 'MALA', 'UHMC', 'HMC'):
            x_sampled = p_sample_loop(
                    diffusion,
                    sampler=sampler,
                    model=model,
                    shape=(batch_size, num_variables),
                    annealed_timestep=timesteps_annealed,
                    model_kwargs={
                        'x_initial': embedding,
                    },
                    device=torch.device(device),
                )
        
        elif sampler_type == 'pem':
            x_sampled = pem_sampling(
                    diffusion,
                    sampler=None,
                    model=model,
                    shape=(batch_size, num_variables),
                    annealed_timestep=timesteps_annealed,
                    pem_num_particles=params.pem_num_particles,
                    model_kwargs={
                        'x_initial': embedding,
                    },
                    device=torch.device(device),
                    interleave_repeat=True,
            )
        elif sampler_type == 'random':
            x_sampled = torch.rand((batch_size, num_variables)).to(device)
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")
        
        x_sampled = x_sampled.reshape(batch_size, 5, params.vocab_size)
        pred_sample = torch.argmax(x_sampled, dim=-1)
        target_sample = target.reshape(batch_size, 5, params.vocab_size)
        target_sample = torch.argmax(target_sample, dim=-1)
        for ii in range(batch_size):    
            target_word = idxs_to_uppercase_string(target_sample[ii].cpu().numpy())
            word = idxs_to_uppercase_string(pred_sample[ii].cpu().numpy())
            is_in_train_vocab = target_word in train_vocab
            print(f"Sampled word: {word}, Target word: {target_word}, In train vocab: {is_in_train_vocab}")


if __name__ == "__main__":
    experiment()