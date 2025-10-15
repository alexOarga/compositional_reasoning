import sys
import gc
import math
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from comp_reasoning.models import MLPModel, EBMDiffModel
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.data.crossword import CrosswordRowDataset
from comp_reasoning.schedulers import get_scheduler

from comp_reasoning.diffusion.anneal_samplers import (
    AnnealedMALASampler, AnnealedCHASampler,
    AnnealedUHASampler, AnnealedULASampler
)
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from comp_reasoning.diffusion.composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)
from comp_reasoning.diffusion.composable_diffusion.sampler_gd import ModelVarType


@hydra.main(version_base=None, config_path="./configs", config_name="crosswords_mini")
def experiment(params: DictConfig):
    device = params.device

    options = {"learn_sigma": False, "noise_schedule": "linear"}

    diffusion = Sampler_create_gaussian_diffusion(
        steps=params.num_steps,
        learn_sigma=options["learn_sigma"],
        noise_schedule=options["noise_schedule"],
        timestep_respacing=str(params.num_steps),
    )

    annealed_diffusion = Sampler_create_gaussian_diffusion(
        steps=params.annealed_steps,
        learn_sigma=options["learn_sigma"],
        noise_schedule=options["noise_schedule"]
    )

    # Datasets and loaders
    train_dataset = CrosswordRowDataset(
      [
        'comp_reasoning/data/crosswords/train.pkl',
      ]
    )
    valid_dataset = CrosswordRowDataset(
      [
        'comp_reasoning/data/crosswords/valid.pkl'
      ]
    )

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")

    # Model setup
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
    ebm_model = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm_model, diffusion, annealed_diffusion,
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive
    ).to(device).train()
    model.load_state_dict(torch.load(params.chk_model, map_location=device))

    optimizer = AdamW(model.parameters(), lr=params.optimizer.lr)
    scheduler = get_scheduler(optimizer, params)

    # Output path
    suffixes = [
        '_diffusion' if params.diffusion else '',
        '_annealed' if params.annealed_landscape else '',
        '_contrastive' if params.contrastive else '',
    ]
    model_path = f"{params.output_folder}/mini_xwords{''.join(suffixes)}.pk"

    best_val_loss = math.inf
    loss_train, loss_val = [], []

    for epoch in range(params.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, is_train=True, params=params, diffusion=diffusion, annealed_diffusion=annealed_diffusion)
        train_loss = train_metrics['loss']
        loss_train.append(train_loss)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        scheduler.step(epoch)

        if epoch % 5 == 0:          
          val_metrics = run_epoch(model, valid_loader, optimizer, device, is_train=False, params=params, diffusion=diffusion, annealed_diffusion=annealed_diffusion)
          val_loss = val_metrics['loss']
          loss_val.append(val_loss)

          # Save best model
          print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
          if val_loss < best_val_loss:
              print(f"Saving best model at epoch {epoch} to {model_path}")
              best_val_loss = val_loss
              torch.save(model.state_dict(), model_path)
              torch.save(optimizer.state_dict(), "optimizer.pt")

        # Plot losses
        plt.plot(loss_train, label='train')
        plt.plot(loss_val, label='val')
        plt.legend()
        plt.savefig('comp_reasoning/out/fine_loss.png')
        plt.close()

def run_epoch(model, loader, optimizer, device, is_train, params, diffusion, annealed_diffusion):
    model.train() if is_train else model.eval()
    total_loss, total_mse, total_energy, count = 0.0, 0.0, 0.0, 0

    for data in tqdm(loader):
        
        target, embedding = data

        timesteps = torch.randint(0, len(diffusion.betas) - 1, (target.shape[0],), device=device)
        timesteps_annealed = torch.randint(0, len(annealed_diffusion.betas) - 1, (target.shape[0],), device=device)

        target = target.float().to(device)
        embedding = embedding.float().to(device)
        #negative = negative.float().to(device)

        if is_train:
            model.zero_grad()

        loss_mse, loss_energy = model.loss(
            target,
            timesteps, 
            timesteps_annealed, 
            x_initial=embedding,
            #x_0_neg=negative,
            #negative_mask=negative_mask
        )

        loss = 0.0
        if params.diffusion:
            loss += loss_mse
        if params.contrastive:
            loss += loss_energy * params.contrastive_loss_scale

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_energy += loss_energy.item() * params.contrastive_loss_scale
        count += 1

        if not is_train:
            with torch.no_grad():
                loss.detach()
                target.detach()
                loss_energy.detach()
            torch.cuda.empty_cache(); gc.collect()

    return {
        'loss': total_loss / count,
        'mse': total_mse / count,
        'energy': total_energy / count
    }


if __name__ == "__main__":
    experiment()