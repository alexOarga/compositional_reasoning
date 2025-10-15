import sys
import math
import gc
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from comp_reasoning.models import MLPModel, EBMDiffModel
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.schedulers import get_lr
from comp_reasoning.data.nqueens import NQueensRowsDataset, NQueensDiagonalsDataset

from comp_reasoning.diffusion.anneal_samplers import (
    AnnealedMALASampler, AnnealedCHASampler,
    AnnealedUHASampler, AnnealedULASampler
)
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from comp_reasoning.diffusion.composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)


def experiment():
    config_path = "./comp_reasoning/configs/nqueens_rows.yaml"
    params = OmegaConf.load(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default='./comp_reasoning/data/nqueens/8_queens_train.txt')
    parser.add_argument('--valid-path', type=str, default='./comp_reasoning/data/nqueens/8_queens_valid.txt')
    parser.add_argument('--rows', action='store_true', default=params.rows)
    parser.add_argument('--diags', action='store_true', default=params.diags)
    args = parser.parse_args()

    params.rows = args.rows
    params.diags = args.diags
    train_path = args.train_path
    valid_path = args.valid_path

    if params.rows and params.diags:
        raise ValueError("Cannot specify both rows and diags.")
    
    device = params.device

    options = {
        "learn_sigma": False,
        "noise_schedule": "linear",
    }
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

    if params.rows:
        dataset_cls = NQueensRowsDataset
    elif params.diags:
        dataset_cls = NQueensDiagonalsDataset
    else:
        raise ValueError("Specify either --rows or --diags.")

    train_dataset = dataset_cls(train_path, split='all')
    valid_dataset = dataset_cls(valid_path, split='all')

    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=params.batch_size)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=params.batch_size,
        sampler=RandomSampler(valid_dataset, replacement=True, num_samples=params.batch_size)
    )

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))

    N = train_dataset.N
    mlp = MLPModel(
        params.num_steps,
        n_steps_annealed=params.annealed_steps,
        n_layers=params.model.n_layers,
        x_dim=N,
        h_dim=params.model.h_dim,
        out_dim=1,
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape
    )
    ebm_model = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm_model,
        diffusion,
        annealed_diffusion,
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def run_epoch(loader, model, optimizer=None):
        is_train = optimizer is not None
        total_loss, total_mse, total_energy, count = 0.0, 0.0, 0.0, 0

        for target, negative in tqdm(loader):
            model.train() if is_train else model.eval()

            target = target.float().to(device)
            negative = negative.float().to(device)
            t_main = torch.randint(0, len(diffusion.betas) - 1, (target.shape[0],)).to(device)
            t_annealed = torch.randint(0, len(annealed_diffusion.betas) - 1, (target.shape[0],)).to(device)

            if is_train:
                model.zero_grad()

            loss_mse, loss_energy = model.loss(
                target, t_main, t_annealed,
                x_initial=None, x_0_neg=negative
            )

            loss = 0.0
            if params.diffusion:
                loss += loss_mse * params.diffusion_loss_scale
            if params.contrastive:
                loss += loss_energy * params.contrastive_loss_scale

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item() if params.diffusion else 0.0
            total_energy += (loss_energy * params.contrastive_loss_scale).item() if params.contrastive else 0.0
            count += 1

        return total_loss / count, total_mse / count, total_energy / count
    
    loss_train, loss_val = [], []
    best_val_loss = math.inf

    name = f'comp_reasoning/pk/ebm_{N}-queens'
    name += '_rows' if params.rows else ''
    name += '_diags' if params.diags else ''
    name += '_annealed' if params.annealed_landscape else ''
    name += '_diffusion' if params.diffusion else ''
    name += '_contrastive' if params.contrastive else ''
    pk_path = './' + name + '.pk'

    for epoch in range(params.epochs):
        train_loss, train_mse, train_energy = run_epoch(train_loader, model, optimizer)
        scheduler.step(epoch)
        print(f"Train epoch {epoch}: Loss={train_loss:.4f}, MSE={train_mse:.4f}, Energy={train_energy:.4f}, LR={get_lr(scheduler, epoch):.6f}")

        #torch.cuda.empty_cache(); gc.collect()

        val_loss, val_mse, val_energy = run_epoch(valid_loader, model)
        print(f"Val epoch {epoch}: Loss={val_loss:.4f}, MSE={val_mse:.4f}, Energy={val_energy:.4f}")

        loss_train.append(train_loss)
        loss_val.append(val_loss)

        if val_loss < best_val_loss:
            print("Saving new best model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), pk_path)
            torch.save(optimizer.state_dict(), "optimizer.pt")


if __name__ == "__main__":
    experiment()