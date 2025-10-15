import sys
import math
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from comp_reasoning.models import ColorMLPModel, EBMDiffModel, MLPModel
from comp_reasoning.diffusion_models import PortableDiffusionModel
from comp_reasoning.schedulers import get_scheduler
from comp_reasoning.data.colors import PairDataset

from comp_reasoning.diffusion.anneal_samplers import (
    AnnealedMALASampler, AnnealedCHASampler,
    AnnealedUHASampler, AnnealedULASampler
)
sys.path.append('./comp_reasoning/diffusion') # this is a quick fix
from comp_reasoning.diffusion.composable_diffusion.model_creation import (
    Sampler_create_gaussian_diffusion,
)

# Diffusion options
options = {
    "learn_sigma": False,
    "noise_schedule": "linear",
}


def pad_colors_tensor(x, num_colors, max_colors, random_num_zeros):
    node1 = x[:, :num_colors]
    node2 = x[:, num_colors:]
    zeros = torch.zeros((x.shape[0], random_num_zeros), device=x.device)
    if random_num_zeros > 0:
        node1 = torch.cat((node1, zeros), dim=1)
        node2 = torch.cat((node2, zeros), dim=1)
    return torch.cat((node1, node2), dim=1)


@hydra.main(version_base=None, config_path="./configs", config_name="color")
def experiment(params: DictConfig):
    device = params.device

    # Create diffusion and annealed diffusion objects
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

    # Datasets and loaders
    train_dataset = PairDataset(N=params.num_colors)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=params.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, sampler=train_sampler,
                              num_workers=params.num_workers, persistent_workers=True)

    valid_dataset = PairDataset(N=params.num_colors)
    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=params.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, sampler=valid_sampler,
                              num_workers=params.num_workers, persistent_workers=True)

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))

    # Model definition
    mlp = MLPModel(
        n_steps=params.num_steps,
        n_steps_annealed=params.annealed_steps,
        n_layers=params.model.n_layers,
        x_dim=params.num_colors * 2,
        h_dim=params.model.h_dim,
        out_dim=1,
        emb_dim=params.model.emb_dim,
        widen=params.model.widen,
        annealed_landscape=params.annealed_landscape
    )
    ebm = EBMDiffModel(mlp)
    model = PortableDiffusionModel(
        ebm,
        diffusion,
        annealed_diffusion,
        annealed_landscape=params.annealed_landscape,
        contrastive=params.contrastive
    )
    #model.load_state_dict(torch.load(params.pk_path, map_location=device))
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=params.optimizer.lr)
    scheduler = get_scheduler(optimizer, params)

    loss_train, loss_val = [], []
    best_val_loss = math.inf

    # Output path
    name = f'{params.output_folder}/color'
    if params.diffusion: name += '_diffusion'
    if params.annealed_landscape: name += '_annealed'
    if params.contrastive: name += '_contrastive'
    output_pk = name + '.pk'

    # Load pretrained if available
    if hasattr(params, 'pretrained'):
        model.load_state_dict(torch.load(params.pretrained, map_location=device))

    num_colors = params.num_colors
    max_colors = params.max_colors

    loss_list = []

    def run_epoch(model, loader, optimizer, scheduler, diffusion, annealed_diffusion, device, params, epoch, mode="train"):
        is_train = mode == "train"
        model.train() if is_train else model.eval()
        loss_sum, loss_sum_mse, loss_sum_energy, count = 0.0, 0.0, 0.0, 0

        if not is_train:
            torch.set_grad_enabled(False)

        for target, neg in loader:
            target = target.to(device)
            neg = neg.to(device).reshape(-1, neg.shape[-1])

            timesteps = torch.randint(0, len(diffusion.betas) - 1, (target.size(0),), device=device)
            timesteps_annealed = torch.randint(0, len(annealed_diffusion.betas) - 1, (target.size(0),), device=device)

            loss_mse, loss_energy = model.loss(target.float(), timesteps, timesteps_annealed, x_initial=None, x_0_neg=neg)

            loss = 0.0
            if params.diffusion:
                loss += loss_mse
            if params.contrastive:
                loss += loss_energy * params.contrastive_loss_scale

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            loss_sum_mse += loss_mse.item()
            loss_sum_energy += loss_energy.item() * params.contrastive_loss_scale
            count += 1

            loss_list.append(loss.item())

        if is_train:
            scheduler.step(epoch)

        avg_loss = loss_sum / count
        avg_mse = loss_sum_mse / count
        avg_energy = loss_sum_energy / count

        print(f"{mode.capitalize()} epoch: {epoch}, Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, Energy: {avg_energy:.4f}")

        if is_train:
            return avg_loss
        else:
            return avg_loss, True  # Placeholder True to indicate saving (can be made conditional if needed)

    # === Main loop ===

    best_val_loss = float('inf')
    best_val_epoch = -1
    for epoch in range(params.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, diffusion, annealed_diffusion, device, params, epoch, mode="train")
        loss_train.append(train_loss)

        if epoch % 100 != 0 and epoch != 0:
            continue

        val_loss, _ = run_epoch(model, valid_loader, optimizer, scheduler, diffusion, annealed_diffusion, device, params, epoch, mode="eval")
        loss_val.append(val_loss)

        # Always save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            print(f"New best validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), output_pk)
            torch.save(optimizer.state_dict(), "optimizer.pt")
        print("Best validation loss so far:", best_val_loss, "at epoch", best_val_epoch)

        # plot loss matplotlib
        plt.figure(figsize=(10, 5))
        plt.plot(loss_train, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        # save plot
        plt.savefig(f"comp_reasoning/out/loss.png")

if __name__ == "__main__":
    experiment()