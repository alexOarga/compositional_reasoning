import math
import warnings
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_scatter import scatter_mean


class PortableDiffusionModel(nn.Module):
    def __init__(
            self,
            net,
            diffusion, 
            annealed_diffusion, 
            annealed_landscape=False,
            contrastive=False,
        ):
        super(PortableDiffusionModel, self).__init__()

        self.net = net
        self.loss_fn = nn.MSELoss()
        self.contrastive_loss_fn = nn.CrossEntropyLoss()
        self.diffusion = diffusion
        self.annealed_diffusion = annealed_diffusion
        self.annealed_landscape = annealed_landscape
        self.contrastive = contrastive

    def forward(self, x, t, x_initial=None, **kwargs):
        out = self.net(x, t, x_initial=x_initial, **kwargs)
        return out

    def loss(self, x_0, t, t_annealed, x_initial, x_0_neg=None, mask=None, negative_mask=None, **kwargs):

        if self.annealed_landscape:
            noise_annealed = torch.randn_like(x_0)
            x_0_annealed = self.annealed_diffusion.q_sample(x_0, t_annealed, noise=noise_annealed)
        else:
            x_0_annealed = x_0

        noise = torch.randn_like(x_0_annealed)
        x_t = self.diffusion.q_sample(x_0_annealed, t, noise=noise)
        x_t.requires_grad_(requires_grad=True)

        epsilon = self.net(
            x_t,
            t,
            x_initial=x_initial,
            **kwargs
        )
        
        assert epsilon.shape == noise.shape
        if mask is None:
            im_loss = self.loss_fn(epsilon, noise)
        else:
            eee = epsilon[mask]
            nnn = noise[mask]
            im_loss = self.loss_fn(eee, nnn)

        if not self.contrastive:
            loss_energy = torch.tensor([0.0])
        else:
            t_inter = t.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            t_annealed_inter = t_annealed.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            noise_inter = noise.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            x_t_neg = self.diffusion.q_sample(x_0_neg, t_inter, noise=noise_inter)

            x_t_concat = torch.cat([x_t, x_t_neg], dim=0)
            t_concat = torch.cat([t, t_inter], dim=0)
            t_annealed_concat = torch.cat([t_annealed, t_annealed_inter], dim=0)

            if type(x_initial) == torch.Tensor:
                x_initial_concat = [x_initial]
                tmp = x_initial.repeat_interleave(x_t_neg.size(0) // x_t.size(0), dim=0)
                x_initial_concat.append(tmp)
                x_initial_concat = torch.cat(x_initial_concat, dim=0)
            elif type(x_initial) == list:
                x_initial_concat = [None] * len(x_initial)
                for i in range(len(x_initial)):
                    tmp = x_initial[i].repeat_interleave(x_t_neg.size(0) // x_t.size(0), dim=0)
                    x_initial_concat[i] = torch.cat([x_initial[i], tmp], dim=0)
            else:
                warnings.warn("x_initial is not a list or a tensor")
                x_initial_concat = x_initial

            # repeat interleave also kwargs
            for k, v in kwargs.items():
                if type(v) == torch.Tensor:
                    tmp = v.repeat_interleave(x_t_neg.size(0) // x_t.size(0), dim=0)
                    kwargs[k] = torch.cat([v, tmp], dim=0)
                elif type(v) == list:
                    # WARNING: this has not been tested
                    for i in range(len(v)):
                        tmp = v[i].repeat_interleave(x_t_neg.size(0) // x_t.size(0), dim=0)
                        v[i] = torch.cat([v[i], tmp], dim=0)
                    kwargs[k] = v
                else:
                    warnings.warn(f"Value of {k} in kwargs is not a tensor or a list")
                
            energy = self.net.neg_logp_unnorm(
                x_t_concat,
                t_concat,
                x_initial=x_initial_concat,
                **kwargs
            )

            energy_pos = energy[:x_t.size(0)]
            energy_neg = energy[x_t.size(0):]
  
            def _masked_xentropy(logits, target, mask):
                logits[~mask] = float('-inf')
                loss_fn = nn.CrossEntropyLoss()
                return loss_fn(logits, target)
                
            if negative_mask is None:
                mask = torch.ones((x_t.size(0), ), dtype=torch.bool, device=x_t.device)
            else:
                # append a column of true at beggining of mask
                mask = torch.cat([
                    torch.ones((negative_mask.size(0), 1), dtype=torch.bool, device=negative_mask.device), 
                    negative_mask
                ], dim=1)
            logits = torch.cat([-energy_pos.unsqueeze(1), -energy_neg.view(x_t.size(0), -1)], dim=1)
            temperature = 1.0
            logits = logits / temperature
            loss_energy = _masked_xentropy(logits, torch.zeros(x_t.size(0), dtype=torch.long, device=logits.device), mask)

        return im_loss.mean(), loss_energy.mean()


    def graph_loss(self, x_0, t, t_annealed, x_initial, x_0_neg=None, mask=None, negative_mask=None, **kwargs):

        if self.annealed_landscape:
            noise_annealed = torch.randn_like(x_0)
            x_0_annealed = self.annealed_diffusion.q_sample(x_0, t_annealed, noise=noise_annealed)
        else:
            x_0_annealed = x_0

        noise = torch.randn_like(x_0_annealed)
        x_t = self.diffusion.q_sample(x_0_annealed, t, noise=noise)
        x_t.requires_grad_(requires_grad=True)

        epsilon = self.net(
            x_t,
            t,
            x_initial=x_initial,
            **kwargs
        )
        
        assert epsilon.shape == noise.shape
        if mask is None:
            im_loss = self.loss_fn(epsilon, noise)
        else:
            eee = epsilon[mask]
            nnn = noise[mask]
            im_loss = self.loss_fn(eee, nnn)

        if not self.contrastive:
            loss_energy = torch.tensor([0.0])
        else:
            t_inter = t.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            t_annealed_inter = t_annealed.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            noise_inter = noise.repeat_interleave(x_0_neg.size(0) // x_t.size(0), dim=0)
            x_t_neg = self.diffusion.q_sample(x_0_neg, t_inter, noise=noise_inter)

            x_t_concat = torch.cat([x_t, x_t_neg], dim=0)
            t_concat = torch.cat([t, t_inter], dim=0)
            t_annealed_concat = torch.cat([t_annealed, t_annealed_inter], dim=0)

            batch = x_initial
            graphs_list = [batch, batch]
            from torch_geometric.data import Batch
            batch_concat = Batch.from_data_list(graphs_list)
            x_initial_concat = batch_concat
                
            energy = self.net.neg_logp_unnorm(
                x_t_concat,
                t_concat,
                x_initial=x_initial_concat,
                **kwargs
            )

            energy_pos = energy[:x_t.size(0)]
            energy_neg = energy[x_t.size(0):]
  
            def _masked_xentropy(logits, target, mask):
                logits[~mask] = float('-inf')
                loss_fn = nn.CrossEntropyLoss()
                return loss_fn(logits, target)
                
            if negative_mask is None:
                mask = torch.ones((x_t.size(0), ), dtype=torch.bool, device=x_t.device)
            else:
                # append a column of true at beggining of mask
                mask = torch.cat([
                    torch.ones((negative_mask.size(0), 1), dtype=torch.bool, device=negative_mask.device), 
                    negative_mask
                ], dim=1)
            logits = torch.cat([-energy_pos.unsqueeze(1), -energy_neg.view(x_t.size(0), -1)], dim=1)
            temperature = 1.0
            logits = logits / temperature
            loss_energy = _masked_xentropy(logits, torch.zeros(x_t.size(0), dtype=torch.long, device=logits.device), mask)

        return im_loss.mean(), loss_energy.mean()


    def p_sample(self, shape, x_initial):
        x_sampled = self.diffusion.p_sample_loop(
                None, # sampler
                model=self.net,
                shape=shape,
                model_kwargs={
                    'x_initial': x_initial,
                }
            )
        return x_sampled

    def p_gradient(self, x, t, **model_args):
        gradient = self.net(x, t, **model_args)
        custom_alphas = torch.sqrt(1.0 / torch.tensor((1 - self.diffusion.alphas_cumprod), dtype=torch.float, device=x.device))
        custom_alphas_t = custom_alphas[t]
        return custom_alphas_t[:, None] * gradient

    def p_energy(self, x, t, **model_args):
        energy = self.net.neg_logp_unnorm(x, t, **model_args)
        custom_alphas = torch.sqrt(1.0 / torch.tensor((1 - self.diffusion.alphas_cumprod), dtype=torch.float, device=x.device))
        custom_alphas_t = custom_alphas[t]
        res = custom_alphas_t * energy
        return res