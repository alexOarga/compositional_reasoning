import torch
import torch_scatter
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def p_sample_loop(
    diffusion,
    sampler,
    model,
    shape,
    annealed_timestep,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    return_samples=False,
):
    """
    Generate samples from the model and yield intermediate samples from
    each timestep of diffusion.
    Arguments are the same as p_sample_loop().
    Returns a generator over dicts, where each dict is the return value of
    p_sample().
    """
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        img = noise
    else:
        img = torch.randn(*shape, device=device)
    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    samples = [img.clone().detach().cpu()]

    for i in indices:
        #print(i)
        t = torch.tensor([i] * shape[0], device=device)
        out = diffusion.p_sample(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
        )
        out = out["sample"].detach()

        if sampler is not None:
            if i > 50 :
                print(i)
                out=sampler.sample_step(out, i, t, model_kwargs).detach()

        img = out
        samples.append(img.clone().detach().cpu())

    if not return_samples:
        return img
    else:
        return img, samples


def pem_sampling(
    diffusion,
    sampler,
    model,
    shape,
    annealed_timestep,
    pem_num_particles,
    noise=None,
    clip_denoised=True,
    clip_range=None,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    interleave_repeat=False,
    return_samples=False,
):
    
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    def repeat_fn(t, num):
        if interleave_repeat == True:
            return t.repeat_interleave(num, 0)
        else:
            return t.repeat(num, *[1] * (t.ndim - 1))

    P = pem_num_particles

    if noise is not None:
        img = noise
        img = repeat_fn(img, P)
    else:
        shape_copy = list(shape)
        shape_copy[0] = shape_copy[0] * P
        img = torch.randn(*shape_copy, device=device)

    annealed_timestep = repeat_fn(annealed_timestep, P)

    for k in model_kwargs:
        if model_kwargs[k] is None:
            pass
        elif isinstance(model_kwargs[k], torch.Tensor):
            model_kwargs[k] = repeat_fn(model_kwargs[k], P)
        elif isinstance(model_kwargs[k], list):
            for j in range(len(model_kwargs[k])):
                model_kwargs[k][j] = repeat_fn(model_kwargs[k][j], P)
        else:
            import warnings
            warnings.warn("model_kwargs must be a tensor or a list of tensors or None")

    noise = torch.randn_like(img)
    log_samples = [noise.clone()]
    initial_step_samples = []
    denoised_samples = []
    resampled_samples = []
    

    for i in indices:
        print(i)
        t = torch.tensor([i] * img.shape[0], device=device)
        
        # 1. IMPORTANCE EVALUATION AND RESAMPLING
        weights_all = model.net.neg_logp_unnorm(
            img,
            t,
            **model_kwargs
        )#.mean(dim=-1)

        # TODO: implement this with an extra dimension instead of a for loop
        for j in range(0, img.shape[0], P):

            # Assuming weights_all, img, P, j, and device are defined
            weights = weights_all[j:j+P]
            weights = F.softmax(-weights, dim=-1)  # Ensure softmax operates along the correct dimension
            weights = weights.squeeze()

            samples = img[j:j+P].squeeze()
            resample = torch.multinomial(weights, num_samples=len(weights), replacement=True)
            img[j:j+P] = samples[resample].to(device).float().reshape(-1, samples.shape[1])
        initial_step_samples.append(img.clone().detach().cpu())

        # 2. SAMPLING: FORWARD AND BACKWARD
        x_estimate = img.clone()
        for j in range(i, -1, -1):
            t_estimate = torch.tensor([j] * x_estimate.shape[0], device=device)
            out = diffusion.p_sample(
                model,
                x_estimate,
                t_estimate,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            out = out["sample"].detach()
            x_estimate = out
        
        if clip_range is not None:
            min_c, max_c = clip_range
            x_estimate = torch.clamp(x_estimate, min_c, max_c)
        
        denoised_samples.append(x_estimate.clone().detach().cpu())
        
        x_0_estimate = x_estimate
        x_t_estimate = diffusion.q_sample(x_0_estimate, t, noise=noise)
        img = x_t_estimate
        resampled_samples.append(img.clone().detach().cpu())

        # 3. DENOISE NEXT STEP
        out = diffusion.p_sample(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
        )
        out = out["sample"].detach()
        img = out

        if sampler is not None:
            if i > 50 :
                print(i)
                out=sampler.sample_step(out, i,t, model_kwargs).detach()

        img = out
        log_samples.append(img.clone().detach())

    filtered_img = []
    filtered_idx = []
    e = model.net.neg_logp_unnorm(
        img,
        torch.tensor([0] * img.shape[0], device=device),
        **model_kwargs
    )
    for j in range(0, img.shape[0], P):
        e_j = e[j:j+P]
        idx = e_j.argmin()
        filtered_idx.append(idx.item())
        filtered_img.append(img[j:j+P][idx])

    img = torch.stack(filtered_img)
    log_samples.append(img.clone().detach())

    if not return_samples:
        return img
    else:
        return img, log_samples, initial_step_samples, denoised_samples, resampled_samples, filtered_idx


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def q_sample_step(diffusion, x_prev, t, noise=None):
    """
    Sample from q(x_t | x_{t-1}).
    :param x_prev: the data batch at step t-1 (x_{t-1}).
    :param t: the current diffusion step (1-based index).
    :param noise: optional Gaussian noise to add.
    :return: A noisy version of x_prev.
    """
    if noise is None:
        noise = torch.randn_like(x_prev)
    assert noise.shape == x_prev.shape
    return (
        _extract_into_tensor(np.sqrt(1.0 - diffusion.betas), t, x_prev.shape) * x_prev
        + _extract_into_tensor(diffusion.betas, t, x_prev.shape) * noise
    )
