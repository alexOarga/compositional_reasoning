import torch

def get_scheduler(optimizer, params):
    if params.optimizer.scheduler == 'CosineLRScheduler':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=params.epochs,
            lr_min=params.optimizer.lr_min,
            warmup_t=params.optimizer.warmup_t,
            warmup_lr_init=params.optimizer.warmup_lr_init,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0,
        )
    return scheduler

def get_lr(scheduler, t):
    # we dont want to depend on timm
    from timm.scheduler import CosineLRScheduler

    if type(scheduler) == CosineLRScheduler:
        return scheduler._get_lr(t)[0]
    elif type(scheduler) == torch.optim.lr_scheduler.ConstantLR:
        return scheduler.get_last_lr()[0]
    else:
        raise ValueError("Invalid scheduler type.")
