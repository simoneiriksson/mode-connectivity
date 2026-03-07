import torch

def make_scheduler(optimizer, train_num_steps, lr_start_warmup, lr, lr_warmup_steps, lr_finetune_halftime, lr_finetune_steps):
    """Make a learning rate scheduler with linear warmup, then constant, then exponential decay.
    Args:
        optimizer: the optimizer for which to schedule the learning rate.
        train_num_steps: the total number of training steps (epochs * steps per epoch).
        lr_start_warmup: the initial learning rate at the start of warmup.
        lr: the base learning rate to use after warmup (and at the start of finetuning).
        lr_warmup_steps: the number of steps to linearly warm up the learning rate.
        lr_finetune_halftime: the number of steps over which to halve the learning rate during finetuning.
        lr_finetune_steps: the total number of steps to use for finetuning (after warmup and constant phases).
    Returns:
        A PyTorch learning rate scheduler that implements the specified warmup and finetuning schedule.
    """
    lr_total_steps = train_num_steps
    lr_start_warmup = lr_start_warmup  
    lr_const = lr 

    lr_warmup_steps = lr_warmup_steps
    lr_finetune_halftime = lr_finetune_halftime
    lr_finetune_steps = min(lr_total_steps, lr_finetune_steps)
    lr_const_steps = lr_total_steps - lr_warmup_steps - lr_finetune_steps

    # Always define "base" as lr_const so finetune starts from lr_const.
    for group in optimizer.param_groups:
        group["lr"] = lr_const

    start_ratio = lr_start_warmup / lr_const  # can be > 1.0, LambdaLR allows that

    pre_steps = max(0, lr_warmup_steps + lr_const_steps)

    def pre_lambda(step: int):
        # step is 0-based in LambdaLR
        if lr_warmup_steps <= 0:
            return 1.0  # no warmup, stay at lr_const
        if step >= lr_warmup_steps:
            return 1.0  # after warmup, hold lr_const through the const phase
        t = step / float(lr_warmup_steps)  # 0 -> 1 over warmup
        return start_ratio + (1.0 - start_ratio) * t

    sched_pre = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=pre_lambda)

    gamma = 0.5 ** (1.0 / lr_finetune_halftime)
    sched_finetune = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Switch into finetune after warmup + const.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[sched_pre, sched_finetune],
        milestones=[pre_steps],
    )
    return scheduler


import torch

def make_scheduler_epochs(optimizer, train_num_epochs, lr_start_warmup, lr, lr_warmup_epochs, lr_finetune_halftime, lr_finetune_epochs):
    """Make a learning rate scheduler with linear warmup, then constant, then exponential decay.
    Args:
        optimizer: the optimizer for which to schedule the learning rate.
        train_num_epochs: the total number of training epochs.
        lr_start_warmup: the initial learning rate at the start of warmup.
        lr: the base learning rate to use after warmup (and at the start of finetuning).
        lr_warmup_epochs: the number of epochs to linearly warm up the learning rate.
        lr_finetune_halftime: the number of epochs over which to halve the learning rate during finetuning.
        lr_finetune_epochs: the total number of epochs to use for finetuning (after warmup and constant phases).
    Returns:
        A PyTorch learning rate scheduler that implements the specified warmup and
        finetuning schedule. Call scheduler.step() once per epoch.
    """
    lr_total_epochs = train_num_epochs
    lr_start_warmup = lr_start_warmup
    lr_const = lr

    lr_warmup_epochs = lr_warmup_epochs
    lr_finetune_halftime = lr_finetune_halftime
    lr_finetune_epochs = min(lr_total_epochs, lr_finetune_epochs)
    lr_const_epochs = lr_total_epochs - lr_warmup_epochs - lr_finetune_epochs

    # Always define "base" as lr_const so finetune starts from lr_const.
    for group in optimizer.param_groups:
        group["lr"] = lr_const

    start_ratio = lr_start_warmup / lr_const  # can be > 1.0, LambdaLR allows that

    pre_epochs = max(0, lr_warmup_epochs + lr_const_epochs)

    def pre_lambda(epoch: int):
        # epoch is 0-based in LambdaLR
        if lr_warmup_epochs <= 0:
            return 1.0  # no warmup, stay at lr_const
        if epoch >= lr_warmup_epochs:
            return 1.0  # after warmup, hold lr_const through the const phase
        t = epoch / float(lr_warmup_epochs)  # 0 -> 1 over warmup
        return start_ratio + (1.0 - start_ratio) * t

    sched_pre = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=pre_lambda)

    gamma = 0.5 ** (1.0 / max(1, lr_finetune_halftime))
    sched_finetune = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Switch into finetune after warmup + const.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[sched_pre, sched_finetune],
        milestones=[pre_epochs],
    )
    return scheduler