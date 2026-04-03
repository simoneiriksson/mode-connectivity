import torch

def make_diy_scheduler(optimizer, train_num_steps, lr_start_warmup, lr, lr_warmup_steps, lr_finetune_halftime, lr_finetune_steps):
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

def build_scheduler(optimizer, total_iter, batches_per_epoch, lr_start, lr_end, scheduler_type):
    if scheduler_type == "exponential":
        gamma = ((lr_end.log()-lr_start.log())/total_iter).exp()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lr_end/lr_start, total_iters=total_iter)
    elif scheduler_type == "diy":
        scheduler = make_diy_scheduler(optimizer, 
                                            train_num_steps=total_iter, 
                                            lr_start_warmup=lr_start.clone(), 
                                            lr=lr_end.clone(), 
                                            lr_warmup_steps=5*batches_per_epoch, 
                                            lr_finetune_halftime=total_iter // (5*3), 
                                            lr_finetune_steps=total_iter // 3
        )
    elif scheduler_type == "none":
        scheduler = None
    else: 
        raise ValueError(f"Wrong model_scheduler: {scheduler_type}")
    return scheduler

def build_optimizer(model, lr, optimizer_type):
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    else: 
        raise ValueError(f"Wrong model_optimizer: {optimizer_type}")
    return optimizer