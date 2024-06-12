# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Collections of utilities related to optimization."""
from bisect import bisect_right
import os

import torch


def update_ema(model, model_ema, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        if hasattr(model, "module"):
            # unwrapping DDP
            model = model.module
        msd = model.state_dict()
        for k, ema_v in model_ema.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    args,
):
    """Adjust the lr according to the schedule.

    Args:
        Optimizer: torch optimizer to update.
        epoch(int): number of the current epoch.
        curr_step(int): number of optimization step taken so far.
        num_training_step(int): total number of optimization steps.
        args: additional training dependent args:
              - lr_drop(int): number of epochs before dropping the learning rate.
              - fraction_warmup_steps(float) fraction of steps over which the lr will be increased to its peak.
              - lr(float): base learning rate
              - lr_backbone(float): learning rate of the backbone
              - text_encoder_backbone(float): learning rate of the text encoder
              - schedule(str): the requested learning rate schedule:
                   "step": all lrs divided by 10 after lr_drop epochs
                   "multistep": divided by 2 after lr_drop epochs, then by 2 after every 50 epochs
                   "linear_with_warmup": same as "step" for backbone + transformer, but for the text encoder, linearly
                                         increase for a fraction of the training, then linearly decrease back to 0.
                   "all_linear_with_warmup": same as "linear_with_warmup" for all learning rates involved.

    """
    try:
        num_warmup_steps = args.num_warmup_steps
    except:
        return

    if epoch > 0:
        return
    
    if curr_step > num_warmup_steps:
        return

    text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
    optimizer.param_groups[-1]["lr"] = args.lr_backbone * text_encoder_gamma


