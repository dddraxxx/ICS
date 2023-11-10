import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
import torch.nn.functional as F

from model.ICSA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset_ics import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         IMAGE_TOKEN_INDEX, IGNORE_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

import sys
import ipdb

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    tokenizer,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    log_frequency = 10

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            data_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            sam_input_name = ["images", "images_clip", "input_masks_list"]
            def to_type(x, dtype):
                if type(x) == list:
                    return [to_type(xx, dtype) for xx in x]
                if type(x) == torch.Tensor:
                    return x.to(dtype)
                raise
            for k in sam_input_name:
                if args.precision == "fp16":
                    input_dict[k] = to_type(input_dict[k], torch.half)
                elif args.precision == "bf16":
                    input_dict[k] = to_type(input_dict[k], torch.bfloat16)
                else:
                    input_dict[k] = to_type(input_dict[k], torch.float32)

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # log the first sample in every n steps
        if global_step % log_frequency == 0 or args.debug:
            log_sample_ids = input_dict["input_ids"][0].detach().cpu()
            img_idx = (log_sample_ids==IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            log_sample = (tokenizer.decode(log_sample_ids[:img_idx]) +
                          " [IMG] " +
                            tokenizer.decode(log_sample_ids[img_idx+1:]))
            log_generated_ids = output_dict["output_ids"][0].detach().cpu()
            # remove image token
            log_generated_ids = torch.cat([log_generated_ids[:img_idx],
                                           log_generated_ids[img_idx+255:]])
            # only get the output of the model (ignore instruction)
            labels = input_dict['labels'].cpu()
            log_generated_ids = log_generated_ids[labels[0]!=IGNORE_INDEX]
            log_generated = tokenizer.decode(log_generated_ids)

            log_sample_image = (input_dict["images_clip"][0].detach().cpu()).float() # 3, H, W
            # Normalize
            log_sample_image = (log_sample_image - log_sample_image.min())/(log_sample_image.max() - log_sample_image.min())*255
            log_sample_image = log_sample_image.byte()

            input_masks = input_dict['input_masks_list'][0].cpu()
            log_input_masks = input_masks[0].detach().cpu() # 1,H, W
            log_input_masks = F.interpolate(log_input_masks[None,None],
                                            size=log_sample_image.shape[-2:],
                                            mode='nearest')[0].bool()
            log_image = tv.utils.draw_segmentation_masks(log_sample_image, log_input_masks, alpha=0.8, colors='red')
            log_image = tv.utils.make_grid([log_sample_image, log_image], nrow=2, padding=5, pad_value=255)

            log_text = 'GT\n' + log_sample + '\n\nGenerated\n' + log_generated

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

                # add the log things
                writer.add_text("train/text", log_text, global_step)
                writer.add_image(
                    "train/sample_image", log_image, global_step, dataformats="CHW"
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter