from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging

_C = CN()
_C.name = 'vpt_deep'

# ----- DATASET BUILDER -----
_C.dataset = 'cifar100'
_C.data_path = './data/' 
_C.type = '1k'

## Long tailed settings
_C.LT = CN()
_C.LT.enable = False
_C.LT.imb_type = 'exp'
_C.LT.imb_factor = 0.01

_C.sampler = CN()
_C.sampler.type = 'default'
_C.sampler.weighted = CN()
_C.sampler.weighted.type = 'balance'

_C.sampler.dual_sample = CN()
_C.sampler.dual_sample.enable = False
_C.sampler.dual_sample.type = 'long-tailed'

_C.head_class_idx = [0, 1]
_C.med_class_idx = [0, 1]
_C.tail_class_idx = [0, 1]

# ----- BACKBONE BUILDER -----
# to do
_C.resume = ''

# ----- LOSS BUILDER -----
_C.loss = CN()
_C.loss.type = "CrossEntropyLoss"
_C.loss.add_extra_info = False
_C.loss.CE = CN()
_C.loss.CE.reweight_CE = False


_C.loss.LDAM = CN()
_C.loss.LDAM.max_m = 0.5
_C.loss.LDAM.s = 30
_C.loss.LDAM.reweight_epoch=-1

_C.loss.GCL = CN()
_C.loss.GCL.s = 30.
_C.loss.GCL.m = 0.
_C.loss.GCL.reweight_epoch=-1
_C.loss.GCL.focal=False
_C.loss.GCL.noise_mul=1.
_C.loss.GCL.gamma=0.

# -----DISTRIBUTED TRAINING -----
# to do
_C.distributed = False


# ----- Train -----
_C.epochs = 100
_C.warmup_epochs = 10
_C.batch_size = 128
_C.optimizer = 'sgd'
_C.base_lr = 0.01
_C.momentum = 0.9
_C.weight_decay = 0.01
_C.decay_rate = 0.2
_C.image_size = 224
_C.scheduler = 'cosine'

# ----- Prompt setting -----
_C.prompt_length = 10
_C.base_model = 'vit_base_patch16_224_in21k'
_C.VPT_type = 'Deep'
_C.patch_size = 16

# ----- Others -----
_C.print_freq = 10
_C.workers = 4
_C.resume = ''
_C.merge_prompt = False
_C.temperature = False
_C.normed = False
_C.dis_mul = 0.1

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # cfg.freeze()
