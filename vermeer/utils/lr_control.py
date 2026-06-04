import math
from pprint import pformat
from typing import Tuple, List, Dict, Union
# from torch.optim.lr_scheduler import _LRScheduler
# import torch.optim as optim


# ## this function is adapted from the VAR repo: https://github.com/FoundationVision/VAR/blob/main/utils/lr_control.py ##### 
# def lr_wd_annealing(sche_type: str, optimizer, peak_lr, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001, wd=None, wd_end=None):
#     """
#     Decay the learning rate with specified schedule, e.g. half-cycle cosine after warmup
#     Decay weight decay from wd to wd_end with cosine annealing, independent of learning rate schedule
#     warmup-phase: linearly increase lr from wp0 to peak_lr while cur_it < wp_it
#     decay-phase: decay lr with specified schedule
#     decay schedules:
#         cos: cosine decay from peak_lr to wpe
#         lin: linear decay after holding at peak for 15% of remaining time 
#         lin0: linear decay after holding at peak for 5% of remaining time
#         lin00: immediate linear decay with no plateau
#         linX: configurable linear schedule
#         exp: exponential decay after 15% plateau
#     Args:
#         sche_type: type of learning rate schedule
#         optimizer: optimizer
#         peak_lr: peak learning rate
#         cur_it: current iteration
#         wp_it: warmup iteration
#         max_it: maximum iteration
#         wp0: warmup learning rate (fraction of peak_lr)
#         wpe: end learning rate (fraction of peak_lr)
#     """
#     wp_it = round(wp_it)
    
#     if cur_it < wp_it:
#         cur_lr = wp0 + (1-wp0) * cur_it / wp_it
#     else:
#         pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
#         rest = 1 - pasd     # [1, 0]
#         if sche_type == 'cos':
#             cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
#         elif sche_type == 'lin':
#             T = 0.15; max_rest = 1-T
#             if pasd < T: cur_lr = 1
#             else: cur_lr = wpe + (1-wpe) * rest / max_rest  # 1 to wpe
#         elif sche_type == 'lin0':
#             T = 0.05; max_rest = 1-T
#             if pasd < T: cur_lr = 1
#             else: cur_lr = wpe + (1-wpe) * rest / max_rest
#         elif sche_type == 'lin00':
#             cur_lr = wpe + (1-wpe) * rest
#         elif sche_type.startswith('lin'):
#             T = float(sche_type[3:]); max_rest = 1-T
#             wpe_mid = wpe + (1-wpe) * max_rest
#             wpe_mid = (1 + wpe_mid) / 2
#             if pasd < T: cur_lr = 1 + (wpe_mid-1) * pasd / T
#             else: cur_lr = wpe + (wpe_mid-wpe) * rest / max_rest
#         elif sche_type == 'exp':
#             T = 0.15; max_rest = 1-T
#             if pasd < T: cur_lr = 1
#             else:
#                 expo = (pasd-T) / max_rest * math.log(wpe)
#                 cur_lr = math.exp(expo)
#         else:
#             raise NotImplementedError(f'unknown sche_type {sche_type}')
    
#     cur_lr *= peak_lr
#     pasd = cur_it / (max_it-1)
    
#     inf = 1e6
#     min_lr, max_lr = inf, -1

#     # if we want different lr for different param_groups
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
#         max_lr = max(max_lr, param_group['lr'])
#         min_lr = min(min_lr, param_group['lr'])

#     # else same lr for all param_groups

#     if min_lr == inf: min_lr = -1
#     return min_lr, max_lr


## this function is adapted from the VAR repo: https://github.com/FoundationVision/VAR/blob/main/utils/lr_control.py ##### 
def lr_wd_annealing(sche_type: str, optimizer, peak_lr, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001, wd=None, wd_end=None):
    """
    Decay the learning rate with specified schedule, e.g. half-cycle cosine after warmup
    Decay weight decay from wd to wd_end with cosine annealing, independent of learning rate schedule
    warmup-phase: linearly increase lr from wp0 to peak_lr while cur_it < wp_it
    decay-phase: decay lr with specified schedule
    decay schedules:
        cos: cosine decay from peak_lr to wpe
        lin: linear decay after holding at peak for 15% of remaining time 
        lin0: linear decay after holding at peak for 5% of remaining time
        lin00: immediate linear decay with no plateau
        linX: configurable linear schedule
        exp: exponential decay after 15% plateau
    weight-decay schedule: cosine annealing from wd to wd_end if wd_end is provided, otherwise no weight decay
    """
    wp_it = round(wp_it)
    
    if cur_it < wp_it:
        cur_lr = wp0 + (1-wp0) * cur_it / wp_it
    else:
        pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
        rest = 1 - pasd     # [1, 0]
        if sche_type == 'cos':
            cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
        elif sche_type == 'lin':
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest  # 1 to wpe
        elif sche_type == 'lin0':
            T = 0.05; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest
        elif sche_type == 'lin00':
            cur_lr = wpe + (1-wpe) * rest
        elif sche_type.startswith('lin'):
            T = float(sche_type[3:]); max_rest = 1-T
            wpe_mid = wpe + (1-wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            if pasd < T: cur_lr = 1 + (wpe_mid-1) * pasd / T
            else: cur_lr = wpe + (wpe_mid-wpe) * rest / max_rest
        elif sche_type == 'exp':
            T = 0.15; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else:
                expo = (pasd-T) / max_rest * math.log(wpe)
                cur_lr = math.exp(expo)
        else:
            raise NotImplementedError(f'unknown sche_type {sche_type}')
            # todo: implement cosine annealing with warm restarts 
    
    cur_lr *= peak_lr
    pasd = cur_it / (max_it-1)
    if wd_end is not None:
        cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))
    else:
        cur_wd = wd
    
    inf = 1e6
    min_lr, max_lr = inf, -1
    min_wd, max_wd = inf, -1
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned as learning rate scaling factor (e.g. for LLRD)
        max_lr = max(max_lr, param_group['lr'])
        min_lr = min(min_lr, param_group['lr'])

        if wd_end is not None:
            param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
            max_wd = max(max_wd, param_group['weight_decay'])
            if param_group['weight_decay'] > 0:
                min_wd = min(min_wd, param_group['weight_decay'])

    if min_lr == inf: min_lr = -1
    if min_wd == inf: min_wd = -1
    if wd_end is not None:
        return min_lr, max_lr, min_wd, max_wd
    else:
        return min_lr, max_lr

#################################################################################
#                             Creating Learning Rate Scheduler                    #
################################################################################# 


# ## TODO: think about how this interacts with param_groups in the optimizer, e.g. if using LLRD 
# class CustomScheduler(_LRScheduler):
#     def __init__(self, optimizer, peak_lr, wp_ep, max_ep, sche_type:str, last_epoch=-1):
#         self.peak_lr = peak_lr
#         self.wp_ep = wp_ep
#         self.max_ep = max_ep
#         self.sche_type = sche_type
#         super(CustomScheduler, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.sche_type == 'cos':
#             pass
#         elif self.sche_type == 'lin':
#             T = 0.15; max_rest = 1-T


# def create_scheduler(optimizer, args):
#     if args.lr_schedule == 'cosine':
#         raise NotImplementedError("Cosine learning rate schedule is not implemented yet")
#     elif args.lr_schedule == 'linear':
#         # linear warmup + linear decay
#         scheduler1 = LambdaLR(optimizer, start_factor=args.lr_warmup_start, end_factor=args.lr, total_iters=args.warmup_epochs)
#         scheduler2 = LambdaLR(optimizer, start_factor=args.lr, end_factor=args.lr_min, total_iters=args.epochs - args.warmup_epochs)
#         scheduler = SequentialLR(
#             optimizer,
#             schedulers=[scheduler1, scheduler2],
#             milestones=[args.warmup_epochs],
#         )
#         return scheduler
#     else:
#         raise ValueError(f"Invalid learning rate schedule: {args.lr_schedule}")