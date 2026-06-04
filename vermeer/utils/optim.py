import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import inspect

import collections
import numpy as np


#################################################################################
#                             Creating Optimizer                                #
#################################################################################

def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

# based on the Electra implementation of LLRD: 
# https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/optimization.py#L188-L193 
def _get_layer_lrs(learning_rate, layer_decay, n_layers):
  """Have lower learning rates for layers closer to the input.
    Assumes output layer is the last layer,
    then n_layers is the number of transformer layers,
    then input layers are cls_embedding and tok_embedding layers.
  """
  output_lr = learning_rate
  key_to_depths = collections.OrderedDict({
      "cls_embedding": 0,
      "tok_embeddings": 0,
      "sos_embed":0,
      "output": n_layers + 2,
      "norm": n_layers + 2
  })
  for layer in range(n_layers):
    key_to_depths["layers." + str(layer) + "."] = layer + 1
  return {
      key: learning_rate * (layer_decay ** (n_layers + 2 - depth))
      for key, depth in key_to_depths.items()
  }

def _get_nodecay_decay_params(model):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [n for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [n for n, p in param_dict.items() if p.dim() < 2]

    return decay_params, nodecay_params

def _do_use_weight_decay(param_name, decay_params, nodecay_params):
    if param_name in decay_params:
        return True
    elif param_name in nodecay_params:
        return False
    else:
        raise ValueError(f"Parameter {param_name} not found in decay params or nodecay params")

def creat_llrd_optimizer(model, weight_decay, learning_rate, betas, logger, layer_decay):
  """ creates optimizer with layer-wise learning rate decay """
  n_layers = model.n_layer
  print(n_layers)
  layer_lrs = _get_layer_lrs(learning_rate, layer_decay, n_layers)
  
  # create separate optim groups for each layer lr parameter that requires grad and has weight decay vs not
  optim_groups = []
  decay_params, nodecay_params = _get_nodecay_decay_params(model) # only returns parameters that require grad

  # Track which params we have added to avoid duplicates or missing ones
  params_added = set()

  for lr_key, lr_value in layer_lrs.items():
    # 1. Group for weights (Decay + Layer LR)
    weights = [p for n, p in model.named_parameters() if n.startswith(lr_key) and n in decay_params]
    if weights:
        optim_groups.append({
            'params': weights,
            'lr': lr_value,
            'weight_decay': weight_decay,
            'name': f"{lr_key}_weight"
        })
        params_added.update([n for n, p in model.named_parameters() if n.startswith(lr_key) and n in decay_params])

    # 2. Group for biases/norms (No Decay + Layer LR)
    norms_biases = [p for n, p in model.named_parameters() if n.startswith(lr_key) and n in nodecay_params]
    if norms_biases:
        optim_groups.append({
            'params': norms_biases,
            'lr': lr_value,
            'weight_decay': 0.0,
            'name': f"{lr_key}_norm_bias"
        })
        params_added.update([n for n, p in model.named_parameters() if lr_key in n and n in nodecay_params])

  # # 3. Catch-all for any parameters missed by layer keys
  remaining_decay = [n for n, p in model.named_parameters() if n in decay_params and n not in params_added]
  remaining_nodecay = [n for n, p in model.named_parameters() if n in nodecay_params and n not in params_added]
  if len(remaining_decay) > 0 or len(remaining_nodecay) > 0:
    print(f"WARNING: Some parameters were not added to any group: {len(remaining_decay)} decay params and {len(remaining_nodecay)} nodecay params")
    print("remaining_decay:", remaining_decay)
    print("remaining_nodecay:", remaining_nodecay)
    print("exiting now")
    exit()

  # count number of parameters in each group
  for i, group in enumerate(optim_groups):
    num_params = sum(p.numel() for p in group['params'])
    logger.info(f"Optim group {i} has {num_params:,} parameters with lr {group['lr']:.6e} and weight decay {group['weight_decay']:.6e}")
    print(f"Optim group {i} has {num_params:,} parameters with lr {group['lr']:.6e} and weight decay {group['weight_decay']:.6e}")

  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  extra_args = dict(fused=True) if fused_available else dict()
  optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
  logger.info(f"using fused AdamW: {fused_available}")

  return optimizer