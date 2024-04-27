import torch
import numpy as np
from tinygrad import Tensor

def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)

def topk(input_, k, dim=-1, largest=True, sorted=False):
  k = min(k, input_.shape[dim]-1)
  input_ = input_.numpy()
  if largest: input_ *= -1
  ind = np.argpartition(input_, k, axis=dim)
  if largest: input_ *= -1
  ind = np.take(ind, np.arange(k), axis=dim) # k non-sorted indices
  input_ = np.take_along_axis(input_, ind, axis=dim) # k non-sorted values
  if not sorted: return Tensor(input_), ind
  if largest: input_ *= -1
  ind_part = np.argsort(input_, axis=dim)
  ind = np.take_along_axis(ind, ind_part, axis=dim)
  if largest: input_ *= -1
  val = np.take_along_axis(input_, ind_part, axis=dim)
  return Tensor(val), ind