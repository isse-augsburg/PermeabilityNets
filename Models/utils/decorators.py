import functools
import torch
from random import random


def augumented_forward_ff_sequence(func):
    @functools.wraps(func)
    def flipped_in_out(self, x: torch.Tensor):
        if self.training:
            flip_x = random() > 0.5
            flip_y = random() > 0.5

            if flip_x:
                x = x.flip([2])
            if flip_y:
                x = x.flip([3])
            out = func(self, x)
            if flip_y:
                out = out.flip([2])
            if flip_x:
                out = out.flip([1])
            return out
        else:
            return func(self, x)
    return flipped_in_out
