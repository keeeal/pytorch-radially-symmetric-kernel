
from functools import partial
from random import random
from typing import Optional, Sequence

import torch
from torch import device, dtype, nn, tensor
from torch.nn.functional import conv2d


def radial_kernel_1d(
    kernel_size: int,
    device: device = None,
    dtype: dtype = None,
) -> torch.Tensor:
    return torch.linspace(
        start=-(kernel_size-1)/2,
        end=(kernel_size-1)/2,
        steps=kernel_size,
        device=device,
        dtype=dtype,
    )


def radial_kernel(
    kernel_size: Sequence[int],
    device: device = None,
    dtype: dtype = None,
) -> torch.Tensor:
    kernel = partial(radial_kernel_1d, device=device, dtype=dtype)
    kernel = map(kernel, kernel_size)
    kernel = torch.stack(torch.meshgrid(*kernel, indexing="xy"))
    kernel = kernel.square().sum(dim=0).sqrt()
    return kernel


class Radial2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int | Sequence[int],
    ):
        super().__init__()
        self.channels = channels
        self.register_buffer("r",
            torch.stack(channels * [radial_kernel(kernel_size).unsqueeze(0)])
        )


class Gauss2d(Radial2d):
    def __init__(
        self,
        channels: int,
        kernel_size: int | Sequence[int],
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        super().__init__(channels=channels, kernel_size=kernel_size)
        alpha = random() if alpha is None else alpha
        sigma = random() if sigma is None else sigma
        self.alpha = nn.Parameter(tensor(alpha, dtype=torch.float32))
        self.sigma = nn.Parameter(tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        weight = torch.exp(-self.r.square() / (2 * self.sigma.square()))
        weight = self.alpha * weight / weight.sum()

        return conv2d(x, weight, groups=self.channels, padding="same")


class Sum(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return torch.stack([layer(x) for layer in self.layers]).sum(dim=0)
