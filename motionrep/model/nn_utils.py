"""
Various utilities for neural networks.
"""

import math
from abc import abstractmethod
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class TriplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = normalization(channels)
        self.norm_xz = normalization(channels)
        self.norm_yz = normalization(channels)

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy)  # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz)  # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz)  # [B, C, W, D]

        # assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        # assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        # assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return (self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))


class TriplaneUpsample2x(nn.Module):
    def __init__(self, only_last_chn=True) -> None:
        super().__init__()

        self.only_last_chn = only_last_chn

        if self.only_last_chn:
            self.scale_factor = (1, 2)  # 1 for temporal dimension
        else:
            self.scale_factor = 2

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        # or tx, ty, tz

        tpl_xy = F.interpolate(
            tpl_xy, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        tpl_xz = F.interpolate(
            tpl_xz, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        tpl_yz = F.interpolate(
            tpl_yz, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        return (tpl_xy, tpl_xz, tpl_yz)


class TriplaneDownsample2x(nn.Module):
    def __init__(self, only_last_chn: bool = True) -> None:
        super().__init__()
        self.only_last_chn = only_last_chn

        if self.only_last_chn:
            self.kernel_size = (1, 2)  # 1 for temporal dimension
        else:
            self.kernel_size = 2

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        # or tx, ty, tz

        tpl_xy = F.avg_pool2d(
            tpl_xy, kernel_size=self.kernel_size, stride=self.kernel_size
        )
        tpl_xz = F.avg_pool2d(
            tpl_xz, kernel_size=self.kernel_size, stride=self.kernel_size
        )
        tpl_yz = F.avg_pool2d(
            tpl_yz, kernel_size=self.kernel_size, stride=self.kernel_size
        )

        return (tpl_xy, tpl_xz, tpl_yz)


class TriplaneConv(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        kernel_size,
        padding,
        is_rollout=False,
        is_spatial=False,
    ) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout
        self.is_spatial = is_spatial

        self.conv_xy = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.conv_xz = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )
        self.conv_yz = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )

    def forward(self, featmaps):
        """
        Args:
            featmaps: (h_xy, h_xz, h_yz) or [h_tx, h_ty, h_tz].
                in shape: [B, C, H, W]
        """
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            if self.is_spatial and 0:
                tpl_xy_h = th.cat(
                    [
                        tpl_xy,
                        th.mean(tpl_yz, dim=-1, keepdim=True)
                        .transpose(-1, -2)
                        .expand_as(tpl_xy),
                        th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy),
                    ],
                    dim=1,
                )  # [B, C * 3, H, W]
                tpl_xz_h = th.cat(
                    [
                        tpl_xz,
                        th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                        th.mean(tpl_yz, dim=-2, keepdim=True).expand_as(tpl_xz),
                    ],
                    dim=1,
                )  # [B, C * 3, H, D]
                tpl_yz_h = th.cat(
                    [
                        tpl_yz,
                        th.mean(tpl_xy, dim=-2, keepdim=True)
                        .transpose(-1, -2)
                        .expand_as(tpl_yz),
                        th.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_yz),
                    ],
                    dim=1,
                )  # [B, C * 3, W, D]
            else:
                # tpl_xy, tpl_xz, tpl_yz = featmaps
                # TODO

                tpl_xy_h = th.cat(
                    [
                        tpl_xy,
                        th.mean(tpl_yz, dim=-1, keepdim=True).expand_as(tpl_xy),
                        th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_xy),
                    ],
                    dim=1,
                )
                tpl_xz_h = th.cat(
                    [
                        tpl_xz,
                        th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_xz),
                        th.mean(tpl_yz, dim=-1, keepdim=True).expand_as(tpl_xz),
                    ],
                    dim=1,
                )  # [B, C * 3, H, D]
                tpl_yz_h = th.cat(
                    [
                        tpl_yz,
                        th.mean(tpl_xy, dim=-1, keepdim=True).expand_as(tpl_yz),
                        th.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_yz),
                    ],
                    dim=1,
                )  # [B, C * 3, W, D]

        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz

        tpl_xy_h = self.conv_xy(tpl_xy_h)
        tpl_xz_h = self.conv_xz(tpl_xz_h)
        tpl_yz_h = self.conv_yz(tpl_yz_h)

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        is_rollout=False,
        is_spatial=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # kernel_size_list = [(3, 3)] * 3 if is_spatial else [(3, 3), (1, 3), (1, 3)]
        # padding_list = [(1, 1)] * 3 if is_spatial else [(1, 1), (0, 1), (0, 1)]

        # temporal kernel size is 1, padding is 0
        kernel_size_list = [(3, 3)] * 3 if is_spatial else [(1, 3), (1, 3), (1, 3)]
        padding_list = [(1, 1)] * 3 if is_spatial else [(0, 1), (0, 1), (0, 1)]

        self.in_layers = nn.Sequential(
            TriplaneNorm(channels),
            TriplaneSiLU(),
            TriplaneConv(
                channels,
                self.out_channels,
                kernel_size_list[0],
                padding=padding_list[0],
                is_rollout=is_rollout,
                is_spatial=is_spatial,
            ),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            TriplaneNorm(self.out_channels),
            TriplaneSiLU(),
            # nn.Dropout(p=dropout),
            zero_module(
                TriplaneConv(
                    self.out_channels,
                    self.out_channels,
                    kernel_size_list[1],
                    padding=padding_list[1],
                    is_rollout=is_rollout,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = TriplaneConv(
                channels,
                self.out_channels,
                kernel_size_list[2],
                padding=padding_list[2],
                is_rollout=False,
            )
        else:
            self.skip_connection = TriplaneConv(
                channels, self.out_channels, 1, padding=0, is_rollout=False
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # x: (h_xy, h_xz, h_yz)

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h[0].dtype)
        while len(emb_out.shape) < len(h[0].shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)

            h = out_norm(h)
            h_xy, h_xz, h_yz = h
            h_xy = h_xy * (1 + scale) + shift
            h_xz = h_xz * (1 + scale) + shift
            h_yz = h_yz * (1 + scale) + shift
            h = (h_xy, h_xz, h_yz)
            # h = out_norm(h) * (1 + scale) + shift

            h = out_rest(h)
        else:
            h_xy, h_xz, h_yz = h
            h_xy = h_xy + emb_out
            h_xz = h_xz + emb_out
            h_yz = h_yz + emb_out
            h = (h_xy, h_xz, h_yz)
            # h = h + emb_out

            h = self.out_layers(h)

        x_skip = self.skip_connection(x)
        x_skip_xy, x_skip_xz, x_skip_yz = x_skip
        h_xy, h_xz, h_yz = h
        return (h_xy + x_skip_xy, h_xz + x_skip_xz, h_yz + x_skip_yz)
        # return self.skip_connection(x) + h
