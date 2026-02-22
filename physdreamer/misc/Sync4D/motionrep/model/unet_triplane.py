import torch as th
import torch.nn as nn
import torch.nn.functional as F

from motionrep.model.fp16_utils import convert_module_to_f16, convert_module_to_f32
from motionrep.model.nn_utils import (
    checkpoint,
    conv_nd,
    linear,
    SiLU,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


def compose_featmaps(feat_xy, feat_xz, feat_yz):
    H, W = feat_xy.shape[-2:]
    D = feat_xz.shape[-1]

    empty_block = th.zeros(
        list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device
    )
    composed_map = th.cat(
        [
            th.cat([feat_xy, feat_xz], dim=-1),
            th.cat([feat_yz.transpose(-1, -2), empty_block], dim=-1),
        ],
        dim=-2,
    )
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, sizes):
    H, W, D = sizes
    feat_xy = composed_map[..., :H, :W]  # (C, H, W)
    feat_xz = composed_map[..., :H, W:]  # (C, H, D)
    feat_yz = composed_map[..., H:, :W].transpose(-1, -2)  # (C, W, D)
    return feat_xy, feat_xz, feat_yz


class TriplaneUNetModelSmall(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks=1,
        dropout=0,
        channel_mult=(1, 2),
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        dims = 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.in_conv = TimestepEmbedSequential(
            TriplaneConv(in_channels, ch, 1, padding=0, is_rollout=False)
        )
        print("In conv: TriplaneConv")

        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if level != 0:
                layers.append(TriplaneDownsample2x())
                print(f"Down level {level}: TriplaneDownsample2x, ch {ch}")

            for _ in range(num_res_blocks):
                layers.append(
                    TriplaneResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                print(f"Down level {level} block {_}: TriplaneResBlock, ch {ch}")

            ch = int(mult * model_channels)
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    ich = 0
                layers.append(
                    TriplaneResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                print(f"Up level {level} block {i}: TriplaneResBlock, ch {ch}")

            ch = int(model_channels * mult)
            if level > 0:
                layers.append(TriplaneUpsample2x())
                print(f"Up level {level}: TriplaneUpsample2x, ch {ch}")

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        # self.out = nn.Sequential(
        #     normalization(ch),
        #     SiLU(),
        #     zero_module(conv_nd(dims, input_ch, out_channels, 1, padding=0)),
        # )
        self.out = nn.Sequential(
            TriplaneNorm(ch),
            TriplaneSiLU(),
            zero_module(
                TriplaneConv(input_ch, out_channels, 1, padding=0, is_rollout=False)
            ),
        )
        print("Out conv: TriplaneConv")

        print(f"number of input blocks: {len(self.input_blocks)}")
        print(f"number of output blocks: {len(self.output_blocks) + 1}")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, H=None, W=None, D=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert H is not None and W is not None and D is not None

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype) if y is None else th.cat([x, y], dim=1).type(self.dtype)
        h_triplane = decompose_featmaps(h, (H, W, D))

        h_triplane = self.in_conv(h_triplane, emb)

        for level, module in enumerate(self.input_blocks):
            h_triplane = module(h_triplane, emb)
            hs.append(h_triplane)

        for level, module in enumerate(self.output_blocks):
            if level == 0:
                h_triplane = hs.pop()
            else:
                h_triplane_pop = hs.pop()
                h_triplane = list(h_triplane)
                if h_triplane[0].shape[2:] != h_triplane_pop[0].shape[2:]:
                    h_triplane[0] = F.interpolate(
                        h_triplane[0],
                        size=h_triplane_pop[0].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if h_triplane[1].shape[2:] != h_triplane_pop[1].shape[2:]:
                    h_triplane[1] = F.interpolate(
                        h_triplane[1],
                        size=h_triplane_pop[1].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if h_triplane[2].shape[2:] != h_triplane_pop[2].shape[2:]:
                    h_triplane[2] = F.interpolate(
                        h_triplane[2],
                        size=h_triplane_pop[2].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                h_triplane = (
                    th.cat([h_triplane[0], h_triplane_pop[0]], dim=1),
                    th.cat([h_triplane[1], h_triplane_pop[1]], dim=1),
                    th.cat([h_triplane[2], h_triplane_pop[2]], dim=1),
                )

            h_triplane = module(h_triplane, emb)

        h_triplane = self.out(h_triplane)
        h = compose_featmaps(*h_triplane)[0]
        assert h.shape == x.shape
        return h
