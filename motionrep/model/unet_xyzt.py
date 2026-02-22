from abc import abstractmethod

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
    TriplaneNorm,
    TriplaneSiLU,
    TimestepEmbedSequential,
    TriplaneConv,
    TriplaneDownsample2x,
    TriplaneUpsample2x,
    TriplaneResBlock,
)


class TemporalUNetModelSmall(nn.Module):
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
        is_spatial=False,
        num_temporal_downsample=2,
        is_rollout=True,
    ):
        super().__init__()

        # assert (
        #     len(channel_mult) - 1
        # ) == num_res_blocks, "invalid channel_mult or num_res_blocks"

        dims = 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        self.num_temporal_downsample = num_temporal_downsample

        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        # kernel 3 in the first layer
        self.in_conv = TimestepEmbedSequential(
            TriplaneConv(in_channels, ch, 3, padding=1, is_rollout=False)
        )
        print("In conv: TriplaneConv")

        # TODO: look at this input_block_chans.. currently num_res_blocks must be 1!
        input_block_chans = [ch]
        self.input_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult):
            layers = []
            if level != 0:
                if level <= num_temporal_downsample:
                    layers.append(TriplaneDownsample2x(only_last_chn=False))
                    print(f"Down level {level}: Downsample2x with temporal, ch {ch}")
                else:
                    layers.append(TriplaneDownsample2x(only_last_chn=is_spatial))
                    print(f"Down level {level}: Downsample2x without temporal, ch {ch}")

            for b_id in range(num_res_blocks):
                layers.append(
                    TriplaneResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_spatial=(
                            is_spatial or b_id == 0
                        ),  # add temporal conv for the first block
                        is_rollout=is_rollout and (not (is_spatial or b_id == 0)),
                    )
                )
                print(
                    f"Down level {level} block {b_id}: ResBlock, inp_chn {ch} out_chm {int(mult * model_channels)}"
                )
                ch = int(mult * model_channels)

            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            ich = input_block_chans.pop()
            for b_id in range(num_res_blocks):
                if level == len(channel_mult) - 1 and b_id == 0:
                    ich = 0

                if b_id == 0:
                    inp_chn = ch + ich
                else:
                    inp_chn = ch
                layers.append(
                    TriplaneResBlock(
                        inp_chn,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        is_spatial=(
                            is_spatial or b_id == 0
                        ),  # add temporal conv for the first block
                        is_rollout=is_rollout and (not (is_spatial or b_id == 0)),
                    )
                )
                print(
                    f"Up level {level} block {b_id}: ResBlock, inch {inp_chn} outch {int(model_channels * mult)}"
                )

                ch = int(model_channels * mult)
            if level > 0:
                if level <= num_temporal_downsample:
                    layers.append(TriplaneUpsample2x(only_last_chn=False))
                    print(f"Up level {level}: Upsample2x with temporal, ch {ch}")
                else:
                    layers.append(TriplaneUpsample2x(only_last_chn=is_spatial))
                    print(f"Up level {level}: Upsample2x without temporal, ch {ch}")

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
                TriplaneConv(
                    input_ch, out_channels, 1, padding=0, is_rollout=is_rollout
                )
            ),
        )
        print("Out conv: TriplaneConv")

        print(f"number of input blocks: {len(self.input_blocks)}")
        print(f"number of output blocks: {len(self.output_blocks) + 1}")

    def forward(self, x, timesteps, y=None):
        """
        Args:
            x: [B, C, Time_dim, Spatial_dim * 3]
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype) if y is None else th.cat([x, y], dim=1).type(self.dtype)

        # decompose. Then re-compose at the end
        tx, ty, tz = th.split(h, h.shape[-1] // 3, dim=-1)

        h_triplane = (tx, ty, tz)
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
        tx, ty, tz = h_triplane

        # re-compose
        output = th.cat([tx, ty, tz], dim=-1)

        return output
