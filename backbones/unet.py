import math
from typing import Union
import torch
import torch.nn as nn
import sys
from backbones.model_tcn import TlioTcn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels,
                      kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels,
                        kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels,
                        kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            # Rearrange('batch t -> batch t 1'),
            nn.Unflatten(1, [-1, 1])
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class UNet1DConditional(nn.Module):

    def __init__(self,
                 input_dim=6,
                 local_cond_dim=None,
                 global_cond_dim=256,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False
                 ):
        super(UNet1DConditional, self).__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed

        global_cond_encoder = None
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
            global_cond_encoder = TlioTcn(
                input_size=6,
                output_size=global_cond_dim,
                num_channels=[64, 64, 64, 64, 128, 128, 128],
                kernel_size=2,
                dropout=0.2,
                activation="GELU"
            )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.global_cond_encoder = global_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond_input=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.

        """
        # sample = einops.rearrange(sample, 'b h t -> b t h')
        sample = sample.transpose(1, 2)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        #embdeds timestep for denoising process
        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond_input is not None:
            assert self.global_cond_encoder is not None
            global_cond = self.global_cond_encoder(global_cond_input)
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # encode local features
        h_local = list()
        if local_cond is not None:
            # local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            local_cond = local_cond.transpose(1, 2)
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        #unet architecture
        #downsampling part
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        #middle of unet architecture
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        # print("h shape", [h_.shape for h_ in h])
        
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            feat = h.pop()
            x = torch.cat((x, feat), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')
        x = x.transpose(1, 2)

        return (x, )