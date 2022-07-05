"""UNet implementation from https://github.com/jvanvugt/pytorch-unet.

Copied from https://github.com/mie-lab/traffic4cast/blob/aea6f90e8884c01689c84255c99e96d2b58dc470/models/unet.py with permission
"""
#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional
from typing import Tuple

import math
import numpy as np
import torch
from torch import nn
import pdb
from einops import rearrange
class UNet(nn.Module):
    def __init__(
            self, dim = 12, in_channels=1, out_channels=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode="upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, time_dim))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, time_dim))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, inp, *args, **kwargs):
        x,time = inp

        t = self.time_mlp(time)
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x, t)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, t, blocks[-i - 1])
        x=self.last(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, time_emb_dim):
        super(UNetConvBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_size))
        )
        block1 = []
        block2 = []

        block1.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block1.append(nn.ReLU())
        if batch_norm:
            block1.append(nn.BatchNorm2d(out_size))

        block2.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block2.append(nn.ReLU())
        if batch_norm:
            block2.append(nn.BatchNorm2d(out_size))

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)

    def forward(self, x, time_emb):  # noqa
        x = self.block1(x)
        time_emb = self.mlp(time_emb)
        x = rearrange(time_emb, 'b c -> b c 1 1 1') + x
        x = self.block2(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, time_emb_dim):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1),)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, time_emb_dim)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_y_target_size_ = diff_y + target_size[0]
        diff_x_target_size_ = diff_x + target_size[1]
        return layer[:, :, diff_y:diff_y_target_size_, diff_x:diff_x_target_size_]

    def forward(self, x, t, bridge):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out, t)

        return out
