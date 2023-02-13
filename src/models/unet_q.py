"""UNet implementation from https://github.com/jvanvugt/pytorch-unet.
adapted for conformalized quantile regression
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

import numpy as np
import torch
from torch import nn
import pdb
from einops import rearrange


class UNetQ(nn.Module):
    def __init__(
        self, params, in_channels=1, out_channels=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode="upconv",
        pos_emb=False, 
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
        super(UNetQ, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.q_lo = params["q_lo"]
        self.q_hi = params["q_hi"]
        self.params = params

        


        if pos_emb:
            self.pos_model = Date2Vec()
        else:
            self.pos_model = None
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, time_emb_dim=6 if pos_emb else None))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, time_emb_dim=6 if pos_emb else None))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        self.lower = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        self.upper = nn.Conv2d(prev_channels, out_channels, kernel_size=1)


    def forward(self, x, *args, **kwargs):
        x, t  = x
        t = self.pos_model(t) if exists(self.pos_model) else None
        blocks = []
        for i, down in enumerate(self.down_path):
            if i == 0:
                x = down(x, t)
            else:
                x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        #prev = x
        #x=self.last(x)
        x = torch.cat((self.lower(x).unsqueeze(1), self.last(x).unsqueeze(1), self.upper(x).unsqueeze(1)), dim=1)

        return x#, prev

# helper functions
def exists(x):
    return x is not None

class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(1, k1)

        self.fc2 = nn.Linear(1, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(k // 2, 6)
        
        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(dim_out)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, 
                time_emb_dim: Optional[int] = None,):
        super(UNetConvBlock, self).__init__()
        
        self.mlp = (nn.Sequential(
            nn.ReLU(), nn.Linear(time_emb_dim, out_size*2))
            if exists(time_emb_dim) else None)

        self.block1 = Block(in_size, out_size)
        self.block2 = Block(out_size, out_size)
        self.res_conv = nn.Conv2d(in_size, out_size, 1) if in_size != out_size else nn.Identity()


    def forward(self, x, time_emb=None):  # noqa
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        out = self.block1(x, scale_shift = scale_shift)
        out = self.block2(out)

        return out + self.res_conv(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, time_emb_dim: Optional[int] = None,):
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

    def forward(self, x, bridge, time_emb=None):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out, time_emb)

        return out

