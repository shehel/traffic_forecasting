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

from torch import nn
import pdb

class AR(nn.Module):
    def __init__(
        self,
            in_channels: 96,
            out_channels: 48
    ):
        """
        Implementation of
        a linear model
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super(AR, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)
