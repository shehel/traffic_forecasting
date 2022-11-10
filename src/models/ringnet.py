import torch
import torch.nn.functional as F
import torch.nn as nn

import pdb
class RingNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            ring_step,
            ring_end
    ):
        super(RingNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # create a list from 3 to 50 with step 2
        self.ring_sizes = list(range(3, ring_end, ring_step))


        # create a 2d array of ones, size 5x5 with outer border elements set to 0
        for idx,ring_s in enumerate(self.ring_sizes):
            tensor = torch.ones((in_channels, 1, ring_s, ring_s))
            tensor[:, :, 1:-1, 1:-1] = 0
            self.register_buffer('ring'+str(idx), tensor)

        self.model = nn.Sequential(
                    nn.BatchNorm2d(1248),
                    nn.Conv2d(1248, 512, kernel_size=[1, 1], stride=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(512, out_channels, kernel_size=[1, 1], stride=(1, 1))
            )



    def forward(self, x):
        # create a list of 2d convolutions with different kernel sizes
        convs = [x]

        # Loop through the internal saved buffers holding the ring weights
        for ring in range(len(self.ring_sizes)):
            convs.append(F.conv2d(x, self._buffers['ring'+str(ring)], padding='same', groups=self.in_channels))

        # concatenate the convolutions
        x = torch.cat(convs, dim=1)
        x = self.model(x)

        return x
