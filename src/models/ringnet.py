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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # create a list from 3 to 50 with step 2
        self.ring_sizes = list(range(3, ring_end, ring_step))


        # create a 2d array of ones, size 5x5 with outer border elements set to 0
        self.rings = []
        for ring_s in self.ring_sizes:
            ring = nn.Parameter(torch.ones((in_channels, 1, ring_s, ring_s)), requires_grad=False).cuda()

            ring[:, :, 1:-1, 1:-1] = 0
            self.rings.append(ring)

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
        for ring in self.rings:
            convs.append(F.conv2d(x, ring, padding='same', groups=self.in_channels))

        # concatenate the convolutions
        x = torch.cat(convs, dim=1)
        x = self.model(x)

        return x
