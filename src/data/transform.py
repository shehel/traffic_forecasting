import torch
import numpy as np
from typing import Optional, Tuple

from src.data.data_transform import DataTransform
import pywt
from pytorch_wavelets import DWTForward, DWTInverse


import pdb
class UNetTransform(DataTransform):
    """Pre-processor and post-processor to convert T4C data to
    be compatible with Unet
    Args:
        stack_time: Decides if the time channels are stacked upon each other
        pre_batch_dim: Whether batch dimension is present in the data provided
                       to pre-processor
        post_batch_dim: Whether batch dimension is present in the data provided
                        to post-processor
        crop_pad: _dim: Tuple of pixels to crop/pad in each side
    """

    def __init__(self, stack_time: bool = False, pre_batch_dim: bool = False,
                 post_batch_dim: bool = True,
                 num_channels: int = 8,
                 crop_pad: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.stack_time = stack_time
        self.pre_batch_dim = pre_batch_dim
        self.post_batch_dim = post_batch_dim
        self.crop_pad = crop_pad
        self.num_channels = num_channels
        self.xfm = DWTForward(J=1, wave='db7', mode='zero')
    def pre_transform(
        self,
        data: np.ndarray,
        from_numpy: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Transform data from `T4CDataset` be used by UNet:
        - put time and channels into one dimension
        - padding
        """
        if from_numpy:
            data = torch.from_numpy(data).float()

        if not self.pre_batch_dim:
            data = torch.unsqueeze(data, 0)

        if self.stack_time:
            data = self.stack_on_time(data, batch_dim=True)

        Yl, Yh = self.xfm(data)
        if Yl.shape[1] == 2000:
            Yh[0] = Yh[0][:, :, :2, :, :]
            data = torch.cat((Yl, Yh[0].reshape(1, data.shape[1]*2, Yl.shape[-2], Yl.shape[-1])), 1)
        else:
            data = torch.cat((Yl, Yh[0].reshape(1, data.shape[1]*3, Yl.shape[-2], Yl.shape[-1])), 1)
        if self.crop_pad is not None:
            zeropad2d = torch.nn.ZeroPad2d(self.crop_pad)
            data = zeropad2d(data)
        if not self.pre_batch_dim:
            data = torch.squeeze(data, 0)

        return data

    def post_transform(
            self, data: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Bring data from UNet back to `T4CDataset` format:

        - separats common dimension for time and channels
        - cropping
        """
        if not self.post_batch_dim:
            data = torch.unsqueeze(data, 0)

        if self.crop_pad is not None:
            _, _, height, width = data.shape
            left, right, top, bottom = self.crop_pad
            right = width - right
            bottom = height - bottom
            data = data[:, :, top:bottom, left:right]
        # if self.stack_time:
        #     data = self.unstack_on_time(data, batch_dim=True)
        if not self.post_batch_dim:
            data = torch.squeeze(data, 0)
        return data

    def stack_on_time(self, data: torch.Tensor, batch_dim: bool = False):
        """
        `(k, 12, 495, 436, 8) -> (k, 12 * 8, 495, 436)`
        """

        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)

        _, num_time_steps, height, width, num_channels = data.shape
        # (k, 12, 495, 436, 8) -> (k, 12, 8, 495, 436)
        data = torch.moveaxis(data, 4, 2)

        # (k, 12, 8, 495, 436) -> (k, 12 * 8, 495, 436)
        data = torch.reshape(data, (data.shape[0],
                                    num_time_steps * num_channels,
                                    height,
                                    width))

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data

    def unstack_on_time(self, data: torch.Tensor, batch_dim:bool = False):
        """
        `(k, 12 * 8, 495, 436) -> (k, 12, 495, 436, 8)`
        """
        _, _, height, width = data.shape
        if not batch_dim:
            # `(12, 495, 436, 8) -> (1, 12, 495, 436, 8)`
            data = torch.unsqueeze(data, 0)

        num_time_steps = int(data.shape[1] / self.num_channels)
        # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
        data = torch.reshape(data, (data.shape[0],
                                    num_time_steps,
                                    self.num_channels,
                                    height,
                                    width))

        # (k, 12, 8, 495, 436) -> (k, 12, 495, 436, 8)
        data = torch.moveaxis(data, 2, 4)

        if not batch_dim:
            # `(1, 12, 495, 436, 8) -> (12, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data
