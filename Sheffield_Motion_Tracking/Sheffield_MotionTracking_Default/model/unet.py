import torch
from torch import nn
import torch.nn.functional as F

from model.layers import ConvBlock, UpBlock
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class UNet(nn.Module):
    '''
    U-net implementation with modifications.
        1. Works for input of 2D or 3D
        2. Change batch normalization to instance normalization

    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels : int
        number of output channels.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    normalization : bool, optional
        Whether to add instance normalization after activation. The default is False.
    '''

    def __init__(self, in_channels, out_channels, dim=2, depth=5, initial_channels=32, normalization='instaNorm'):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            current_channels = 2 ** i * initial_channels
            if i== self.depth-1:
                self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization, dropout=True))
            else:
                self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization, dropout=True))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 2 ** i * initial_channels
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                  align_corners=True, recompute_scale_factor=False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


