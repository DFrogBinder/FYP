import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization='instaNorm', LeakyReLU_slope=0.2, dropout=False):
        super().__init__()
        block = []
        if dim == 2:
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True))
            if normalization == 'instaNorm':
                block.append(nn.InstanceNorm2d(out_channels))
            elif normalization == 'batchNorm':
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))

            block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=True))
            if normalization == 'instaNorm':
                block.append(nn.InstanceNorm2d(out_channels))
            elif normalization == 'batchNorm':
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))

            if dropout:
                block.append(nn.Dropout())


        elif dim == 3:
            block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,bias=True))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
            if normalization == 'instaNorm':
                block.append(nn.InstanceNorm3d(out_channels))
            elif normalization == 'batchNorm':
                block.append(nn.BatchNorm3d(out_channels))

            block.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1,bias=True))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
            if normalization == 'instaNorm':
                block.append(nn.InstanceNorm3d(out_channels))
            elif normalization == 'batchNorm':
                block.append(nn.BatchNorm3d(out_channels))
        else:
            raise (f'dim should be 2 or 3, got {dim}')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization):
        super().__init__()
        self.dim = dim
        if dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_block = ConvBlock(in_channels, out_channels, dim, normalization)

    def forward(self, x, skip):
        x_up = F.interpolate(x, skip.shape[2:], mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        x_up_conv = self.conv(x_up)
        out = torch.cat([x_up_conv, skip], 1)
        out = self.conv_block(out)
        return out