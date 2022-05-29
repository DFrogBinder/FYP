from . import unet
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from utils.SpatialTransformer import SpatialTransformer

class RegNet_single(nn.Module):
    '''
    Groupwise implicit template CNN registration method.
    Parameters
    ----------
    dim : int
        Dimension of input image.
    n : int
        Number of image in the group.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : int, optional
        Number of initial channels. The default is 64.
    normalization : int, optional
        Whether to add instance normalization after activation. The default is True.
    '''

    def __init__(self, dim, n, scale=1, depth=5, initial_channels=64, normalization='instaNorm'):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.n = n
        self.scale = scale

        self.unet = unet.UNet(in_channels=n, out_channels=dim * n, dim=dim, depth=depth,
                              initial_channels=initial_channels, normalization=normalization)
        self.spatial_transform = SpatialTransformer(self.dim)

    def forward(self, input_image):
        '''
        Parameters
        ----------
        input_image : (n, 1, h, w) or (n, 1, d, h, w)
            The first dimension contains the grouped input images.
        Returns
        -------
        warped_input_image : (n, 1, h, w) or (n, 1, d, h, w)
            Warped input image.
        template : (1, 1, h, w) or (1, 1, d, h, w)
            Implicit template image derived by averaging the warped_input_image
        disp_t2i : (n, 2, h, w) or (n, 3, d, h, w)
            Flow field from implicit template to input image. The starting point of the displacement is on the regular grid defined on the implicit template and the ending point corresponding to the same structure in the input image.
        warped_template : (n, 1, h, w) or (n, 1, d, h, w)
            Warped template images that should match the original input image.
        disp_i2t : (n, 2, h, w) or (n, 3, d, h, w)
            Flow field from input image to implicit template. The starting point of the displacement is on the regular grid defined on the input image and the ending point corresponding to the same structure in the implicit template.
        '''

        original_image_shape = input_image.shape[2:]
        print("input_image : (n, 1, h, w) or (n, 1, d, h, w) The first dimension contains the grouped input images.")
        print("Original Image Shape in regnet.py: "+str(original_image_shape))
        # plt.imshow(input_image[10,0,:,:])
        # plt.show()
        if self.scale < 1:
            scaled_image = F.interpolate(torch.transpose(input_image, 0, 1), scale_factor=self.scale,
                                         align_corners=True, mode='bilinear' if self.dim == 2 else 'trilinear',
                                         recompute_scale_factor=False)  # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = torch.transpose(input_image, 0, 1)

        scaled_image_shape = scaled_image.shape[2:]
        scaled_disp_t2i = torch.squeeze(self.unet(scaled_image), 0).reshape(self.n, self.dim,
                                                                            *scaled_image_shape)  # (n, 2, h, w) or (n, 3, d, h, w)
        if self.scale < 1:
            disp_t2i = torch.nn.functional.interpolate(scaled_disp_t2i, size=original_image_shape,
                                                       mode='bilinear' if self.dim == 2 else 'trilinear',
                                                       align_corners=True)
        else:
            disp_t2i = scaled_disp_t2i

        warped_input_image = self.spatial_transform(input_image, disp_t2i)  # (n, 1, h, w) or (n, 1, d, h, w)
        template_mean = torch.mean(warped_input_image, 0, keepdim=True)  # (1, 1, h, w) or (1, 1, d, h, w)
        print("Original template shape is: "+str(template_mean))
        template = torch.reshape(input_image[1,0,:,:],[1,1,384,384])

        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image,
               'template': template}

        if self.scale < 1:
            scaled_template = torch.nn.functional.interpolate(template, size=scaled_image_shape,
                                                              mode='bilinear' if self.dim == 2 else 'trilinear',
                                                              align_corners=True)
        else:
            scaled_template = template
        res = {'disp_t2i': disp_t2i, 'scaled_disp_t2i': scaled_disp_t2i, 'warped_input_image': warped_input_image,
               'template': template, 'scaled_template': scaled_template}

        return res

