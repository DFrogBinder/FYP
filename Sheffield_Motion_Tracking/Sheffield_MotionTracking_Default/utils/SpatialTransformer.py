import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image


    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow, mode='bilinear'):
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)

        return:
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape])
            grid = torch.stack(grids[::-1],
                               dim=0)  # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid = torch.unsqueeze(grid, 0)
            grid = grid.to(dtype=flow.dtype, device=flow.device)
            norm_coeff = 2. / (torch.tensor(img_shape[::-1], dtype=flow.dtype,
                                            device=flow.device) - 1.)  # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff
            # logging.info(f'\nAdd grid shape {tuple(img_shape)}')
        new_grid = grid + flow

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1)  # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1)  # n x d x h x w x 3

        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img = F.grid_sample(input_image, new_grid * norm_coeff - 1., mode=mode, align_corners=True,
                                         padding_mode='border')
        return warped_input_img