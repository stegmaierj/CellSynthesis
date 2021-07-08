# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:09:03 2020

"""


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F



class PixelShuffle3d(nn.Module):
    '''
    reference: http://www.multisilicon.com/blog/a25332339.html
    '''
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    
    


class GaussianSmoothing(nn.Module):
    """
    @author: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. 
    Filtering is performed seperately for each channel in the input,
    using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. 
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 3 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3, padding=0):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)





class PoolSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. 
    Arguments:
        kernel_size (int, sequence): Size of the pooling kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 3 (spatial).
    """
    def __init__(self, kernel_size, dim=3, padding=0, iterations=1):
        
        super(PoolSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
            
        self.kernel_size=kernel_size
        self.padding = padding
        self.iterations = iterations

        if dim == 1:
            self.pool = F.avg_pool1d
        elif dim == 2:
            self.pool = F.avg_pool2d
        elif dim == 3:
            self.pool = F.avg_pool3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply pool smoothing to input.
        Arguments:
            input (torch.Tensor): Input to apply pooling on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        for i in range(self.iterations):
            input = self.pool(input, kernel_size=self.kernel_size, padding=self.padding, count_include_pad=False, stride=1)
        
        return input
