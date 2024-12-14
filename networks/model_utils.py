import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
from conv_custom import AdaptiveDilatedConv

##########
import torch
import torch.nn as nn
from conv_custom import AdaptiveDilatedConv  # Import your custom AdaptiveDilatedConv module

class InceptionLKAModule(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Inception Adaptive Dilated LKA (I-ADLKA) module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        """
        super(InceptionLKAModule, self).__init__()

        # Initial convolution to process input features
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        # Adaptive dilated convolutions
        self.spatial_convs = nn.ModuleList([
            AdaptiveDilatedConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3).cuda(),
            #AdaptiveDilatedConv(in_channels=in_channels, out_channels=in_channels, kernel_size=5).cuda(),
            #AdaptiveDilatedConv(in_channels=in_channels, out_channels=in_channels, kernel_size=7).cuda()
        ])

        # Pointwise convolution for channel-wise refinement
        #self.conv1 = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)
        self.conv1 = nn.Conv2d( in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()

        # Initial convolution for global context
        attn = self.conv0(x)

        # Apply adaptive dilated convolutions
        spatial_attns = [conv(attn) for conv in self.spatial_convs]

        # Concatenate results and refine with a 1x1 convolution
        attn = torch.cat(spatial_attns, dim=1)
        attn = self.conv1(attn)

        # Return input modulated by attention
        return original_input * attn

###########

class InceptionLKAModule2(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Inception LKA (I-LKA) module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        """
        super(InceptionLKAModule2, self).__init__()
        
        kernels = [3, 5, 7]
        paddings = [1, 4, 9]
        dilations = [1, 2, 3]
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.spatial_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=kernels[i], stride=1,
                                                    padding=paddings[i], groups=in_channels,
                                                    dilation=dilations[i]) for i in range(len(kernels))])
        self.conv1 = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1)
        

    def forward(self, x):
        original_input = x.clone()
        attn = self.conv0(x)
        spatial_attns = [conv(attn) for conv in self.spatial_convs]
        attn = torch.cat(spatial_attns, dim=1)
        attn = self.conv1(attn)
        return original_input * attn
    


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Attention module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        
        :return: Output tensor after applying attention module
        """
        super(AttentionModule, self).__init__()
        self.proj_1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating = InceptionLKAModule(in_channels)
        #self.spatial_gating = AdaptiveDilatedConv(in_channels, in_channels, kernel_size=3).cuda()
        self.proj_1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        x = self.proj_1x1_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_1x1_2(x)
        
        return original_input + x
    
    
class DeformableConv(nn.Module):
    def __init__(self, nChannel, batch=1, height=256, width=256):
        """
        Initialize the Deformable Convolution module.

        :param batch: Batch size
        :type batch: int
        :param height: Height of input feature maps
        :type height: int
        :param width: Width of input feature maps
        :type width: int
        
        :return: Output tensor after applying deformable convolution
        """
        super().__init__()
        kh, kw = 3, 3
        self.weight = nn.Parameter(torch.rand(nChannel, nChannel, kh, kw))
        self.offset = nn.Parameter(torch.rand(batch, 2 * kh * kw, height, width))
        self.mask = nn.Parameter(torch.rand(batch, kh * kw, height, width))

    def forward(self, x):
        return deform_conv2d(x, self.offset, self.weight, mask=self.mask, padding=1)
    

if __name__ == '__main__':
    x = torch.rand(2, 4, 16, 16).cuda()
    m = InceptionLKAModule2(in_channels=4).cuda()
    m.eval()
    y = m(x)
    print(y.shape)
    