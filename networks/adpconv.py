import torch
from conv_custom import AdaptiveDilatedConv
x = torch.rand(2, 4, 16, 16).cuda()
#m = AdaptiveDilatedConv(in_channels=4, out_channels=4, kernel_size=5).cuda()
#y = m(x)
#print('out1',y.shape)
from model_utils import AttentionModule, DeformableConv
att = AttentionModule(in_channels=4).cuda()
out = att(x)
print('out2', out.shape)
#print("Adaptive Weight Shape (before reshape):", adaptive_weight.shape)