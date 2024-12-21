import numbers
import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

# from .networks.segformer import *
from .model_utils import AttentionModule as AdaptiveAttentionModule
from .segformer import *
from .masag import MultiScaleGatedAttn
from .merit_lib.networks import MaxViT4Out_Small
# from segformer import *
# from attentions import MultiScaleGatedAttn

from timm.models.layers import DropPath, to_2tuple
import math

##################################
#
#            Modules
#
##################################

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, isQuery = True):
        super().__init__()
        self.isQuery =isQuery
        inner_dim = dim_head *  heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, 2*inner_dim, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim 
            kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1] 
            return k,v

class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        # print("LKA return shape: {}".format(x.shape))
        return x

class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        ## Here channel weighting and Eigenvalues
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        # print("Dual transformer return shape: {}".format(mx.shape))
        return mx


class Attention_st(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_st, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   



        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_st_cross(nn.Module):
    def __init__(self, dim, num_heads, bias):
        dim = dim //2
        super(Attention_st_cross, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, 2*dim*2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(2*dim*2, 2*dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(2*dim, 2*dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        kv = self.kv_dwconv(self.kv(x[:, :c//2]))
        k,v = kv.chunk(2, dim=1)   
        q = self.q(x[:, :c//2])

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNormst(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNormst, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, cross= False):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNormst(dim, LayerNorm_type)
        if not cross:
            self.attn = Attention_st(dim, num_heads, bias)
        else:
            self.attn = Attention_st_cross(dim, num_heads, bias)
        self.norm2 = LayerNormst(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class LightWeightPromptGenBlock(nn.Module):
    def __init__(self,input_size, prompt_dim=48, prompt_len=5 ,lin_dim = 192):
        super().__init__()

        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim, input_size, input_size)) # B, N , C, H, W
        
        self.linear_layer = nn.Linear(lin_dim,prompt_len)

        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1)) # B, C
        #print(emb.shape)

        prompt_weights = F.softmax(self.linear_layer(emb),dim=1) # B, C , C = 5
        #print(prompt_weights.shape)

        #prompt_ = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #print(prompt_.shape)
        
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #print(prompt.shape)

        #prompt__ = self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #print(prompt__.shape)
        
        prompt = torch.sum(prompt,dim=1)
        #print(prompt.shape)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        

        return prompt
    

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # print(x_pooled.shape)
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)
        print(y.shape)

        y = self.fc(y).view(n, c, 1, 1)
        print(y.shape)
        return x * y.expand_as(x)

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

class FreqLightWeightPromptGenBlock(nn.Module):
    def __init__(self, 
                 dct_h,
                 dct_w,
                 input_size,  
                 prompt_dim=48, 
                 prompt_len=5, 
                 lin_dim = 192,
                 freq_sel_method = 'top16'):
        
        super().__init__()

        self.dct_h = dct_h
        self.dct_w = dct_w
        input_size_w = input_size // 2 + 1

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim, input_size, input_size_w, 2)) # B, N , C, H, (W//2+1)

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, prompt_dim)
        
        self.linear_layer = nn.Linear(lin_dim,prompt_len)

        self.conv3x3 = nn.Conv3d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape

        w = (W // 2) + 1
        emb = self.dct_layer(x)
        #print(emb.shape)
        # emb = x.mean(dim=(-2,-1)) # B, C (Simple GAP)

        prompt_weights = F.softmax(self.linear_layer(emb),dim=1) # B, C , C = 5
        
        p1 = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #print(p1.shape)
        # print(self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1, 1).squeeze(1).shape)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1, 1).squeeze(1)

        # p2 = self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        # print(p2.shape)
        #print(prompt.shape)
        prompt = torch.sum(prompt,dim=1)
        #print(prompt.shape)
        prompt = F.interpolate(prompt,(H,w, 2),mode="trilinear") # B, N, C, (W//2 + 1)
        prompt = self.conv3x3(prompt)
        

        return prompt

class FrqRefiner(nn.Module):
    def __init__(self, dim=3,h=128,w=128):
        super().__init__()
        self.h = h
        self.w = w
        # w = w//2 + 1
        self.complex_weights = FreqLightWeightPromptGenBlock(dct_h= h,
                                                             dct_w= h,
                                                             input_size=h,
                                                             prompt_dim=dim,
                                                             prompt_len=5,
                                                             lin_dim=dim)
        #self.complex_weight = nn.Parameter(torch.randn(1,dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.body = nn.Sequential(nn.Conv2d(2*dim,2*dim,kernel_size=1,stride=1),
                                    nn.GELU())
        # self.kv_conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1))

    def forward(self, x, H = None, W = None):
        # if self.flag_highF:
        #     ffm = F.interpolate(ffm, size=(H, W), mode='bilinear')
        #     y_att = self.body(ffm)

        #     y_f = y_att * ffm
        #     y = y_f * self.weight
        if self.h != None and self.w != None:
            H = self.h
            W = self.w
        
        x = F.interpolate(x, size=(H, W), mode='bicubic')
        y = torch.fft.rfft2(x.to(torch.float32).cuda())
        # print(y.shape)

        y_imag = y.imag

        # print(y_imag.shape)
        y_real = y.real
        # print(y_real.shape)
        y_f = torch.cat([y_real, y_imag], dim=1)

        # print(y_f.shape)

        ## Weight Making ##
 
        weight = torch.complex(self.complex_weights(x)[..., 0],self.complex_weights(x)[..., 1])

        ########
        # print(self.complex_weights(x)[..., 0].shape)
        # print(weight.shape)
        # print("shape is : ", weight.shape)
        
        y_att = self.body(y_f)
        # print(y_att.shape)

        y_f = y_f * y_att
        # print(y_f.shape)
        
        y_real, y_imag = torch.chunk(y_f, 2, dim=1)
        # print(y_real.shape, y_imag.shape)
        y = torch.complex(y_real, y_imag)
        y = y * weight
        y = torch.fft.irfft2(y, s=(H, W))
        
        return y


class FrqRefinerEnhanced(nn.Module):
    def __init__(self, dim=3,h=128,w=128):
        super().__init__()
        self.h = h
        self.w = w
        # w = w//2 + 1
        self.complex_weights = FreqLightWeightPromptGenBlock(dct_h= h,
                                                             dct_w= h,
                                                             input_size=h,
                                                             prompt_dim=dim,
                                                             prompt_len=5,
                                                             lin_dim=dim)
        
        self.body = nn.Sequential(nn.Conv2d(2*dim,2*dim,kernel_size=1,stride=1),
                                    nn.GELU())
        
        self.conv_enhancer = nn.Sequential(nn.Conv2d(dim,
                                                     dim,
                                                     kernel_size=1,
                                                     stride=1
                                                     ),
                                          nn.BatchNorm2d(dim),
                                          nn.GELU())
        
    def forward(self, x, H = None, W = None):
        
        if self.h != None and self.w != None:
            H = self.h
            W = self.w
        
        x = F.interpolate(x, size=(H, W), mode='bicubic')
        y = torch.fft.rfft2(x.to(torch.float32).cuda())
        # print(y.shape)

        y_imag = y.imag

        # print(y_imag.shape)
        y_real = y.real
        # print(y_real.shape)
        y_f = torch.cat([y_real, y_imag], dim=1)

        # print(y_f.shape)

        ## Weight Making ##
 
        weight = torch.complex(self.complex_weights(x)[..., 0],self.complex_weights(x)[..., 1])

        ########
        # print(self.complex_weights(x)[..., 0].shape)
        # print(weight.shape)
        # print("shape is : ", weight.shape)
        
        y_att = self.body(y_f)
        # print(y_att.shape)

        y_f = y_f * y_att
        # print(y_f.shape)
        
        y_real, y_imag = torch.chunk(y_f, 2, dim=1)
        # print(y_real.shape, y_imag.shape)
        y = torch.complex(y_real, y_imag)
        y = y * weight
        y = torch.fft.irfft2(y, s=(H, W))
        y = self.conv_enhancer(y)
        
        return y

class FrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak,win_size, num_heads, qkv_bias=True, qk_scale=None, bias=False):
        super(FrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.win_size = win_size  # Wh, Ww
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias,isQuery=True)
        self.to_kv = LinearProjection(dim_bak,num_heads,dim//num_heads,bias=qkv_bias,isQuery=False)
        
        self.kv_dwconv = nn.Conv2d(dim_bak , dim_bak, kernel_size=3, stride=1, padding=1, groups=dim_bak, bias=bias)
        
        self.softmax = nn.Softmax(dim=-1)

        self.project_out = nn.Linear(dim, dim)

    def forward(self, query_feature, key_value_feature):

        b,c,h,w = query_feature.shape
        _,c_2,_,_ = key_value_feature.shape
        
        key_value_feature = self.kv_dwconv(key_value_feature)
        
        # partition windows
        query_feature = rearrange(query_feature, ' b c1 h w -> b h w c1 ', h=h, w=w)
        query_feature_windows = window_partition(query_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        query_feature_windows = query_feature_windows.view(-1, self.win_size * self.win_size, c)  # nW*B, win_size*win_size, C
        
        key_value_feature = rearrange(key_value_feature, ' b c2 h w -> b h w c2 ', h=h, w=w)
        key_value_feature_windows = window_partition(key_value_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        key_value_feature_windows = key_value_feature_windows.view(-1, self.win_size * self.win_size, c_2)  # nW*B, win_size*win_size, C
        
        B_, N, C = query_feature_windows.shape
        
        query = self.to_q(query_feature_windows)
        query = query * self.scale
        
        key,value = self.to_kv(key_value_feature_windows)
        attn = (query @ key.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)

        out = (attn @ value).transpose(1, 2).contiguous().reshape(B_, N, C)

        out = self.project_out(out)

        # merge windows
        attn_windows = out.view(-1, self.win_size, self.win_size, C)
        attn_windows = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C
        return rearrange(attn_windows, 'b h w c -> b c h w', h=h, w=w)


class FrequencyPromptFusionEnhanced(nn.Module):
    def __init__(self, dim, dim_bak, num_heads,win_size=8, bias=False):
        super(FrequencyPromptFusionEnhanced, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ap_kv = nn.AdaptiveAvgPool2d(1)
        self.kv = nn.Conv2d(dim_bak, dim * 2, kernel_size=1, bias=bias)

        self.project_out = nn.Sequential(nn.Conv2d( dim, dim, kernel_size=1, bias=bias),
                                         nn.BatchNorm2d(dim))

    def forward(self, feature, prompt_feature):
        b, c1,h,w = feature.shape
        _, c2,_,_ = prompt_feature.shape

        query = self.q(feature).reshape(b, h * w, self.num_heads, c1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        prompt_feature = self.ap_kv(prompt_feature)#.reshape(b, c2, -1).permute(0, 2, 1)
        key_value = self.kv(prompt_feature).reshape(b, 2*c1, -1).permute(0, 2, 1).contiguous().reshape(b, -1, 2, self.num_heads, c1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]

        attn = (query @ key.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ value)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out
    

##########################################
#
#         General Decoder Blocks
#
##########################################
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x

##########################################
#
#         MSA^2Net Decoder Blocks
#
##########################################

class MyDecoderLayerLKA(nn.Module):
    def __init__(self,
                 input_size: tuple,
                 in_out_chan: tuple,
                 n_class=9,
                 norm_layer=nn.LayerNorm,
                 is_last=False):
        
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        self.layer_lka_2 = LKABlock(dim=out_dim)

        

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            #attn_gate = self.ag_attn(x=x2_new, g=x1_expand)  # B C H W

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            # print(tran_layer_1.shape)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayerLKAPrompt(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.prompt1 = LightWeightPromptGenBlock(prompt_dim=dim_p,
                                                 input_size= input_size[0],
                                                 prompt_len = 5,
                                                 lin_dim= dim_p)
        
            self.noise_level1 = TransformerBlock(dim=int(dim_p*2**1) ,
                                             num_heads=1, 
                                             ffn_expansion_factor=2.66, 
                                             bias=False, LayerNorm_type='WithBias')
        
            self.reduce_noise_level1 = nn.Conv2d(int(dim_p*2),int(dim_p*1),kernel_size=1,bias=False)

        self.layer_lka_2 = LKABlock(dim=out_dim)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.prompt1(refined_feature)
                cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                cat_input_prompt = self.noise_level1(cat_input_prompt)
                refined_feature = self.reduce_noise_level1(cat_input_prompt)

            tran_layer_2 = self.layer_lka_2(refined_feature)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayerDAEFormer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        reduction_ratio = reduction_ratio
        head_count = head_count
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim, value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)

            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)

            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.tran_layer1 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)
        self.tran_layer2 = DualTransformerBlock(in_dim=dims,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                head_count=head_count)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()

            b2, h2, w2, c2 = x2.shape  # B C H W --> B H W C
            x2 = x2.view(b2, -1, c2)  # B C H W --> B (HW) C

            x1_expand = self.x1_linear(x1)  # B N C
            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B (HW) C --> B C H W

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2)  # B N C --> B C H W

            attn_gate = self.ag_attn(x=x2_new, g=x1_expand)  # B C H W

            cat_linear_x = x1_expand + attn_gate  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            # cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W
            cat_linear_x = cat_linear_x.view(b2, -1, c2)  # B H W C --> B (HW) C

            tran_layer_1 = self.tran_layer1(cat_linear_x, h2, w2)  # B N C
            # print(tran_layer_1.shape)
            tran_layer_2 = self.tran_layer2(tran_layer_1, h2, w2)  # B N C
            # print(tran_layer_2.shape)

            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out

class MyDecoderLayerLKAFreq(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefiner(dim = dim_p,
                                      h = input_size[0],
                                      w = input_size[0])
        
            self.fused = FrequencyPromptFusion(dim = dim_p,
                                               dim_bak= dim_p,
                                               win_size= 8,
                                               num_heads= 2)
        
            self.mlp = nn.Conv2d(int(dim_p),int(dim_p),kernel_size=1,bias=False)

        self.layer_lka_2 = LKABlock(dim=out_dim)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                # cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                fused_map = self.fused(refined_feature, prompt_layer_1)
                refined_feature = self.mlp(fused_map).contiguous() 
#                 print(refined_feature.shape)
                      
            tran_layer_2 = self.layer_lka_2(refined_feature)

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayerAdapLKAFreq(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefiner(dim = dim_p,
                                      h = input_size[0],
                                      w = input_size[0])
        
            self.fused = FrequencyPromptFusion(dim = dim_p,
                                               dim_bak= dim_p,
                                               win_size= 8,
                                               num_heads= 2)
        
            self.mlp = nn.Conv2d(int(dim_p),int(dim_p),kernel_size=1,bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_features = out_dim)
        self.layer_lka_2 = AdaptiveAttentionModule(in_channels=out_dim)
        self.bn2 = nn.BatchNorm2d(num_features = out_dim)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                # cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                fused_map = self.fused(refined_feature, prompt_layer_1)
                refined_feature = self.mlp(fused_map).contiguous() 

            tran_layer_2 = self.bn2(self.layer_lka_2(self.bn1(refined_feature)))

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
    
class MyDecoderLayerLKAFreqEnhanced(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefinerEnhanced(dim = dim_p,
                                              h = input_size[0],
                                              w = input_size[0])
        
            self.fused = FrequencyPromptFusionEnhanced(dim = dim_p,
                                                       dim_bak= dim_p,
                                                       win_size= 8,
                                                       num_heads= 2)
        
            self.mlp = nn.Conv2d(int(dim_p),int(dim_p),kernel_size=3,bias=False, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features = out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)
        self.bn2 = nn.BatchNorm2d(num_features = out_dim)
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                # cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                fused_map = self.fused(refined_feature, prompt_layer_1)
                refined_feature = self.mlp(fused_map).contiguous()

            tran_layer_2 = self.bn2(self.layer_lka_2(self.bn1(refined_feature)))

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out

class MyDecoderLayerLKAFreqEnhancedAdaptive(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefinerEnhanced(dim = dim_p,
                                              h = input_size[0],
                                              w = input_size[0])
        
            self.fused = FrequencyPromptFusionEnhanced(dim = dim_p,
                                                       dim_bak= dim_p,
                                                       win_size= 8,
                                                       num_heads= 2)
        
            self.mlp = nn.Conv2d(int(dim_p),int(dim_p),kernel_size=3,bias=False, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features = out_dim)
        self.layer_lka_2 = AdaptiveAttentionModule(in_channels=out_dim)
        self.bn2 = nn.BatchNorm2d(num_features = out_dim)
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                # cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                fused_map = self.fused(refined_feature, prompt_layer_1)
                refined_feature = self.mlp(fused_map).contiguous()

            tran_layer_2 = self.bn2(self.layer_lka_2(self.bn1(refined_feature)))

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out


class MyDecoderLayerLKAFreqEnhancedCat(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefinerEnhanced(dim = dim_p,
                                              h = input_size[0],
                                              w = input_size[0])
        
            # self.fused = FrequencyPromptFusionEnhanced(dim = dim_p,
            #                                            dim_bak= dim,
            #                                            win_size= 8,
            #                                            num_heads= 2)
            self.noise_level1 = TransformerBlock(dim=int(dim_p*2**1) ,
                                             num_heads=1, 
                                             ffn_expansion_factor=2.66, 
                                             bias=False, LayerNorm_type='WithBias')
            
            self.mlp = nn.Conv2d(int(dim_p),int(dim_p),kernel_size=1,bias=False, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features = out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)
        self.bn2 = nn.BatchNorm2d(num_features = out_dim)
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                # fused_map = self.fused(refined_feature, prompt_layer_1)
                fused_map = self.noise_level1(cat_input_prompt)
                refined_feature = self.mlp(fused_map).contiguous()

            tran_layer_2 = self.bn2(self.layer_lka_2(self.bn1(refined_feature)))

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
    
class MyDecoderLayerLKAFreqEnhancedCatAdapt(nn.Module):
    def __init__(
            self, input_size: tuple, in_out_chan: tuple, n_class=9,
            norm_layer=nn.LayerNorm, is_last=False, decoder_prompt = False
    ):
        super().__init__()
        out_dim = in_out_chan[0]
        x1_dim = in_out_chan[1]
        self.decoder_prompt = decoder_prompt
        # prompt_ratio = prompt_ratio
        
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            #self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)

            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        
        self.layer_lka_1 = LKABlock(dim=out_dim)
        ## Prompt Module must be located here.

        #dim_p = int(out_dim * 0.75)
        if decoder_prompt: 
            dim_p = out_dim
            self.refiner = FrqRefinerEnhanced(dim = dim_p,
                                              h = input_size[0],
                                              w = input_size[0])
        
            # self.fused = FrequencyPromptFusionEnhanced(dim = dim_p,
            #                                            dim_bak= dim,
            #                                            win_size= 8,
            #                                            num_heads= 2)
            self.noise_level1 = TransformerBlock(dim=int(dim_p*2**1) ,
                                             num_heads=1, 
                                             ffn_expansion_factor=2.66, 
                                             bias=False, LayerNorm_type='WithBias')
            
            self.mlp = nn.Conv2d(int(dim_p**2),int(dim_p),kernel_size=1,bias=False, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features = out_dim)
        self.layer_lka_2 = AdaptiveAttentionModule(in_channels=out_dim)
        self.bn2 = nn.BatchNorm2d(num_features = out_dim)
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            x2 = x2.contiguous()
            # b, c, h, w = x1.shape
            b2, h2, w2, c2 = x2.shape  # e.g: 1 28 28 320, 1 56 56 128
            x2 = x2.view(b2, -1, c2)  # e.g: 1 784 320, 1 3136 128

            x1_expand = self.x1_linear(x1)  # e.g: 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            x2_new = x2.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W
            

            x1_expand = x1_expand.view(x2.size(0), x2.size(2), x2.size(1) // w2, x2.size(1) // h2) # B, C, H, W

            # print(f'the x1_expand shape is: {x1_expand.shape}\n\t the x2_new shape is: {x2_new.shape}')

            cat_linear_x = x1_expand + x2_new  # B C H W
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 1)  # B H W C
            cat_linear_x = self.ag_attn_norm(cat_linear_x)  # B H W C

            cat_linear_x = cat_linear_x.permute(0, 3, 1, 2).contiguous()  # B C H W

            refined_feature = self.layer_lka_1(cat_linear_x)
            
            
            if self.decoder_prompt:
                prompt_layer_1 = self.refiner(refined_feature)
                cat_input_prompt = torch.cat([refined_feature, prompt_layer_1], dim= 1)
                # fused_map = self.fused(refined_feature, prompt_layer_1)
                fused_map = self.noise_level1(cat_input_prompt)
                refined_feature = self.mlp(fused_map).contiguous()

            tran_layer_2 = self.bn2(self.layer_lka_2(self.bn1(refined_feature)))

            tran_layer_2 = tran_layer_2.view(tran_layer_2.size(0), tran_layer_2.size(3) * tran_layer_2.size(2),
                                             tran_layer_2.size(1))
            if self.last_layer:
                
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b2, 4 * h2, 4 * w2, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
##########################################
#
#                MSA^2Net
#
##########################################
# from .merit_lib.networks import MaxViT4Out_Small#, MaxViT4Out_Small3D
# from networks.merit_lib.networks import MaxViT4Out_Small
# from .merit_lib.decoders import CASCADE_Add, CASCADE_Cat

class Msa2Net_V1(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKA(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKA(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKA(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKA(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0



class Msa2Net_V2(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAPrompt(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAPrompt(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAPrompt(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKAPrompt(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0


class Msa2Net_V3(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreq(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    
class Msa2Net_V4(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreq(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerAdapLKAFreq(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    
class Msa2Net_V5(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreq(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreq(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerAdapLKAFreq(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=False,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    
class Msa2Net_V6(nn.Module):
    """
    MSA^2Net V6 without Adaptive Attention Module + MyDecoderLayerLKAFreqEnhanced
    """
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    
class Msa2Net_V7(nn.Module):
    """
    MSA^2Net V7 with Adaptive Attention Module + MyDecoderLayerLKAFreqEnhanced
    """
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreqEnhanced(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKAFreqEnhancedAdaptive(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    

class Msa2Net_V8(nn.Module):
    """
    MSA^2Net V8 with Adaptive Attention Module + MyDecoderLayerLKAFreqEnhancedCat 
    """
    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder
        self.backbone = MaxViT4Out_Small(n_class=num_classes, img_size=224)

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]


        self.decoder_3 = MyDecoderLayerLKAFreqEnhancedCat(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            decoder_prompt=False,
            n_class=num_classes)

        self.decoder_2 = MyDecoderLayerLKAFreqEnhancedCat(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_1 = MyDecoderLayerLKAFreqEnhancedCat(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            decoder_prompt=False,
            n_class=num_classes)
        
        self.decoder_0 = MyDecoderLayerLKAFreqEnhancedCatAdapt(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=num_classes,
            decoder_prompt=True,
            is_last=True)

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc_3, output_enc_2, output_enc_1, output_enc_0 = self.backbone(x)

        b, c, _, _ = output_enc_3.shape
        # print(output_enc_3.shape)
        # ---------------Decoder-------------------------
        tmp_3 = self.decoder_3(output_enc_3.permute(0, 2, 3, 1).view(b, -1, c))
        tmp_2 = self.decoder_2(tmp_3, output_enc_2.permute(0, 2, 3, 1))
        tmp_1 = self.decoder_1(tmp_2, output_enc_1.permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc_0.permute(0, 2, 3, 1))

        return tmp_0
    
    
if __name__ == "__main__":
    input0 = torch.rand((1, 3, 224, 224)).cuda(0)
    input = torch.randn((1, 768, 7, 7)).cuda(0)
    input2 = torch.randn((1, 384, 14, 14)).cuda(0)
    input3 = torch.randn((1, 192, 28, 28)).cuda(0)
    b, c, _, _ = input.shape
    dec1 = MyDecoderLayerDAEFormer(input_size=(7, 7), in_out_chan=([768, 768, 768, 768, 768]), head_count=32,
                                     token_mlp_mode='mix_skip', reduction_ratio=16).cuda(0)
    dec2 = MyDecoderLayerDAEFormer(input_size=(14, 14), in_out_chan=([384, 384, 384, 384, 384]), head_count=16,
                                     token_mlp_mode='mix_skip', reduction_ratio=8).cuda(0)
    dec3 = MyDecoderLayerLKA(input_size=(28, 28), in_out_chan=([192, 192, 192, 192, 192]), head_count=1,
                                     token_mlp_mode='mix_skip', reduction_ratio=6).cuda(0)
    output = dec1(input.permute(0, 2, 3, 1).view(b, -1, c))
    output2 = dec2(output, input2.permute(0, 2, 3, 1))
    output3 = dec3(output2, input3.permute(0, 2, 3, 1))

    # net = MaxViT_deformableLKAFormer().cuda(0)

    # output0 = net(input0)
    print("Out shape: " + str(output.shape) + 'Out2 shape:' + str(output2.shape) + "Out shape3: " + str(output3.shape))
    # print("Out shape: " + str(output0.shape))
