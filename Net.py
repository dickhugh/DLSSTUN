import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
from timm.models.layers import DropPath, to_2tuple

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):   
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  
        y = self.conv_du(y)
        return x * y
    
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False, act = nn.PReLU()):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class ProPorcess(nn.Module):
    def __init__(self, dim=32, expand=2, cs=27):
        super(ProPorcess, self).__init__()
        self.dim = dim
        self.stage = 2
        
        # Input projection
        self.in_proj = nn.Conv2d(cs, dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                nn.Conv2d(dim_stage * expand, dim_stage*expand, 1, 1, 0, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = ASPP(dim_stage, [3,6], dim_stage)

        # Decoder:
        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
            ]))
            dim_stage //= 2

        self.out_conv2 = nn.Conv2d(self.dim, cs, 3, 1, 1, bias=False)
        self.fusion1 = nn.Sequential(
            CAB(128),
            nn.Conv2d(128, 16, 3, 1, 1, bias=False)
        )
        self.fusion2 = nn.Sequential(
            CAB(64),
            nn.Conv2d(64, 12, 3, 1, 1, bias=False) 
        )
        self.fusion3 = nn.Sequential(
            CAB(32),
            nn.Conv2d(32, 4, 3, 1, 1, bias=False) 
        )
        self.fusion4 = CAB(32) 
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.hiddendown = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)

    def forward(self, x, phi):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(torch.mul(x, phi)))
        # Encoder
        fea_hi = []
        fea_encoder = []  # [c 2c ]
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
        # Bottleneck
        fea = self.bottleneck(fea)+fea
        fea_hi.append(F.interpolate(self.fusion1(fea), scale_factor=4))
        # Decoder
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage-1-i]
            if i == 0:
                fea_hi.append(F.interpolate(self.fusion2(fea), scale_factor=2))
            if i == 1:
                fea_hi.append(self.fusion3(fea))
        hidden = []
        hidden1 = self.fusion4(torch.cat(fea_hi, dim=1))
        hidden.append(hidden1)
        hidden2 = self.hiddendown(hidden1)
        hidden.append(hidden2)
        # Output projection
        out = self.out_conv2(fea)
        return out,hidden

def At(y, Phi):
    x = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = x * Phi
    return x

def A(x, Phi):
    [_, nC, _, _] = x.shape
    x = x * Phi
    y = torch.sum(x, 1) / nC * 2
    return y

class Phi(nn.Module):
    def __init__(self,dim,embedding_dim) -> None:
        super(Phi,self).__init__()
        self.AdaptivePhi = nn.Sequential(
            nn.Conv2d(dim + 1,embedding_dim,1,1),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False, groups=embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim,embedding_dim, 3, 1, 1, bias=False, groups=embedding_dim),
            nn.Conv2d(embedding_dim,dim,1,1)
        )
    
    def forward(self,x):
        phi = self.AdaptivePhi(x)
        return phi
 
class GD(nn.Module):
    def __init__(self,dim) -> None:
        super(GD,self).__init__()
        self.Phi = Phi(dim, 32)
        self.r = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x, Phi, y):
        phi_res = self.Phi(torch.cat([y.unsqueeze(1), Phi], axis=1))
        phi = phi_res + Phi                             
        AX = A(x, phi)
        res = AX - y
        ATres = At(res, phi)
        v = x - self.r * ATres

        return v

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) 
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

#FeedForward Network(FFN) 和Soft Threshold Generator(STG)
class FFNandSTG(nn.Module):   
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FFNandSTG, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        x = self.project_in(x.permute(0, 3, 1, 2))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x.permute(0, 2, 3, 1)
    
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02) # trunc_normal_函数：截断正太分布
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        output:(num_windows*B, N, C)
                """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SwinTransformer(nn.Module):
    r""" Swin Transformer.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift_size > 0:
            # calculate attention mask for SW
            H, W = self.input_resolution, self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            #print(mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W/SW
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x
    
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, stage=1):
        super(SwinTransformerBlock, self).__init__()
        self.w_msa = SwinTransformer(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=0)
        self.sw_msa = SwinTransformer(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=4)
        self.mlp = PreNorm(dim, FFNandSTG(dim=dim))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.w_msa(x) + x
        x = self.mlp(x) + x
        x = self.sw_msa(x) + x
        x = self.mlp(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class SoftThresholdBlock(nn.Module):
    def __init__(self, dim):
        super(SoftThresholdBlock, self).__init__()

        self.stg = FFNandSTG(dim)

        self.proj_forward = nn.Sequential(            
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False)
        )

        self.proj_backward = nn.Sequential(            
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim, 3, 1, 1, bias=False)
        )

        self.proj_back = nn.Conv2d(dim*2,dim,1,1)

    
    def forward(self, x):
        soft_thr = self.stg(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_fd = self.proj_forward(x)

        x_thr = torch.mul(torch.sign(x_fd), F.relu(torch.abs(x_fd) - soft_thr))

        x_bd = self.proj_backward(x_thr)

        x_res = self.proj_backward(x_fd) - x

        x_out = self.proj_back(torch.cat([x_bd, x], dim = 1)) + x
    
        return x_out, x_res

class InformationTransfer(nn.Module):
    def __init__(self, dim, hidden_dim=32, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)

    def forward(self, hidden):
        out = self.conv(hidden)
        out = self.act(out)
        out = self.dwconv(out)
        
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim, out_dim, dim, scale=2) -> None:
        super(Denoiser,self).__init__()
        self.dim = dim
        self.scale = scale
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scale):
            self.encoder_layers.append(nn.ModuleList([
                SwinTransformerBlock(dim=dim_scale, stage=i+1),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2
        
        # Bottleneck
        self.bottleneck = SoftThresholdBlock(dim=dim_scale)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scale):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale * 3 // 2, dim_scale // 2, 1, 1, bias=False),
                SwinTransformerBlock(dim=dim_scale // 2, stage=self.scale-i),
            ]))
            dim_scale //= 2

        self.stage_transfer = nn.ModuleList([
                        InformationTransfer(dim = self.dim, hidden_dim = self.dim),
                        InformationTransfer(dim = self.dim * 2, hidden_dim = self.dim * 2)
                    ])

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, hidden, stage = 0):

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for ST, DownSample in self.encoder_layers:
            fea = ST(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # Bottleneck
        fea, sym_k = self.bottleneck(fea)

        # Decoder
        for i, (UpSample, Fution, ST) in enumerate(self.decoder_layers):
            fea = UpSample(fea)
            transfer = self.stage_transfer[self.scale-1-i](hidden[self.scale-1-i])
            fea = Fution(torch.cat([fea, fea_encoder[self.scale-1-i], transfer], dim=1))
            fea = ST(fea)

        # Output projection
        out = self.mapping(fea) + x
        return out, sym_k, fea_encoder

class Phases(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super(Phases,self).__init__()
        self.GD = GD(dim)
        self.Denoiser = Denoiser(in_dim=dim, out_dim=dim, dim=hidden_dim) 
    
    def forward(self, x, y, Phi, hidden, stage): 
        v = self.GD(x, Phi, y)
        xk, sym_k, hidden= self.Denoiser(v, hidden, stage)
        return xk, sym_k, hidden
       
class X(nn.Module):
    def __init__(self, dim, stage) -> None:
        super(X,self).__init__()
        self.stage = stage
        self.init = ProPorcess(32)
        self.Phases = nn.ModuleList([])
        for _ in range(stage):
            self.Phases.append(
                Phases(dim, 32)
            )
    def forward(self, y, phi):
        meas_3D = torch.unsqueeze(y, 1).repeat(1, phi.shape[1], 1, 1)
        x, hidden = self.init(meas_3D, phi)
        layers_sym = []
        for stage, phase in enumerate(self.Phases):
            x, sym_k, hidden = phase(x, y, phi, hidden, stage)
            layers_sym.append(sym_k)
        return x, layers_sym
