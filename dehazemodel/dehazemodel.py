import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_
import math


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size**2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - \
        coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(
        1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(
        relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log


class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(
            (input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(
                std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8, conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (
                                 i % 2 == 0) else window_size // 2,
                             use_attn=True,
                             conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim //
                          self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3,
                          padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3,
                          padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5,
                                  padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h %
                     self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w %
                     self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            # nW*B, window_size**2, C
            qkv = window_partition(shifted_QKV, self.window_size)

            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(
                attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(
                self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                # no attention and use conv, no projection
                out = self.conv(X)
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(
            dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim,
                       hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x


class Global(nn.Module):
    def __init__(self, in_chans=3, out_chans=12, window_size=8,
                 embed_dims=[8, 16, 24, 16, 8],
                 mlp_ratios=4.,
                 depths=[0, 0, 2, 0, 0],
                 num_heads=2,
                 #  conv_type=['None', 'None', 'None', 'None', 'None'],
                 conv_type='DWConv',
                 norm_layer=RLN
                 ):
        super(Global, self).__init__()

        # setting
        self.patch_size = 2
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.patch_merge1 = PatchEmbed(
            patch_size=self.patch_size, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.patch_merge2 = PatchEmbed(
            patch_size=self.patch_size, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                num_heads=num_heads, mlp_ratio=mlp_ratios,
                                norm_layer=norm_layer, window_size=window_size,
                                conv_type=conv_type)

        self.patch_split1 = PatchUnEmbed(
            patch_size=self.patch_size, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=self.patch_size, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)  # conv2d
        skip1 = x

        x = self.patch_merge1(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        return feat


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0,
                      bias=True, groups=channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0,
                      bias=True, groups=channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ConvLayer(nn.Module):
    def __init__(self, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect'))
        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True))
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out


class PWLocalBranch2(nn.Module):
    def __init__(self):
        super(PWLocalBranch2, self).__init__()
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = ConvLayer(dim=3, kernel_size=3, gate_act=nn.Sigmoid)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1)
        self.conv3 = ConvLayer(dim=6, kernel_size=3, gate_act=nn.Sigmoid)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=27, out_channels=12, kernel_size=1)
        self.conv6 = ConvLayer(dim=12, kernel_size=3, gate_act=nn.Sigmoid)
        self.conv7 = nn.Conv2d(in_channels=27, out_channels=12, kernel_size=1)
        self.conv8 = ConvLayer(dim=12, kernel_size=3, gate_act=nn.Sigmoid)

    def forward(self, x, x1):
        feat1 = self.lrelu(self.conv2(self.conv1(x)))
        feat2 = self.lrelu(self.conv4(self.conv3(feat1)))
        feat3 = torch.cat((x, x1, feat2), 1)  # 3+12+12=27
        feat4 = self.lrelu(self.conv5(feat3))
        feat5 = self.conv6(feat4)
        feat6 = torch.cat((x, feat2, feat5), 1)  # 3+12+12=27
        feat7 = self.lrelu(self.conv7(feat6))
        feat8 = self.conv8(feat7)

        return feat8


class AODPWnet2(nn.Module):
    def __init__(self):
        super(AODPWnet2, self).__init__()
        self.conv00 = nn.Conv2d(
            in_channels=15, out_channels=6, kernel_size=3, padding=1, padding_mode='reflect')

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1)
        self.PA1 = PALayer(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3,
                               kernel_size=3, padding=1, padding_mode='reflect')
        self.PA2 = PALayer(9)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=3,
                               kernel_size=5, padding=2, padding_mode='reflect')
        self.PA3 = PALayer(12)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=3,
                               kernel_size=7, padding=3, padding_mode='reflect')
        self.PA4 = PALayer(15)
        self.conv5 = nn.Conv2d(in_channels=15, out_channels=3,
                               kernel_size=3, padding=1, padding_mode='reflect')
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv00(x)
        x = self.lrelu(x)

        x1 = self.lrelu(self.conv1(x))
        x1 = self.PA1(x1)
        x2 = self.lrelu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        cat1 = self.PA2(cat1)
        x3 = self.lrelu(self.conv3(cat1))
        cat2 = torch.cat((x1, x2, x3), 1)
        cat2 = self.PA3(cat2)
        x4 = self.lrelu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        cat3 = self.PA4(cat3)
        k = self.lrelu(self.conv5(cat3))

        return k


class dehaze(nn.Module):
    def __init__(self):
        super(dehaze, self).__init__()
        self.swin = Global()
        self.local = PWLocalBranch2()
        self.aod2 = AODPWnet2()
        self.fusion1 = SKFusion(12)
        self.b = 1.0

    def forward(self, x):
        x1 = self.swin(x)
        x2 = self.local(x, x1)
        x3 = self.fusion1([x1, x2])
        cat1 = torch.cat((x3, x), 1)

        k = F.leaky_relu(self.aod2(cat1))
        output = k * x - k + self.b
        return output


if __name__ == '__main__':
    model = dehaze().to('cuda:0')
    num_params = sum(p.numel() for p in model.parameters())
    print('num params', num_params)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('grad num params ', num_params)

    img = torch.ones((1, 3, 640, 640)).to('cuda:0')
    for i in range(500):
        out = model(img)

    t1 = time.time()
    for i in range(500):
        out = model(img)
    t2 = time.time()
    print('time:', t2-t1)
