import math
import numbers
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from einops import rearrange
 

class PointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.fc(x)

class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden_dim = int(dim//2 * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim//2, hidden_dim, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim//2, 1, 1, 0),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        return rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        # return x / torch.sqrt(sigma+1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Shuffle Mixing layer
class SMLayer(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.mlp1 = SplitPointMlp(dim, mlp_ratio)
        self.mlp2 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x):
        x = self.mlp1(self.norm1(x)) + x
        x = self.spatial(x)
        x = self.mlp2(self.norm2(x)) + x
        return x


# Feature Mixing Block
class FMBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim + 16, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim + 16, dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.net(x) + x
        x = self.conv(x) + x
        return x

 
class ShuffleMixer(nn.Module):
    """
    Args:
        n_feats (int): Number of channels. Default: 64 (32 for the tiny model).
        kerenl_size (int): kernel size of Depthwise convolution. Default:7 (3 for the tiny model).
        n_blocks (int): Number of feature mixing blocks. Default: 5.
        mlp_ratio (int): The expanding factor of point-wise MLP. Default: 2.
        upscaling_factor: The upscaling factor. [2, 3, 4]
    """
    def __init__(self, n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, n_feats, 3, 1, 1, bias=False)

        self.blocks = nn.Sequential(
            *[FMBlock(n_feats, kernel_size, mlp_ratio) for _ in range(n_blocks)]
        )

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_feats, n_feats * 4 , 1, 1, 0),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale),
                nn.SiLU(inplace=True)
            )

        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        base = x
        x = self.to_feat(x)
        x = self.blocks(x)
        x = self.upsapling(x)
        x = self.tail(x)
        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + base

if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # 720p [1280 * 720]
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    x = torch.randn(1, 3, 256, 256)

    model = ShuffleMixer(n_feats=32, kernel_size=21, n_blocks=5, mlp_ratio=2, upscaling_factor=4)
    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(f'output: {output.shape}')