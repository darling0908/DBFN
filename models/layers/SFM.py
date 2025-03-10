import torch
from torch import nn
import torch.nn.functional as F


# channels_first:(B,C,H,W)
# channels_last:(B,H,W,C)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)  # [batch_size, 1, height, width]
            var = (x - mean).pow(2).mean(1, keepdim=True)  # [batch_size, 1, height, width]
            x = (x - mean) / torch.sqrt(var + self.eps)  # 将x标准化为均值为0，方差为1的分布
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(MLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        # out = self.conv2(out)
        # out = self.gelu(out)
        # out = self.conv3(out)

        return out


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class SFM(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(SFM, self).__init__()
        # without Fi-1
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        # with Fi-1
        self.Updim = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)

        # CNN branch
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # CNN branch
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )

        # final part
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.residual = MLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.GELU()

    def forward(self, l, g):
        X_f = torch.cat([g, l], 1)
        X_f = self.norm2(X_f)
        X_f = self.W(X_f)
        X_f = self.relu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)  # eg:stage1:(b,1,56,56)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump  # eg:stage1:(b,96,56,56)

        # channel attetion for transformer branch
        g_jump = g
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)  # eg:stage1:(b,96,1,1)
        g = self.sigmoid(max_out + avg_out) * g_jump  # eg:stage1:(b,96,56,56)

        # print("1111111", g.shape, l.shape)
        f = torch.cat([g, l, X_f], 1)
        f = self.norm3(f)
        f = self.residual(f)
        f = self.drop_path(f)
        # print("1111111", fuse.shape)
        return f
