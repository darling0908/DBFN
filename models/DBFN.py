import torch
from timm.layers import trunc_normal_
from torch import nn

from models.layers.EACB import EACB
from models.layers.SFM import SFM
from models.layers.TFEB import TFEB
from models.layers.stem import ConvBNReLU


class DBFN(nn.Module):
    def __init__(self, num_classes, down_path_dropout, stem_chs=[64, 32, 64], in_chans=3,
                 down_depths=[2, 2, 2, 2], up_depths=[2, 2, 8, 2], HFF_dp=0.1, embed_dim=[96, 192, 384, 768],
                 topks=[1, 4, 16, 49], qk_dims=[96, 192, 384, 768], head_dim=32, kv_per_wins=[-1, -1, -1, -1],
                 kv_downsample_kernels=[4, 2, 1, 1], kv_downsample_ratios=[4, 2, 1, 1], up_path_dropout=0.,
                 ):
        super().__init__()
        # Preprocessing Setting Start #
        self.stem = nn.Sequential(
            ConvBNReLU(in_chans, stem_chs[0], kernel_size=4, stride=4),
            # ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            # ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            # ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )

        # Down Branch Setting Start #
        dpr = [x.item() for x in torch.linspace(0, down_path_dropout, sum(down_depths))]  # stochastic depth decay rule
        in_channel = stem_chs[-1]
        self.down1 = nn.Sequential(
            EACB(in_channel, 96, path_dropout=dpr[0]),
            EACB(96, 96, path_dropout=dpr[1]),
        )
        self.down2 = nn.Sequential(
            EACB(96, 192, path_dropout=dpr[2], stride=2),
            EACB(192, 192, path_dropout=dpr[3])
        )
        self.down3 = nn.Sequential(
            EACB(192, 384, path_dropout=dpr[4], stride=2),
            EACB(384, 384, path_dropout=dpr[5]),
        )
        self.down4 = nn.Sequential(
            EACB(384, 768, path_dropout=dpr[6], stride=2),
            EACB(768, 768, path_dropout=dpr[7]),
        )
        # Down Branch Setting End #

        # Up Branch Setting Start #
        in_channel = stem_chs[-1]
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        up_stem = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(embed_dim[0])
        )
        self.downsample_layers.append(up_stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=1, stride=1),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            self.downsample_layers.append(downsample_layer)
        dpr = [x.item() for x in torch.linspace(0, up_path_dropout, sum(up_depths))]
        nheads = [dim // head_dim for dim in qk_dims]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[TFEB(dim=embed_dim[i], drop_path=dpr[cur + j], topk=topks[i], num_heads=nheads[i],
                      qk_dim=qk_dims[i], kv_per_win=kv_per_wins[i],
                      kv_downsample_ratio=kv_downsample_ratios[i],
                      kv_downsample_kernel=kv_downsample_kernels[i]) for j in range(up_depths[i])],
            )
            self.stages.append(stage)
            cur += up_depths[i]
        # Up Branch Setting End #
        # Hierarchical Feature Fusion Block Setting Start #
        self.fu4 = SFM(ch_1=768, ch_2=768, r_2=32, ch_int=768, ch_out=768, drop_rate=HFF_dp)
        # Hierarchical Feature Fusion Block Setting End #
        # Output Start #
        self.conv_norm = nn.BatchNorm2d(768, eps=1e-5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Linear(768, num_classes)
        # print('initialize_weights...')
        # self._initialize_weights()
        # Output End #

    # def merge_bn(self):
    #     self.eval()
    #     for idx, module in self.named_modules():
    #         if isinstance(module, EACB) or isinstance(module, TFEB):
    #             module.merge_bn()

    def forward(self, x):
        x_stem = self.stem(x)
        #  Up Branch Input #
        x_up_0 = self.downsample_layers[0](x_stem)
        x_up_0 = self.stages[0](x_up_0)
        x_up_1 = self.downsample_layers[1](x_up_0)
        x_up_1 = self.stages[1](x_up_1)
        x_up_2 = self.downsample_layers[2](x_up_1)
        x_up_2 = self.stages[2](x_up_2)
        x_up_3 = self.downsample_layers[3](x_up_2)
        x_up_3 = self.stages[3](x_up_3)

        # Down Branch Input #
        x_down_0 = self.down1(x_stem)
        x_down_1 = self.down2(x_down_0)
        x_down_2 = self.down3(x_down_1)
        x_down_3 = self.down4(x_down_2)

        #  Feature Fusion  #
        x_f_4 = self.fu4(x_down_3, x_up_3)

        x_fin = self.conv_norm(x_f_4)
        x_fin = self.avgpool(x_fin)
        x_fin = torch.flatten(x_fin, 1)
        x_fin = self.conv_head(x_fin)

        return x_fin


def mainNet(num_classes: int):
    model = DBFN(num_classes=num_classes, down_path_dropout=0.1)
    return model
