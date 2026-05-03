
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class dsifLevel(nn.Module):
    def __init__(self, in_channels, out_channels, target_idx):
        super().__init__()
        self.n = len(in_channels)
        self.target_idx = target_idx

        self.align = nn.ModuleList([
            Conv(int(c), out_channels, 1, 1) for c in in_channels
        ])

        fused_c = out_channels * self.n
        self.pw = nn.Conv2d(fused_c, fused_c, 1, 1)
        self.smooth = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 1) for _ in range(self.n)
        ])

        self.c2 = out_channels

    def forward(self, feats):
        target_hw = feats[self.target_idx].shape[-2:]
        resized = []

        for j, x in enumerate(feats):
            x = self.align[j](x)

            if j < self.target_idx:
                x = F.adaptive_avg_pool2d(x, target_hw)
            elif j > self.target_idx:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

            resized.append(x)

        F_cat = torch.cat(resized, dim=1)

        gate = torch.sigmoid(self.pw(F.adaptive_avg_pool2d(F_cat, 1)))
        F_hat = F_cat + F_cat * gate

        splits = torch.chunk(F_hat, self.n, dim=1)
        smoothed = [self.smooth[i](splits[i]) for i in range(self.n)]

        out = smoothed[0]
        for i in range(1, self.n):
            out = out * smoothed[i]

        return out


class BIMA_TD_P4(nn.Module):
    def __init__(self, c1, out_c):
        super().__init__()
        self.dsif = dsifLevel(c1, out_c, target_idx=2)
        self.c2 = out_c

    def forward(self, x):
        return self.dsif(x)


class BIMA_TD_P3(nn.Module):
    def __init__(self, c1, out_c):
        super().__init__()
        self.dsif = dsifLevel(c1, out_c, target_idx=1)
        self.c2 = out_c

    def forward(self, x):
        return self.dsif(x)


class BIMA_TD_P2(nn.Module):
    def __init__(self, c1, out_c):
        super().__init__()
        self.dsif = dsifLevel(c1, out_c, target_idx=0)
        self.c2 = out_c

    def forward(self, x):
        return self.dsif(x)


class BIMA_BU_P3(nn.Module):
    def __init__(self, c1, out_c):
        super().__init__()
        self.dsif = dsifLevel(c1, out_c, target_idx=1)
        self.c2 = out_c

    def forward(self, x):
        return self.dsif(x)


class BIMA_BU_P4(nn.Module):
    def __init__(self, c1, out_c):
        super().__init__()
        self.dsif = dsifLevel(c1, out_c, target_idx=2)
        self.c2 = out_c

    def forward(self, x):
        return self.dsif(x)
