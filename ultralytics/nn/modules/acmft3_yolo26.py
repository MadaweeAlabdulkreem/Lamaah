
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMLP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        h = max(channels // 4, 1)
        self.net = nn.Sequential(
            nn.Linear(channels, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, channels),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class LearnableUpsample2x(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        return self.deconv(x)


class CSFALevel(nn.Module):
    def __init__(self, channels: int, target_idx: int = 0):
        super().__init__()
        C = int(channels)

        # scale weights (λ)
        init = torch.zeros(3)
        init[target_idx] = 1.0
        self.lambdas = nn.Parameter(init)

        # channel attention
        self.mlp = ChannelMLP(C)

        # dw-conv لكل مستوى قبل الـ pooling
        self.dw = nn.ModuleList([
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
            for _ in range(3)
        ])

        # shared weights (مثل CSFA)
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, feats):  # feats = aligned features
        B, C, _, _ = feats[0].shape

        # -----------------
        # Scale attention
        # -----------------
        s = F.softmax(self.lambdas, dim=0)  # [3]

        # -----------------
        # Channel attention
        # -----------------
        ch_attn = []
        for feat, dw in zip(feats, self.dw):
            f =  dw(feat)  # ← Depthwise Conv قبل pooling
            pool = F.adaptive_max_pool2d(f, 1) + F.adaptive_avg_pool2d(f, 1)
            ch_attn.append(self.mlp(pool))  # [B, C]

        cw = F.softmax(torch.stack(ch_attn, dim=1), dim=1)  # [B, 3, C]

        # Fusion
        # -----------------
        out = sum(
            (self.w1 * cw[:, i, :].view(B, C, 1, 1)
             + self.w2 * s[i])
            * feats[i]
            for i in range(3)
        )

        return out


class _CSFABase(nn.Module):
    def __init__(self, c1, target_idx: int = 0, *args):
        super().__init__()

        chs = list(c1) if isinstance(c1, (list, tuple)) else [int(c1), int(args[0]), int(args[1])]
        C = chs[target_idx]
        self.c2 = C

        # projection
        self.proj = nn.ModuleList([
            nn.Conv2d(c, C, kernel_size=1, bias=False) for c in chs
        ])


        align_configs = {
            0: [nn.Identity(),
                LearnableUpsample2x(C),
                nn.Sequential(LearnableUpsample2x(C), LearnableUpsample2x(C))],
            1: [nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.Identity(),
                LearnableUpsample2x(C)],
            2: [nn.Sequential(nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                               nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False)),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.Identity()],
        }

        self.align = nn.ModuleList(align_configs[target_idx])

        # CSFA بدل AMSF
        self.fusion = CSFALevel(C, target_idx)

    def forward(self, x):
        projected = [p(f) for p, f in zip(self.proj, x)]
        aligned   = [a(f) for a, f in zip(self.align, projected)]
        return self.fusion(aligned)


# -----------------------------
# Heads
# -----------------------------
class CSFA_TD_P2(_CSFABase):
    def __init__(self, c1, *args): super().__init__(c1, 0, *args)

class CSFA_TD_P3(_CSFABase):
    def __init__(self, c1, *args): super().__init__(c1, 1, *args)

class CSFA_TD_P4(_CSFABase):
    def __init__(self, c1, *args): super().__init__(c1, 2, *args)
