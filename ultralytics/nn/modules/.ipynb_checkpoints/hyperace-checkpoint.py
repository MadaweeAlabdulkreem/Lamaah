import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Conv & DSConv ---
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, (k - 1) // 2 * d, dilation=d, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU() if act else nn.Identity(),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

# --- 2. DSC3k ---
class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv(c1, c_, k1, 1)
        self.cv2 = DSConv(c_, c2, k2, 1, d2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x * 0.2 + out if self.add else out

class DSC3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, k1=3, k2=5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(DSBottleneck(c_, c_, shortcut, 1.0, k1, k2) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# --- 3. AdaHyperedgeGen (محسّن) ---
class AdaHyperedgeGen(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        prototypes = self.prototype_base.unsqueeze(0).expand(B, -1, -1)
        X_proj = self.pre_head_proj(X)

        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges)

        # Dual attention: hyperedge + node
        att_hyperedge = F.softmax(self.dropout(logits), dim=1)
        att_node = F.softmax(self.dropout(logits), dim=2)
        att = (att_hyperedge + att_node) / 2
        return att.mean(dim=1)

# --- 4. AdaHGComputation (محسّن) ---
class AdaHGConv(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout)
        self.edge_proj = nn.Linear(embed_dim, embed_dim)
        self.node_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        A = self.edge_generator(X)
        deg = A.sum(dim=1, keepdim=True) + 1e-6
        He = torch.bmm(A.transpose(1, 2), X) / deg.transpose(1, 2)
        He = F.gelu(self.edge_proj(He))
        X_new = torch.bmm(A, He)
        X_new = F.gelu(self.node_proj(X_new))
        # Res Scaling + LayerNorm
        return X + 0.2 * F.layer_norm(X_new, X_new.shape[1:])

class AdaHGComputation(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads, dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens)
        return tokens.transpose(1, 2).view(B, C, H, W)

# --- 5. Channel Attention (SE) ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# --- 6. C3AH & HyperACE (محسّن) ---
class C3AH(nn.Module):
    def __init__(self, c1, c2, e=0.5, num_hyperedges=8):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = AdaHGComputation(c_, num_hyperedges)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class HyperACE(nn.Module):
    def __init__(self, c_list, c2, num_hyperedges=16, e=0.5):
        super().__init__()
        self.c_mid = c2 // 4 # تقسيم القنوات ليتناسب مع الـ Concat النهائي (4 فروع)

        # طبقات توحيد القنوات (1x1 Conv)
        self.cv_p3 = Conv(c_list[0], self.c_mid, 1)
        self.cv_p4 = Conv(c_list[1], self.c_mid, 1)
        self.cv_p5 = Conv(c_list[2], self.c_mid, 1)

        # ممر الحفاظ على الأجسام الصغيرة
        self.p3_preserve = Conv(self.c_mid, self.c_mid, 3, g=self.c_mid)

        self.cv_reduce = Conv(self.c_mid * 3, self.c_mid, 1)

        # الفروع
        self.global_branch = C3AH(self.c_mid, self.c_mid, e=e, num_hyperedges=num_hyperedges)
        self.local_branch = DSC3k(self.c_mid, self.c_mid, n=2)

        self.cv_fusion = nn.Sequential(
            Conv(self.c_mid * 4, c2, 1),
            SELayer(c2)
        )

    def forward(self, x):
        b3, b4, b5 = x
        target_size = b4.shape[-2:]

        p3 = F.interpolate(self.cv_p3(b3), size=target_size, mode='bilinear')
        p4 = self.cv_p4(b4)
        p5 = F.interpolate(self.cv_p5(b5), size=target_size, mode='bilinear')

    
        p3 = p3 * 1.2 

        combined = torch.cat([p3, p4, p5], dim=1)
  
        reduced_x = self.cv_reduce(combined)

        g_feat = self.global_branch(reduced_x)
        l_feat = self.local_branch(reduced_x)

        final_cat = torch.cat([g_feat, l_feat, reduced_x, p3], dim=1)
        return self.cv_fusion(final_cat)