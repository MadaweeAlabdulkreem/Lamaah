
#    (1)  F  = δ( AP( Conv(X) ) )
#    (2)  Q, K, V = SparseQ(X), SparseK(X), SparseV(X)
#    (3)  F′ = Conv( Unsparse( CA-MHSA(Q, K, V) ) )
#    (4)  X′ = Reshape( F ⊗ F′ )

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelMHSA(nn.Module):

    def __init__(self, c: int, num_heads: int = 8):
        super().__init__()
        assert c % num_heads == 0, (f"channels {c} must be divisible by num_heads {num_heads}")

        self.nh = num_heads
        self.hd = c // num_heads       # head dimension
        self.sc = self.hd ** -0.5      # 1/√(head_dim) scale

        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)

        self.proj = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H·W, C)
        tokens = x.flatten(2).transpose(1, 2)

        # Eq (2): Q, K, V - each (B, H·W, C)
        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)    

        # Multi-head split: (B, H·W, C) → (B, nh, H·W, hd)
        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, -1, self.nh, self.hd).transpose(1, 2)

        q, k, v = _heads(q), _heads(k), _heads(v)

        # Scaled dot-product attention (CA-MHSA)
        attn = (q @ k.transpose(-2, -1)) * self.sc     
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                

        # Merge heads: (B, nh, H·W, hd) → (B, H·W, C)
        out = out.transpose(1, 2).reshape(B, -1, C)

        out = self.proj(out)                             # (B, H·W, C)

        # (B, H·W, C) → (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
class NSSA(nn.Module):
    """
    Non-Semantic Sparse Attention  

     ig. 2 of the paper shows an interleaved
    (checkerboard / stride-S) partition, NOT contiguous quadrant blocks:

      S = 2, 4×4 example         contiguous (WRONG)   interleaved (CORRECT)
      ─────────────────────       ──────────────────   ────────────────────
      Input pixel layout          0 0 1 1              0 1 0 1
                                  0 0 1 1              2 3 2 3
                                  2 2 3 3              0 1 0 1
                                  2 2 3 3              2 3 2 3

    Each block covers the full spatial extent with stride S, so attention
    within a block captures long-range non-semantic structure rather than
    only a local corner — this is the core reason the mechanism works.
    """

    def __init__(self, c: int, S: int = 2, num_heads: int = 8):
        super().__init__()
        assert c % num_heads == 0, (f"channels {c} must be divisible by num_heads {num_heads}")
        self.S = S

        # Eq (1) branch: Conv in  F = δ(AP(Conv(X)))
        self.b1 = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # Eq (2–3) branch: CA-MHSA on sparsified features
        self.ca = ChannelMHSA(c, num_heads)

        # Eq (3): Conv in  F′ = Conv(Unsparse(CA-MHSA(…)))
        self.b2 = nn.Conv2d(c, c, kernel_size=1, bias=False)

    # ── Sparsify 
    def _sp(self, x: torch.Tensor, S: int) -> torch.Tensor:
        B, C, H, W = x.shape
        return (
            x.reshape(B, C, H // S, S, W // S, S)   # H → (H/S rows) × (S strides)
             .permute(0, 3, 5, 1, 2, 4)             # → (B, S, S, C, H/S, W/S)
             .contiguous()
             .reshape(B * S * S, C, H // S, W // S)  # merge B and S² into batch
        )

    # ── Unsparse 
    def _unsp(self, x: torch.Tensor, B: int, H: int, W: int, S: int) -> torch.Tensor:
        C = x.shape[1]
        return (
            x.reshape(B, S, S, C, H // S, W // S)   # separate B from S²
             .permute(0, 3, 4, 1, 5, 2)             # → (B, C, H/S, S, W/S, S)
             .contiguous()
             .reshape(B, C, H, W)                   # merge strides back
        )

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        S = self.S

        if H % S != 0 or W % S != 0:
            S = 1

        # ── Eq (1): Branch 1 — channel-wise attention weights ─────────────
        F_w = torch.sigmoid(F.adaptive_avg_pool2d(self.b1(x), output_size=1))                                                  # (B, C, 1, 1)

        # ── Eqs (2–3): Branch 2 — sparse spatial attention features ───────
        #   Sparsify X into S² interleaved blocks   →  (B·S², C, H/S, W/S)
        #   Q, K, V from the sparsified input (Eq. 2)
        #   CA-MHSA within each block              →  (B·S², C, H/S, W/S)
        #   Unsparse                               →  (B, C, H, W)
        #   Conv                                   →  F′ (B, C, H, W)
        xs  = self._sp(x, S)                               # (B·S², C, H/S, W/S)
        F_p = self.b2( self._unsp(self.ca(xs), B, H, W, S))                                                  # (B, C, H, W)

        # ── Eq (4): X′ = Reshape( F ⊗ F′ ) ───────────────────────────────
        return F_w * F_p                                   # (B, C, H, W)
