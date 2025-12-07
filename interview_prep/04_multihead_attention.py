from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Minimal multi-head self-attention module (no bias, no dropout, no KV cache).

    Shape convention:
        x: (batch, seq_len, d_model)

    Internal:
        We project x into q, k, v of shape (batch, n_heads, seq_len, d_head)
        where d_head = d_model // n_heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections for queries, keys, and values.
        # Each maps from d_model to d_model; we later split into heads.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Final output projection after concatenating heads.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional mask broadcastable to (batch, n_heads, seq_len, seq_len)

        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape
        print(f"\n{batch=}\n{seq_len=}\n{self.d_model=}\n{self.d_head=}\n{self.n_heads=}\n")

        # TODO:
        # 1. Project x into q, k, v
        # 2. Reshape / transpose to (batch, n_heads, seq_len, d_head)
        #    - you can practice both .view(...) and .reshape(...)
        # 3. Call your scaled dot-product attention implementation
        # 4. Merge heads back to (batch, seq_len, d_model)
        # 5. Apply output projection
        # (batch, seq_len, d_model)
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)

        # 2) Reshape / transpose to (batch, n_heads, seq_len, d_head)
        #    First view as (B, T, H, d_head) then transpose T and H.
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3) Scaled dot-product attention for each head: (B, H, T, T)
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if mask is not None:
            # Assume mask has 1 for allowed positions and 0 for disallowed.
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4) Apply attention weights to values: (B, H, T, d_head)
        attn_values = attn_weights @ v

        # 5) Merge heads back to (batch, seq_len, d_model)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # 6) Final output projection
        out = self.out_proj(attn_values)

        return out, attn_weights


if __name__ == "__main__":
    print("Running basic checks for 04_multihead_attention.py")

    batch, seq_len, d_model, n_heads = 2, 5, 16, 4
    x = torch.randn(batch, seq_len, d_model)

    try:
        mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
        out, attn = mha(x, mask=None)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, n_heads, seq_len, seq_len)
        # Check that rows of attention weights form probability distributions.
        attn_sum = attn.sum(dim=-1)
        ones = torch.ones_like(attn_sum)
        assert torch.allclose(attn_sum, ones, atol=1e-5), "Attention weights should sum to 1 over keys"
        print("MultiHeadSelfAttention shape + normalization check passed:", out.shape, attn.shape)

        # Optional: with an identity-like input, verify basic symmetry properties.
        x_eye = torch.eye(seq_len, d_model)[:seq_len].unsqueeze(0).expand(batch, -1, -1)
        out_eye, attn_eye = mha(x_eye)
        assert out_eye.shape == (batch, seq_len, d_model)
        print("MultiHeadSelfAttention additional sanity check passed.")
    except NotImplementedError:
        print("MultiHeadSelfAttention: TODOs not implemented yet.")
