import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Scaled dot-product attention.

    Common shape convention (you should be able to explain this in the interview):
        q: (batch, n_heads, seq_len_q, d_k)
        k: (batch, n_heads, seq_len_k, d_k)
        v: (batch, n_heads, seq_len_k, d_v)
        mask (optional): broadcastable to (batch, n_heads, seq_len_q, seq_len_k)

    Returns:
        attn_output: (batch, n_heads, seq_len_q, d_v)
        attn_weights: (batch, n_heads, seq_len_q, seq_len_k)

    Steps:
        1. Compute scores = q @ k.transpose(-2, -1) / sqrt(d_k)
        2. Optionally apply mask (e.g., add -inf where mask == 0)
        3. Apply softmax over the last dimension (keys dimension)
        4. Compute output = softmax(scores) @ v
    """
    # TODO: implement scaled dot-product attention with correct shape handling
    raise NotImplementedError


def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a standard causal mask for self-attention:
        mask[i, j] = 0 if j > i (position j is in the future of i)
                    1 otherwise

    Returns:
        mask of shape (1, 1, seq_len, seq_len) suitable for broadcasting.
    """
    # TODO: implement upper-triangular causal mask
    raise NotImplementedError


if __name__ == "__main__":
    print("Running basic checks for 03_scaled_dot_product_attention.py")

    batch, n_heads, seq_len_q, seq_len_k, d_k, d_v = 2, 4, 5, 5, 8, 8
    q = torch.randn(batch, n_heads, seq_len_q, d_k)
    k = torch.randn(batch, n_heads, seq_len_k, d_k)
    v = torch.randn(batch, n_heads, seq_len_k, d_v)

    try:
        mask = causal_mask(seq_len_k)
        assert mask.shape == (1, 1, seq_len_k, seq_len_k)
        # Check that the mask zeros out strictly future positions and keeps current/past.
        # Diagonal and lower triangle should be 1, upper triangle should be 0.
        base = mask[0, 0]
        assert torch.all(base.tril() == 1), "Lower triangle of causal_mask should be 1"
        assert torch.all(base.triu(1) == 0), "Upper triangle of causal_mask should be 0"
        print("causal_mask shape + value check passed:", mask.shape)
    except NotImplementedError:
        print("causal_mask: TODO not implemented yet.")

    try:
        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        assert out.shape == (batch, n_heads, seq_len_q, d_v)
        assert attn.shape == (batch, n_heads, seq_len_q, seq_len_k)
        # Check that attention weights form a proper probability distribution over keys.
        attn_sum = attn.sum(dim=-1)
        ones = torch.ones_like(attn_sum)
        assert torch.allclose(attn_sum, ones, atol=1e-5), "Attention weights should sum to 1 over keys"
        print("scaled_dot_product_attention shape + normalization check passed:", out.shape, attn.shape)

        # Check that causal masking actually removes probability mass from future positions.
        mask = causal_mask(seq_len_k)
        out_masked, attn_masked = scaled_dot_product_attention(q, k, v, mask=mask)
        # For each query i, positions j > i should have very small probability (ideally 0).
        upper = attn_masked[:, :, :, :].triu(1)
        assert torch.all(upper < 1e-4), "Causal mask not properly applied to future positions"
        print("scaled_dot_product_attention causal masking check passed.")
    except NotImplementedError:
        print("scaled_dot_product_attention: TODO not implemented yet.")
