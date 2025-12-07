import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) module.

    Idea:
        - We have many query heads (n_q) but fewer key/value heads (n_kv).
        - Multiple query heads share the same key/value head (grouping).

    Shape convention:
        x: (batch, seq_len, d_model)

    Args:
        d_model: model dimension
        n_q_heads: number of query heads
        n_kv_heads: number of key/value heads (must divide n_q_heads)
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        assert d_model % n_q_heads == 0, "d_model must be divisible by n_q_heads"

        # Per-query-head dimension and grouping factor
        self.d_head = d_model // n_q_heads
        self.groups = n_q_heads // n_kv_heads  # how many query heads share one KV head

        # Projections:
        # - Queries: d_model -> d_model (n_q_heads * d_head)
        # - Keys/Values: d_model -> n_kv_heads * d_head (shared across groups of query heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)

        # Output projection after concatenating all query heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional mask broadcastable to
                  (batch, n_kv_heads, seq_len_q, seq_len_k)

        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, n_q_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape

        # 1) Compute q, k, v projections
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  # (B, T, n_kv_heads * d_head)
        v = self.v_proj(x)  # (B, T, n_kv_heads * d_head)

        # 2) Reshape:
        #    q: (B, T, n_q, d_head) -> (B, n_q, T, d_head)
        #    k, v: (B, T, n_kv, d_head) -> (B, n_kv, T, d_head)
        q = q.view(batch, seq_len, self.n_q_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # 3) Group query heads by KV heads.
        #    q: (B, n_q, T, d) -> (B, n_kv, groups, T, d)
        q = q.view(batch, self.n_kv_heads, self.groups, seq_len, self.d_head)
        #    k, v: (B, n_kv, T, d) -> (B, n_kv, 1, T, d) then broadcast over groups
        k = k.unsqueeze(2)  # (B, n_kv, 1, T, d)
        v = v.unsqueeze(2)  # (B, n_kv, 1, T, d)

        # 4) Scaled dot-product attention within each group:
        #    scores: (B, n_kv, groups, T_q, T_k)
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if mask is not None:
            # mask expected to be broadcastable to (B, n_kv_heads, groups, T_q, T_k)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 5) Apply attention to values:
        #    (B, n_kv, groups, T_q, T_k) @ (B, n_kv, 1, T_k, d) -> (B, n_kv, groups, T_q, d)
        attn_output = attn_weights @ v

        # 6) Merge groups back into n_q_heads: (B, n_kv, groups, T, d) -> (B, n_q, T, d)
        attn_output = attn_output.view(batch, self.n_q_heads, seq_len, self.d_head)
        attn_weights_flat = attn_weights.view(batch, self.n_q_heads, seq_len, seq_len)

        # 7) Merge heads and apply out_proj: (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(attn_output)

        return out, attn_weights_flat


class KVCache:
    """
    Minimal KV cache abstraction for autoregressive attention.

    We store past keys and values and append new ones at each step.

    Shapes:
        k, v: (batch, n_heads, seq_len, d_head)
    """

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        Append new keys/values along the sequence dimension.

        Args:
            k_new, v_new: (batch, n_heads, 1, d_head)
        """
        # If cache is empty, initialize with first values
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            # Concatenate along sequence dimension (dim=2)
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            k, v: cached tensors of shape (batch, n_heads, total_seq_len, d_head)
        """
        if self.k is None or self.v is None:
            raise RuntimeError("KVCache is empty. Call append() before get().")
        return self.k, self.v


def causal_mask_from_length(seq_len: int) -> torch.Tensor:
    """
    Build a causal mask with shape (1, 1, seq_len, seq_len).
    mask[i, j] = 0 if j > i (future positions), 1 otherwise.
    """
    # Lower-triangular matrix of ones -> allow attending to self and past tokens
    base = torch.ones((seq_len, seq_len), dtype=torch.float32)
    base = torch.tril(base)
    # Add singleton batch and head dimensions
    mask = base.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask


class CrossAttention(nn.Module):
    """
    Cross-attention module:
        queries come from one sequence (e.g., decoder),
        keys/values come from another (e.g., encoder).

    Shape convention:
        q: (batch, tgt_len, d_model)  # target / decoder side
        kv: (batch, src_len, d_model) # source / encoder side
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        # Standard multi-head projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, tgt_len, d_model)
            kv: (batch, src_len, d_model)
            mask: optional mask broadcastable to (batch, n_heads, tgt_len, src_len)

        Returns:
            out: (batch, tgt_len, d_model)
            attn_weights: (batch, n_heads, tgt_len, src_len)
        """
        batch, tgt_len, _ = q.shape
        _, src_len, _ = kv.shape

        # Project to Q, K, V
        Q = self.q_proj(q)   # (B, T_tgt, d_model)
        K = self.k_proj(kv)  # (B, T_src, d_model)
        V = self.v_proj(kv)  # (B, T_src, d_model)

        # Reshape to multi-head format
        Q = Q.view(batch, tgt_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_tgt, d_head)
        K = K.view(batch, src_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_src, d_head)
        V = V.view(batch, src_len, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_src, d_head)

        # Scaled dot-product attention over encoder positions
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_head)  # (B, H, T_tgt, T_src)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # Apply to values
        attn_output = attn_weights @ V  # (B, H, T_tgt, d_head)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, tgt_len, self.d_model)
        out = self.out_proj(attn_output)

        return out, attn_weights


def sequence_cross_entropy_with_padding(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Cross-entropy over sequences with padding tokens ignored.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) integer token ids
        pad_token_id: token id used for padding (these positions should be ignored)

    Returns:
        Scalar loss (averaged over non-pad tokens)
    """
    # 1. Flatten logits and targets
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch * seq_len, vocab_size)
    targets_flat = targets.view(batch * seq_len)

    # 2. Mask out pad positions
    non_pad_mask = targets_flat != pad_token_id  # (batch * seq_len,)

    # 3. Per-token cross-entropy
    per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # 4. Mask and normalize by number of non-pad tokens
    if non_pad_mask.sum() == 0:
        # Avoid division by zero: return zero loss if everything is padding
        return per_token_loss.new_tensor(0.0)

    loss = (per_token_loss * non_pad_mask.float()).sum() / non_pad_mask.float().sum()
    return loss


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    For each token vector x (last dimension = d_model):
        rms(x) = sqrt(mean(x^2))
        y = x / rms(x) * weight

    Unlike LayerNorm, RMSNorm does not subtract the mean; it only normalizes by
    the root mean square and applies a learned scale.
    """

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale parameter, one per feature
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            Tensor of same shape as x.
        """
        # 1. Compute mean of squared values along the last dimension
        rms = torch.mean(x.pow(2), dim=-1, keepdim=True)

        # 2. Take sqrt with epsilon for numerical stability
        rms = torch.sqrt(rms + self.eps)  # (..., 1)

        # 3. Normalize and scale
        x_norm = x / rms
        return x_norm * self.weight


class PagedAttention(nn.Module):
    """
    Simple "paged" attention implementation.

    Instead of attending over the full key/value sequence, we only attend over
    a fixed-size window of the most recent context positions, which approximates
    the idea of paging / sliding context.

    Expects already-projected and multi-head-shaped tensors.
    """

    def __init__(self, max_context: int) -> None:
        super().__init__()
        self.max_context = max_context

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, n_heads, tgt_len, d_head)
            k: (batch, n_heads, src_len, d_head)
            v: (batch, n_heads, src_len, d_head)
            mask: optional mask broadcastable to (batch, n_heads, tgt_len, window_len)

        Returns:
            out: (batch, n_heads, tgt_len, d_head)
            attn_weights: (batch, n_heads, tgt_len, window_len)
        """
        batch, n_heads, tgt_len, d_head = q.shape
        _, _, src_len, _ = k.shape

        # Restrict keys/values to the most recent window
        window_len = min(self.max_context, src_len)
        k_window = k[:, :, -window_len:, :]  # (B, H, W, d_head)
        v_window = v[:, :, -window_len:, :]  # (B, H, W, d_head)

        # Scaled dot-product attention over the window
        scores = q @ k_window.transpose(-1, -2) / math.sqrt(d_head)  # (B, H, T_tgt, W)
        if mask is not None:
            # Assume mask is already aligned to the truncated key dimension
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ v_window  # (B, H, T_tgt, d_head)
        return out, attn_weights


if __name__ == "__main__":
    print("Running basic checks for 06_advanced_attention_and_norms.py")

    # These checks are very light and mostly verify shapes / NotImplemented status.

    batch, seq_len, d_model = 2, 5, 16
    n_q_heads, n_kv_heads = 8, 4

    x = torch.randn(batch, seq_len, d_model)

    try:
        gqa = GroupedQueryAttention(d_model=d_model, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)
        out, attn = gqa(x)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, n_q_heads, seq_len, seq_len)
        print("GroupedQueryAttention shape check passed:", out.shape, attn.shape)
    except NotImplementedError:
        print("GroupedQueryAttention: TODOs not implemented yet.")

    try:
        cache = KVCache()
        k_new = torch.randn(batch, 4, 1, 8)
        v_new = torch.randn(batch, 4, 1, 8)
        cache.append(k_new, v_new)
        k_cached, v_cached = cache.get()
        assert k_cached.shape == (batch, 4, 1, 8)
        assert v_cached.shape == (batch, 4, 1, 8)
        print("KVCache basic check passed:", k_cached.shape)
    except NotImplementedError:
        print("KVCache: TODOs not implemented yet.")

    try:
        mask = causal_mask_from_length(seq_len=6)
        assert mask.shape == (1, 1, 6, 6)
        print("causal_mask_from_length shape check passed:", mask.shape)
    except NotImplementedError:
        print("causal_mask_from_length: TODO not implemented yet.")

    try:
        cross_attn = CrossAttention(d_model=d_model, n_heads=4)
        q = torch.randn(batch, seq_len, d_model)
        kv = torch.randn(batch, seq_len + 2, d_model)
        out, attn = cross_attn(q, kv)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, 4, seq_len, seq_len + 2)
        print("CrossAttention shape check passed:", out.shape, attn.shape)
    except NotImplementedError:
        print("CrossAttention: TODOs not implemented yet.")

    try:
        logits = torch.randn(batch, seq_len, 10)
        targets = torch.randint(0, 10, (batch, seq_len))
        pad_id = 0
        loss = sequence_cross_entropy_with_padding(logits, targets, pad_token_id=pad_id)
        assert loss.shape == ()
        print("sequence_cross_entropy_with_padding scalar check passed:", loss.item())
    except NotImplementedError:
        print("sequence_cross_entropy_with_padding: TODO not implemented yet.")

    try:
        rms = RMSNorm(d_model=d_model)
        y = rms(x)
        assert y.shape == x.shape
        print("RMSNorm shape check passed:", y.shape)
    except NotImplementedError:
        print("RMSNorm: TODOs not implemented yet.")

    try:
        # Simple PagedAttention sanity check
        batch, n_heads, tgt_len, src_len, d_head = 2, 4, 3, 10, 8
        q = torch.randn(batch, n_heads, tgt_len, d_head)
        k = torch.randn(batch, n_heads, src_len, d_head)
        v = torch.randn(batch, n_heads, src_len, d_head)
        paged_attn = PagedAttention(max_context=5)
        out, attn = paged_attn(q, k, v)
        assert out.shape == (batch, n_heads, tgt_len, d_head)
        assert attn.shape[0] == batch and attn.shape[1] == n_heads and attn.shape[2] == tgt_len
        print("PagedAttention shape check passed:", out.shape, attn.shape)
    except NotImplementedError:
        print("PagedAttention: TODOs not implemented yet.")
