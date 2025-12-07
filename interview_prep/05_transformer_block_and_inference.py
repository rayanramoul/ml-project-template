from typing import Optional, Tuple
import math

import torch
from torch.nn import functional as F
from torch import nn


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

        # 1) Project x into q, k, v: (B, T, d_model)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2) Reshape / transpose to (batch, n_heads, seq_len, d_head)
        #    First view as (B, T, H, d_head) then transpose T and H -> (B, H, T, d_head)
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


class PositionwiseFeedForward(nn.Module):
    """
    Standard Transformer FFN:
        FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # First linear projects d_model -> d_ff (expansion)
        self.fc1 = nn.Linear(d_model, d_ff)
        # Second linear projects d_ff -> d_model (back to model dimension)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Transformer FFN: Linear2(ReLU(Linear1(x)))
        return self.fc2(F.relu(self.fc1(x)))


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
        x -> LayerNorm -> MHA -> residual
          -> LayerNorm -> FFN -> residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
        """
        # 1) Apply LayerNorm to x (pre-norm)
        h = self.norm1(x)

        # 2) Multi-head self-attention (ignore returned weights)
        attn_out, _ = self.self_attn(h, mask=mask)

        # 3) Add residual connection + dropout
        x = x + self.dropout(attn_out)

        # 4) Apply second LayerNorm
        h2 = self.norm2(x)

        # 5) Apply FFN + residual + dropout
        ffn_out = self.ffn(h2)
        x = x + self.dropout(ffn_out)

        return x


def greedy_autoregressive_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Very simple autoregressive generation loop (no KV cache).

    Args:
        model: a model that maps input_ids -> logits over vocabulary
        input_ids: (batch, seq_len) tensor of token ids
        max_new_tokens: how many new tokens to generate
        pad_token_id: id used for padding (if needed)
        eos_token_id: if provided, stop generation when this token is produced

    Returns:
        generated_ids: (batch, seq_len + max_new_tokens)

    This is intentionally high-level; in an interview you may:
        - implement this around a simple Transformer language model
        - or just explain how you'd adapt it for KV-cache.
    """
    # Make a working copy so we don't modify the caller's tensor in-place
    generated = input_ids

    for _ in range(max_new_tokens):
        # 1) Run model to obtain logits over vocabulary for each position
        logits = model(generated)  # (batch, seq_len, vocab_size)

        # 2) Take the last time step's logits and pick the most likely token
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
        next_token = next_token_logits.argmax(dim=-1)  # (batch,)

        # 3) Append the new token to the sequence
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        # 4) If eos_token_id is defined, optionally stop when all sequences hit EOS
        if eos_token_id is not None:
            if torch.all(next_token == eos_token_id):
                break

    return generated


if __name__ == "__main__":
    print("Running basic checks for 05_transformer_block_and_inference.py")

    batch, seq_len, d_model, n_heads, d_ff = 2, 5, 16, 4, 64
    x = torch.randn(batch, seq_len, d_model)

    try:
        block = TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        out = block(x)
        assert out.shape == (batch, seq_len, d_model)
        # Basic sanity: output should be finite and not identically equal to input.
        assert torch.isfinite(out).all(), "TransformerEncoderBlock output contains non-finite values"
        assert not torch.allclose(out, x), "TransformerEncoderBlock output is identical to input"
        print("TransformerEncoderBlock shape + basic value checks passed:", out.shape)
    except NotImplementedError:
        print("TransformerEncoderBlock: TODOs not implemented yet.")

    # Optional: simple test harness for greedy_autoregressive_generate
    class _ToyLM(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.emb = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # input_ids: (batch, seq_len)
            x = self.emb(input_ids)
            logits = self.linear(x)
            return logits  # (batch, seq_len, vocab_size)

    vocab_size = 10
    toy_model = _ToyLM(vocab_size=vocab_size)
    input_ids = torch.zeros((2, 3), dtype=torch.long)  # start with all zeros

    try:
        generated = greedy_autoregressive_generate(
            model=toy_model,
            input_ids=input_ids,
            max_new_tokens=4,
            pad_token_id=0,
            eos_token_id=None,
        )
        assert generated.shape == (2, 3 + 4)
        print("greedy_autoregressive_generate shape check passed:", generated.shape)
    except NotImplementedError:
        print("greedy_autoregressive_generate: TODO not implemented yet.")
