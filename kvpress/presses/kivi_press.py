# SPDX-FileCopyrightText: Copyright (c) 2025 KVPress Contributors
# SPDX-License-Identifier: Apache-2.0


import torch
from dataclasses import dataclass
from torch import nn

from kvpress.presses.base_press import BasePress


def _asym_grouped_quantize_dequantize_per_channel(
    data: torch.Tensor,  # (B, nh, T, D)
    group_size: int,
    num_bits: int,
) -> torch.Tensor:
    """
    Per-channel asymmetric grouped quantization for keys.

    Groups tokens along the sequence dimension (T), computing min/max per group
    across all channels. This is the KIVI key cache quantization scheme.

    Parameters
    ----------
    data : torch.Tensor
        Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
    group_size : int
        Number of tokens per quantization group along the sequence dimension.
    num_bits : int
        Number of bits for quantization (e.g., 2 or 4).

    Returns
    -------
    torch.Tensor
        Dequantized tensor of the same shape as input.
    """
    B, nh, T, D = data.shape
    assert T % group_size == 0, f"Sequence length {T} must be divisible by group_size {group_size}"
    num_groups = T // group_size

    # Reshape: (B, nh, num_groups, group_size, D)
    # Quantize each group of `group_size` tokens independently per channel
    new_shape = (B, nh, num_groups, group_size, D)
    data_groups = data.view(new_shape)

    max_int = 2 ** num_bits - 1

    # Min/max along the group_size dimension (dim=-2), per channel
    mn = torch.min(data_groups, dim=-2, keepdim=True)[0]  # (B, nh, num_groups, 1, D)
    mx = torch.max(data_groups, dim=-2, keepdim=True)[0]  # (B, nh, num_groups, 1, D)

    scale = (mx - mn) / max_int
    # Clamp scale to avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    # Quantize
    quantized = data_groups - mn
    quantized = quantized / scale
    quantized = torch.clamp(quantized, 0, max_int).round_()

    # Dequantize
    dequantized = quantized * scale + mn

    return dequantized.view(B, nh, T, D)


def _asym_grouped_quantize_dequantize_per_token(
    data: torch.Tensor,  # (B, nh, T, D)
    group_size: int,
    num_bits: int,
) -> torch.Tensor:
    """
    Per-token asymmetric grouped quantization for values.

    Groups channels along the head_dim dimension (D), computing min/max per group
    for each token. This is the KIVI value cache quantization scheme.

    Parameters
    ----------
    data : torch.Tensor
        Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
    group_size : int
        Number of channels per quantization group along the head_dim dimension.
    num_bits : int
        Number of bits for quantization (e.g., 2 or 4).

    Returns
    -------
    torch.Tensor
        Dequantized tensor of the same shape as input.
    """
    B, nh, T, D = data.shape
    assert D % group_size == 0, f"Head dimension {D} must be divisible by group_size {group_size}"
    num_groups = D // group_size

    # Reshape: (B, nh, T, num_groups, group_size)
    # Quantize each group of `group_size` channels independently per token
    new_shape = (B, nh, T, num_groups, group_size)
    data_groups = data.view(new_shape)

    max_int = 2 ** num_bits - 1

    # Min/max along the group_size dimension (dim=-1), per token
    mn = torch.min(data_groups, dim=-1, keepdim=True)[0]  # (B, nh, T, num_groups, 1)
    mx = torch.max(data_groups, dim=-1, keepdim=True)[0]  # (B, nh, T, num_groups, 1)

    scale = (mx - mn) / max_int
    # Clamp scale to avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    # Quantize
    quantized = data_groups - mn
    quantized = quantized / scale
    quantized = torch.clamp(quantized, 0, max_int).round_()

    # Dequantize
    dequantized = quantized * scale + mn

    return dequantized.view(B, nh, T, D)


@dataclass
class KIVIPress(BasePress):
    """
    KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (prefill phase only).

    During prefill, KIVI quantizes the key cache per-channel and the value cache
    per-token using asymmetric grouped quantization. A configurable number of recent
    tokens (residual_length) are kept in full precision to preserve accuracy for
    nearby context.

    Unlike most kvpress methods that prune tokens, KIVI compresses by reducing the
    numerical precision of the KV cache entries, introducing quantization error while
    keeping all tokens. This reduces memory usage when the quantized representations
    are stored in packed integer format.

    Based on KIVI (https://arxiv.org/abs/2402.02750).

    Parameters
    ----------
    k_bits : int, default=2
        Number of bits for key cache quantization. Supported: 2, 4.
    v_bits : int, default=2
        Number of bits for value cache quantization. Supported: 2, 4.
    group_size : int, default=128
        Group size for quantization. Keys are grouped along the sequence dimension;
        values are grouped along the head_dim dimension.
    residual_length : int, default=128
        Number of recent tokens to keep in full precision (not quantized).
        These tokens form the "residual" portion of the cache that preserves
        high precision for nearby context during attention computation.
    """

    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 128
    residual_length: int = 128

    def __post_init__(self):
        assert self.k_bits in [2, 4, 8], f"k_bits must be 2, 4, or 8, got {self.k_bits}"
        assert self.v_bits in [2, 4, 8], f"v_bits must be 2, 4, or 8, got {self.v_bits}"
        assert self.group_size > 0, f"group_size must be positive, got {self.group_size}"
        assert self.residual_length >= 0, f"residual_length must be non-negative, got {self.residual_length}"

    @property
    def compression_ratio(self) -> float:
        """
        Effective memory compression ratio of the KV quantization (excluding residual overhead).

        Computed as 1 minus the fraction of bits retained per element:
        1 - (k_bits + v_bits) / 32 - 2 / group_size

        The 2 / group_size term accounts for the scale and min metadata stored per group
        (each stored in fp16). This represents the memory saved by the quantization scheme.

        Returns
        -------
        float
            Compression ratio in [0, 1). 0 means no compression (fp16), higher means more compression.
        """
        return 1 - (self.k_bits + self.v_bits) / 32 - 2 / self.group_size

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply KIVI quantization to keys and values during prefill.

        Keys are quantized per-channel (groups along the sequence dimension).
        Values are quantized per-token (groups along the head_dim dimension).
        The most recent `residual_length` tokens are kept in full precision.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        hidden_states : torch.Tensor
            Input hidden states of shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors of shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors of shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights (unused in KIVI).
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Quantized-dequantized keys and values. Same shapes as input, but with
            quantization error introduced in the non-residual portion.
        """
        k_len = keys.shape[2]

        # If the sequence is shorter than residual_length, no quantization is applied
        if k_len <= self.residual_length:
            return keys, values

        # Split into quantized portion and residual (full-precision) portion
        quant_len = k_len - self.residual_length

        # Quantize keys: per-channel, along the sequence dimension
        # Only quantize the portion that aligns to group_size
        keys_quant_len = (quant_len // self.group_size) * self.group_size
        if keys_quant_len > 0:
            keys_to_quant = keys[:, :, :keys_quant_len, :].contiguous()
            keys_quantized = _asym_grouped_quantize_dequantize_per_channel(
                keys_to_quant, self.group_size, self.k_bits
            )
            # Remainder tokens between keys_quant_len and quant_len stay full precision
            keys = torch.cat([keys_quantized, keys[:, :, keys_quant_len:, :]], dim=2)

        # Quantize values: per-token, along the head_dim dimension
        values_to_quant = values[:, :, :quant_len, :].contiguous()
        values_residual = values[:, :, quant_len:, :].contiguous()

        if values_to_quant.shape[-1] % self.group_size == 0 and values_to_quant.shape[2] > 0:
            values_quantized = _asym_grouped_quantize_dequantize_per_token(
                values_to_quant, self.group_size, self.v_bits
            )
            values = torch.cat([values_quantized, values_residual], dim=2)

        return keys, values
