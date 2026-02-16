from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_3bit(x: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 1D tensor into 3-bit codes (0..7) per weight using symmetric quantization.

    We quantize per group:
      scale = max(|x|)/3
      q = round(x/scale) clipped to [-3, 3]
      code = q + 3  -> codes in [0, 6] (7 unused)
    Then pack 8 codes (8*3=24 bits) into 3 bytes.
    """
    assert x.dim() == 1
    assert x.numel() % group_size == 0
    assert group_size % 8 == 0

    xg = x.view(-1, group_size).to(torch.float32).contiguous()
    max_abs = xg.abs().max(dim=-1, keepdim=True).values
    max_abs = torch.clamp(max_abs, min=1e-8)
    scale = max_abs / 3.0  # float32

    q = torch.round(xg / scale).clamp(-3, 3).to(torch.int8)
    codes = (q + 3).to(torch.uint8)  # 0..6 (7 unused)

    # Pack per 8 codes -> 3 bytes
    codes8 = codes.view(codes.size(0), -1, 8)  # [G, group_size/8, 8]
    c0 = codes8[..., 0]
    c1 = codes8[..., 1]
    c2 = codes8[..., 2]
    c3 = codes8[..., 3]
    c4 = codes8[..., 4]
    c5 = codes8[..., 5]
    c6 = codes8[..., 6]
    c7 = codes8[..., 7]

    b0 = (c0) | (c1 << 3) | ((c2 & 0x03) << 6)
    b1 = ((c2 >> 2) & 0x01) | (c3 << 1) | (c4 << 4) | ((c5 & 0x01) << 7)
    b2 = ((c5 >> 1) & 0x03) | (c6 << 2) | (c7 << 5)

    packed = torch.stack([b0, b1, b2], dim=-1).reshape(codes.size(0), -1).to(torch.int8)
    return packed, scale.to(torch.float16)


def block_dequantize_3bit(packed: torch.Tensor, scale: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """
    Reverse of block_quantize_3bit. Returns 1D float32 tensor.
    """
    assert packed.dim() == 2
    assert group_size % 8 == 0

    scale_f = scale.to(torch.float32)  # [G, 1]
    # packed is [G, group_size/8*3]
    triplets = packed.to(torch.uint8).view(packed.size(0), -1, 3)
    b0 = triplets[..., 0]
    b1 = triplets[..., 1]
    b2 = triplets[..., 2]

    c0 = b0 & 0x07
    c1 = (b0 >> 3) & 0x07
    c2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    c3 = (b1 >> 1) & 0x07
    c4 = (b1 >> 4) & 0x07
    c5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    c6 = (b2 >> 2) & 0x07
    c7 = (b2 >> 5) & 0x07

    codes8 = torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1)  # [G, group_size/8, 8]
    codes = codes8.reshape(packed.size(0), group_size).to(torch.int8)

    q = (codes - 3).clamp(-3, 3).to(torch.float32)  # [-3..3]
    xg = q * scale_f
    return xg.reshape(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 64) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        assert (out_features * in_features) % group_size == 0
        assert group_size % 8 == 0

        num_groups = out_features * in_features // group_size
        bytes_per_group = (group_size // 8) * 3  # 8 vals -> 3 bytes

        self.register_buffer(
            "weight_q3",
            torch.zeros(num_groups, bytes_per_group, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(num_groups, 1, dtype=torch.float16),
            persistent=False,
        )

        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self.requires_grad_(False)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            w = weight.detach().to(torch.float32).contiguous().view(-1)
            q3, sc = block_quantize_3bit(w, group_size=self._group_size)
            self.weight_q3.copy_(q3.to(self.weight_q3.device))
            self.weight_scale.copy_(sc.to(self.weight_scale.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w = block_dequantize_3bit(self.weight_q3, self.weight_scale, group_size=self._group_size)
            w = w.view(self._shape)
            return torch.nn.functional.linear(x.to(torch.float32), w, self.bias).to(x.dtype)


class BigNetLowerPrecision(torch.nn.Module):
    """
    Extra credit model: BigNet with <4 bits/param on average (<9MB target).
    Uses 3-bit weights (+ float16 scale per group).
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int, group_size: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 64):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    net = BigNetLowerPrecision()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
