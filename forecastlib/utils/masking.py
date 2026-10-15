import torch


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu") -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Created on the device directly, a CPU mask would be copied to the GPU with the CPU waiting.
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu") -> None:
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        batch, heads = torch.arange(B, device=device)[:, None, None], torch.arange(H, device=device)[None, :, None]
        indicator = _mask_ex[batch, heads, index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask
