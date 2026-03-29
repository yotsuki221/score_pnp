# deepinv/models/rdunet_custom.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

# 例: 添付 model.py の RDUNet をリポジトリ側にコピーして import できるようにする
# from .external_rdunet import RDUNet
from .external_rdunet import RDUNet


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # DataParallel 保存時に付く "module." を除去
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


class RDUNetDenoiser(nn.Module):
    """
    score_pnp (同梱 deepinv) 互換デノイザ:
      - forward(x, sigma) を提供
      - potential(x, sigma) を提供（RED/GSPnP の cost 用）
    """
    def __init__(
        self,
        channels: int = 3,
        base_filters: int = 64,
        pretrained: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> None:
        super().__init__()

        # 添付 RDUNet は kwargs キーに空白を含むため辞書で渡すのが安全
        self.net = RDUNet(**{"channels": channels, "base filters": base_filters})

        if pretrained:
            ckpt = torch.load(pretrained, map_location="cpu")

            # checkpoint が "state_dict" を内包する形式だった場合も救う
            if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]

            if not isinstance(ckpt, dict):
                raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

            ckpt = _strip_module_prefix(ckpt)
            self.net.load_state_dict(ckpt, strict=strict)

        self.net.eval()

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma: Any = None) -> torch.Tensor:
        # sigma は互換性のため受け取る（DnCNN も sigma を無視する実装）:
        # deepinv の PnP/RED は denoiser(x, sigma) と位置引数で呼ぶ。　
        return self.net(x)

    @torch.no_grad()
    def potential(self, x: torch.Tensor, sigma: Any, *args, **kwargs) -> torch.Tensor:
        # deepinv/models/dncnn.py 等と同型: 0.5 * ||x - D(x,sigma)||^2
        n = self.forward(x, sigma)
        return 0.5 * torch.norm((x - n).view(x.shape[0], -1), p=2, dim=-1) ** 2
