"""GroundingDINO 与 transformers 5.x 兼容：补回 BertModel.get_head_mask。"""

from __future__ import annotations

from typing import List, Optional, Union

import torch


def _get_head_mask(
    head_mask: Optional[torch.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
    """与 transformers 4.x BertPreTrainedModel.get_head_mask 行为一致。"""
    if head_mask is not None:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1, -1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def apply_gdino_transformers_compat() -> None:
    from transformers import BertModel

    if hasattr(BertModel, "get_head_mask"):
        return
    BertModel.get_head_mask = staticmethod(_get_head_mask)
