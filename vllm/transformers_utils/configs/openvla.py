# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.configuration_utils import PretrainedConfig


class OpenVLAConfig(PretrainedConfig):
    """Configuration class for OpenVLA model."""

    model_type = "openvla"

    def __init__(
        self,
        timm_model_ids: list[str] | None = None,
        image_sizes: list[int] | None = None,
        use_fused_vision_backbone: bool = True,
        image_token_index: int = 32000,
        n_action_bins: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timm_model_ids = timm_model_ids or [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.image_sizes = image_sizes or [224, 224]
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_token_index = image_token_index
        self.n_action_bins = n_action_bins
