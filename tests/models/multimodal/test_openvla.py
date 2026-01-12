# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for OpenVLA model implementation.

Tests cover:
- Model architecture components
- Vision encoder (dual backbone)
- Projector MLP
- Action token decoding
"""

import pytest
import torch

from vllm.model_executor.models.openvla import (
    OpenVLAForActionPrediction,
    PrismaticProjector,
    PrismaticVisionBackbone,
)


class TestPrismaticVisionBackbone:
    """Tests for the dual vision encoder backbone."""

    def test_initialization(self):
        """Test backbone initializes with correct model IDs."""
        timm_model_ids = [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        backbone = PrismaticVisionBackbone(
            timm_model_ids=timm_model_ids,
            image_sizes=[224, 224],
            use_fused_vision_backbone=True,
        )

        assert backbone.timm_model_ids == timm_model_ids
        assert backbone.image_sizes == [224, 224]
        assert backbone.use_fused is True
        # Models not loaded until _init_timm_models called
        assert backbone.featurizer is None
        assert backbone.fused_featurizer is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        pytest.importorskip("timm")

        backbone = PrismaticVisionBackbone(
            timm_model_ids=[
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            ],
            image_sizes=[224, 224],
            use_fused_vision_backbone=True,
        )
        backbone._init_timm_models()
        backbone = backbone.cuda().eval()

        # Batch of 2 images, 3 channels, 224x224
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224, device="cuda")

        with torch.no_grad():
            features = backbone(pixel_values)

        # Expected: (batch, num_patches, embed_dim)
        # num_patches = (224 / 14)^2 = 256
        # embed_dim = 1024 (DINOv2) + 1152 (SigLIP) = 2176
        assert features.shape[0] == batch_size
        assert features.shape[1] == 256  # (224/14)^2
        assert features.shape[2] == 2176  # 1024 + 1152


class TestPrismaticProjector:
    """Tests for the MLP projector."""

    def test_fused_projector_shape(self):
        """Test fused projector produces correct output shape."""
        vision_dim = 2176  # DINOv2 + SigLIP
        llm_dim = 4096  # Llama-2-7B hidden dim

        projector = PrismaticProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            use_fused=True,
        )

        # Input: (batch, num_patches, vision_dim)
        batch_size = 2
        num_patches = 256
        x = torch.randn(batch_size, num_patches, vision_dim)

        output = projector(x)

        # Output: (batch, num_patches, llm_dim)
        assert output.shape == (batch_size, num_patches, llm_dim)

    def test_simple_projector_shape(self):
        """Test simple (non-fused) projector."""
        vision_dim = 1024
        llm_dim = 4096

        projector = PrismaticProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            use_fused=False,
        )

        batch_size = 1
        num_patches = 256
        x = torch.randn(batch_size, num_patches, vision_dim)

        output = projector(x)
        assert output.shape == (batch_size, num_patches, llm_dim)


class TestActionTokenDecoding:
    """Tests for action token decoding."""

    def test_decode_action_tokens_center(self):
        """Test decoding tokens at center bin (127-128)."""
        # Token 32127 should map to ~0.0 (center of [-1, 1])
        action_tokens = torch.tensor([[32127, 32128, 32127, 32128, 32127, 32128, 32127]])

        actions = OpenVLAForActionPrediction.decode_action_tokens(
            action_tokens, action_token_offset=32000
        )

        # Values should be near 0
        assert actions.shape == (1, 7)
        assert torch.all(actions.abs() < 0.01)  # Close to center

    def test_decode_action_tokens_extremes(self):
        """Test decoding tokens at extreme bins."""
        # Token 32000 -> -1.0, Token 32255 -> +1.0
        action_tokens = torch.tensor([[32000, 32255, 32000, 32255, 32000, 32255, 32000]])

        actions = OpenVLAForActionPrediction.decode_action_tokens(
            action_tokens, action_token_offset=32000
        )

        assert actions.shape == (1, 7)
        assert actions[0, 0].item() == pytest.approx(-1.0, abs=0.01)
        assert actions[0, 1].item() == pytest.approx(1.0, abs=0.01)

    def test_decode_action_tokens_batch(self):
        """Test batch decoding of action tokens."""
        batch_size = 4
        action_tokens = torch.randint(32000, 32256, (batch_size, 7))

        actions = OpenVLAForActionPrediction.decode_action_tokens(
            action_tokens, action_token_offset=32000
        )

        assert actions.shape == (batch_size, 7)
        assert torch.all(actions >= -1.0)
        assert torch.all(actions <= 1.0)


class TestOpenVLAProcessingInfo:
    """Tests for processing info."""

    def test_num_image_tokens(self):
        """Test image token count calculation."""
        # OpenVLA uses 224x224 images with 14x14 patches = 256 tokens
        from vllm.model_executor.models.openvla import OpenVLAProcessingInfo

        # Create minimal mock context
        class MockConfig:
            timm_model_ids = [
                "vit_large_patch14_reg4_dinov2.lvd142m",
                "vit_so400m_patch14_siglip_224",
            ]
            image_sizes = [224, 224]
            use_fused_vision_backbone = True
            image_token_index = 32000
            n_action_bins = 256

        class MockCtx:
            def get_hf_config(self, config_class):
                return MockConfig()

        info = OpenVLAProcessingInfo(MockCtx())
        num_tokens = info.get_num_image_tokens(image_width=224, image_height=224)

        assert num_tokens == 256  # (224 / 14)^2


class TestRegistration:
    """Tests for model registration."""

    def test_model_registered(self):
        """Test OpenVLA is properly registered in vLLM."""
        from vllm.model_executor.models.registry import _MULTIMODAL_MODELS

        assert "OpenVLAForActionPrediction" in _MULTIMODAL_MODELS
        module_name, class_name = _MULTIMODAL_MODELS["OpenVLAForActionPrediction"]
        assert module_name == "openvla"
        assert class_name == "OpenVLAForActionPrediction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
