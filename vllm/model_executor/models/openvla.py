# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenVLA model implementation for vLLM inference.

OpenVLA (Open Vision-Language-Action) is a 7B VLA model for robotic manipulation.
Architecture: DINOv2 + SigLIP (fused) -> MLP Projector -> Llama-2-7B -> Action Tokens

References:
    - Paper: https://arxiv.org/abs/2406.09246
    - Model: https://huggingface.co/openvla/openvla-7b
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, Optional

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import OpenVLAConfig

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class LayerScale(nn.Module):
    """Layer scale module for DINOv2."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale_factor


class ViTMLP(nn.Module):
    """MLP module for ViT blocks."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, mlp_dim: int | None = None):
        super().__init__()
        # Use explicit mlp_dim if provided, otherwise compute from ratio
        hidden_dim = mlp_dim if mlp_dim is not None else int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTAttention(nn.Module):
    """Attention module for ViT blocks."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DINOv2Block(nn.Module):
    """DINOv2 transformer block with layer scale."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(embed_dim, num_heads)
        self.ls1 = LayerScale(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, mlp_ratio)
        self.ls2 = LayerScale(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SigLIPBlock(nn.Module):
    """SigLIP transformer block (no layer scale)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int = 4304):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, mlp_dim=mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """Attention pooling for SigLIP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int = 4304):
        super().__init__()
        self.latent = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, mlp_dim=mlp_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        latent = self.latent.expand(B, -1, -1)

        q = self.q(latent).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, -1)
        out = self.proj(out)
        out = out + self.mlp(self.norm(out))
        return out.squeeze(1)


class DINOv2Encoder(nn.Module):
    """DINOv2 vision encoder matching OpenVLA checkpoint structure.

    OpenVLA uses DINOv2-Large with 4 register tokens.
    Positional embedding is only applied to patch tokens (not CLS/register).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_blocks: int = 24,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Sequential()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        # Positional embedding only for patches (256), not CLS/register tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            DINOv2Block(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding to patches only
        x = x + self.pos_embed

        # Prepend CLS token and register tokens (they don't get positional embedding)
        cls_token = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_token.expand(B, -1, -1)
        # DINOv2 with registers: [CLS, register tokens, patches]
        x = torch.cat([cls_token, reg_tokens, x], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return patch tokens only (skip CLS and register tokens)
        return x[:, 1 + self.num_register_tokens:, :]


class SigLIPEncoder(nn.Module):
    """SigLIP vision encoder matching OpenVLA checkpoint structure."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1152,
        num_heads: int = 16,
        num_blocks: int = 27,
        mlp_dim: int = 4304,  # SigLIP uses non-standard MLP dimension
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Sequential()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            SigLIPBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_pool = AttentionPooling(embed_dim, num_heads, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x  # Return all patch tokens


class PrismaticVisionBackbone(nn.Module):
    """Fused vision backbone combining DINOv2 and SigLIP.

    OpenVLA uses a fused dual-encoder vision backbone that concatenates
    features from DINOv2 (structural/geometric) and SigLIP (semantic).
    """

    def __init__(
        self,
        image_sizes: list[int],
        use_fused_vision_backbone: bool = True,
    ):
        super().__init__()
        self.use_fused = use_fused_vision_backbone
        image_size = image_sizes[0] if image_sizes else 224

        # DINOv2-Large: 1024 dim, 16 heads, 24 blocks
        self.featurizer = DINOv2Encoder(
            image_size=image_size,
            patch_size=14,
            embed_dim=1024,
            num_heads=16,
            num_blocks=24,
            mlp_ratio=4.0,
            num_register_tokens=4,
        )

        # SigLIP-SO400M: 1152 dim, 16 heads, 27 blocks
        if use_fused_vision_backbone:
            self.fused_featurizer = SigLIPEncoder(
                image_size=image_size,
                patch_size=14,
                embed_dim=1152,
                num_heads=16,
                num_blocks=27,
                mlp_dim=4304,  # SigLIP uses non-standard MLP dimension
            )
            self.embed_dim = 1024 + 1152  # 2176
        else:
            self.fused_featurizer = None
            self.embed_dim = 1024

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and fuse features from both vision encoders.

        Args:
            pixel_values: Images of shape (batch, 6, height, width) where
                          the 6 channels are 2 stacked 3-channel images:
                          - channels 0-2: image for DINOv2
                          - channels 3-5: image for SigLIP

        Returns:
            Patch features of shape (batch, num_patches, embed_dim).
        """
        # Split the stacked images for each encoder
        # PrismaticProcessor outputs [batch, 6, h, w] = [batch, 2*3, h, w]
        dinov2_images = pixel_values[:, :3, :, :]  # First 3 channels
        siglip_images = pixel_values[:, 3:, :, :]  # Last 3 channels

        features = self.featurizer(dinov2_images)

        if self.use_fused and self.fused_featurizer is not None:
            fused_features = self.fused_featurizer(siglip_images)
            features = torch.cat([features, fused_features], dim=-1)

        return features


class PrismaticProjector(nn.Module):
    """MLP projector to align vision features with LLM embedding space.

    For fused vision backbone (OpenVLA default):
        vision_dim -> 4*vision_dim -> llm_dim -> llm_dim
        with GELU activations between layers.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        use_fused: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_fused = use_fused

        if use_fused:
            # Fused projector: 3-layer MLP
            intermediate_dim = 4 * vision_dim
            self.fc1 = ColumnParallelLinear(
                vision_dim,
                intermediate_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            self.act_fn1 = get_act_fn("gelu")
            self.fc2 = RowParallelLinear(
                intermediate_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )
            self.act_fn2 = get_act_fn("gelu")
            self.fc3 = ColumnParallelLinear(
                llm_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc3",
            )
        else:
            # Simple 2-layer MLP
            self.fc1 = ColumnParallelLinear(
                vision_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1",
            )
            self.act_fn1 = get_act_fn("gelu")
            self.fc2 = RowParallelLinear(
                llm_dim,
                llm_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2",
            )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space.

        Args:
            vision_features: Shape (batch, num_patches, vision_dim).

        Returns:
            Projected features of shape (batch, num_patches, llm_dim).
        """
        x, _ = self.fc1(vision_features)
        x = self.act_fn1(x)
        x, _ = self.fc2(x)

        if self.use_fused:
            x = self.act_fn2(x)
            x, _ = self.fc3(x)

        return x


@dataclass
class OpenVLAImagePixelInputs:
    """Schema for OpenVLA image pixel inputs."""

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: torch.Tensor = field(default=None)  # Shape: (batch * num_images, 3, height, width)


class OpenVLAProcessingInfo(BaseProcessingInfo):
    """Processing info for OpenVLA model."""

    def get_hf_config(self) -> OpenVLAConfig:
        return self.ctx.get_hf_config(OpenVLAConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}  # OpenVLA typically uses single image

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        # OpenVLA uses 224x224 images with 14x14 patches = 256 tokens
        return (224 // 14) ** 2  # 256 patches

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=224, height=224)

    def get_max_image_tokens(self) -> int:
        return self.get_num_image_tokens(image_width=224, image_height=224)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Return the maximum number of tokens per image.

        OpenVLA has a fixed image size (224x224) with 14x14 patches = 256 tokens.
        Returning this directly avoids the profiling flow.
        """
        return {"image": self.get_max_image_tokens()}  # 256


class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    """Builds dummy inputs for profiling OpenVLA."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Empty string - image tokens are inserted at prefix, not as replacement
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        return {
            "image": self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
                overrides=mm_options.get("image") if mm_options else None,
            )
        }


class OpenVLAMultiModalProcessor(BaseMultiModalProcessor[OpenVLAProcessingInfo]):
    """Multi-modal processor for OpenVLA."""

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        """Tokenize text without using the HF processor.

        OpenVLA's PrismaticProcessor requires images even for text tokenization,
        so we use the tokenizer directly for text-only processing.
        """
        tokenizer = self.info.ctx.tokenizer
        # Use add_special_tokens=True to include BOS token for prefix matching
        return tokenizer.encode(prompt_text, add_special_tokens=True)

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Apply HF processor on multimodal data.

        OpenVLA's PrismaticProcessor requires images, so we generate dummy
        images if needed for profiling.
        """
        mm_counts = mm_items.get_all_counts()
        num_images = mm_counts.get("image", 0)

        # If no images, create dummy images for profiling
        if num_images == 0:
            num_images = self.allowed_mm_limits.get("image", 1)
            # Create a dummy image for profiling
            import PIL.Image
            dummy_image = PIL.Image.new("RGB", (224, 224), color=(128, 128, 128))
            # Create dummy mm_items with the dummy image
            from vllm.multimodal.parse import ImageProcessorItems
            mm_items = MultiModalDataItems({"image": ImageProcessorItems([dummy_image])})
            mm_counts = {"image": num_images}

        _, mm_processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return mm_processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        # Use image_token_index (32000) as placeholder for image features
        image_token_id = getattr(hf_config, "image_token_index", 32000)
        # Get BOS token from tokenizer
        tokenizer = self.info.ctx.tokenizer
        bos_token_id = tokenizer.bos_token_id

        def get_insertion(item_idx: int):
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=224,
                image_height=224,
            )
            image_tokens = [image_token_id] * num_image_tokens

            # Return with proper token selection for embeddings
            return PromptUpdateDetails.select_token_id(
                image_tokens,
                embed_token_id=image_token_id,
            )

        # Insert image tokens at the start of the prompt (after BOS if present)
        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix(
                    [bos_token_id] if bos_token_id is not None else []
                ),
                insertion=get_insertion,
            ),
        ]


def _build_openvla_processor(
    info: OpenVLAProcessingInfo,
    dummy_inputs: OpenVLADummyInputsBuilder,
    *,
    cache=None,
) -> OpenVLAMultiModalProcessor:
    return OpenVLAMultiModalProcessor(info, dummy_inputs, cache=cache)


@MULTIMODAL_REGISTRY.register_processor(
    _build_openvla_processor,
    info=OpenVLAProcessingInfo,
    dummy_inputs=OpenVLADummyInputsBuilder,
)
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal, SupportsPP):
    """OpenVLA model for action prediction via vLLM.

    Architecture follows Prismatic VLM with fused vision backbone:
    - Vision: DINOv2 + SigLIP (concatenated features)
    - Projector: 3-layer MLP with GELU
    - LLM: Llama-2-7B

    Action prediction:
    - 7D action space: [dx, dy, dz, drx, dry, drz, gripper]
    - 256 bins per dimension (tokens 32000-32255 in Llama vocab)
    - Autoregressive generation of 7 action tokens
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "language_model.model.",
            "language_model.lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            # Return None because image tokens are inserted at prefix,
            # not replaced from a placeholder string in the prompt
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Vision backbone config
        self.timm_model_ids = getattr(
            config,
            "timm_model_ids",
            ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        )
        self.image_sizes = getattr(config, "image_sizes", [224, 224])
        self.use_fused_vision_backbone = getattr(
            config, "use_fused_vision_backbone", True
        )

        # Vision backbone
        if multimodal_config is not None and multimodal_config.get_limit_per_prompt("image"):
            self.vision_backbone = PrismaticVisionBackbone(
                image_sizes=self.image_sizes,
                use_fused_vision_backbone=self.use_fused_vision_backbone,
            )
            # Get LLM hidden dim from text_config
            text_config = getattr(config, "text_config", None)
            llm_dim = getattr(text_config, "hidden_size", 4096) if text_config else 4096
            # Create projector with known dimensions
            self.projector = PrismaticProjector(
                vision_dim=self.vision_backbone.embed_dim,  # 2176 for fused
                llm_dim=llm_dim,
                use_fused=self.use_fused_vision_backbone,
                prefix="projector",
            )
        else:
            self.vision_backbone = None
            self.projector = None

        # Language model
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config if hasattr(config, "text_config") else config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Image token handling
        self.image_token_id = getattr(config, "image_token_index", 32000)

        # Action token config
        self.n_action_bins = getattr(config, "n_action_bins", 256)
        self.action_dim = 7  # 6 DoF pose + gripper

        # Number of image patches (224/14)^2 = 256
        self.num_patches = (self.image_sizes[0] // 14) ** 2

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> OpenVLAImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        return OpenVLAImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _process_image_input(
        self,
        image_input: OpenVLAImagePixelInputs,
    ) -> torch.Tensor:
        """Process image through vision backbone and projector."""
        if self.vision_backbone is None or self.projector is None:
            raise RuntimeError("Vision components not initialized")

        pixel_values = image_input.pixel_values
        vision_features = self.vision_backbone(pixel_values)
        return self.projector(vision_features)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Compute multimodal embeddings from image inputs."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass for OpenVLA.

        Args:
            input_ids: Flattened input_ids for the batch.
            positions: Position indices for input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings with multimodal
                features already merged.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load OpenVLA weights from HuggingFace checkpoint.

        Weight mapping:
        - vision_backbone.featurizer.* -> DINOv2 weights
        - vision_backbone.fused_featurizer.* -> SigLIP weights
        - projector.fc1/fc2/fc3.* -> Projector weights
        - language_model.* -> Llama weights
        """
        skip_prefixes = []
        if self.vision_backbone is None and self.projector is None:
            skip_prefixes.extend(["vision_backbone.", "projector."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self):
        """Get the module prefix in multimodal models."""
        from .module_mapping import MultiModelKeys

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model="vision_backbone",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        """Returns number of multi-modal encoder tokens."""
        return num_image_tokens

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        """Returns number of multi-modal connector tokens."""
        return num_vision_tokens

    @staticmethod
    def decode_action_tokens(
        action_tokens: torch.Tensor,
        action_token_offset: int = 32000,
    ) -> torch.Tensor:
        """Convert action tokens to continuous action values.

        OpenVLA uses 256-bin discretization per action dimension.
        Tokens are in range [action_token_offset, action_token_offset + 255].
        Values are mapped to [-1, 1].

        Args:
            action_tokens: Token IDs of shape (batch, 7).
            action_token_offset: Starting token ID for action tokens.

        Returns:
            Continuous actions of shape (batch, 7) in [-1, 1].
        """
        # Convert token IDs to bin indices
        bin_indices = action_tokens - action_token_offset

        # Map to continuous values: bin / 255 * 2 - 1
        actions = (bin_indices.float() / 255.0) * 2.0 - 1.0

        return actions
