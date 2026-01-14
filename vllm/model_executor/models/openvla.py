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
    PromptReplacement,
    PromptUpdate,
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


class PrismaticVisionBackbone(nn.Module):
    """Fused vision backbone combining DINOv2 and SigLIP.

    OpenVLA uses a fused dual-encoder vision backbone that concatenates
    features from DINOv2 (structural/geometric) and SigLIP (semantic).
    Both are ViT models loaded via TIMM.
    """

    def __init__(
        self,
        timm_model_ids: list[str],
        image_sizes: list[int],
        use_fused_vision_backbone: bool = True,
    ):
        super().__init__()
        self.use_fused = use_fused_vision_backbone
        self.timm_model_ids = timm_model_ids
        self.image_sizes = image_sizes

        # Will be loaded via load_weights
        self.featurizer: Optional[nn.Module] = None
        self.fused_featurizer: Optional[nn.Module] = None
        self.embed_dim: Optional[int] = None

    def _init_timm_models(self):
        """Initialize TIMM models. Called after config is loaded.

        Important: Must specify img_size to match OpenVLA's training configuration.
        DINOv2 defaults to 518x518 but OpenVLA uses 224x224.
        """
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for OpenVLA. Install with: pip install timm"
            ) from e

        # Get image size from config (default 224 for OpenVLA)
        img_size = self.image_sizes[0] if self.image_sizes else 224

        # Primary encoder (DINOv2)
        # Must specify img_size=224 because DINOv2 defaults to 518x518
        self.featurizer = timm.create_model(
            self.timm_model_ids[0],
            pretrained=False,  # weights loaded separately
            num_classes=0,
            img_size=img_size,
        )
        self.embed_dim = self.featurizer.embed_dim

        # Secondary encoder (SigLIP) for fused backbone
        if self.use_fused and len(self.timm_model_ids) > 1:
            self.fused_featurizer = timm.create_model(
                self.timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )
            self.embed_dim += self.fused_featurizer.embed_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract and fuse features from both vision encoders.

        Args:
            pixel_values: Images of shape (batch, channels, height, width).

        Returns:
            Patch features of shape (batch, num_patches, embed_dim).
        """
        if self.featurizer is None:
            raise RuntimeError("Vision backbone not initialized. Call _init_timm_models first.")

        # Get features from primary encoder
        features = self.featurizer.forward_features(pixel_values)

        # Remove CLS token if present (keep only patch tokens)
        if features.shape[1] > (pixel_values.shape[-1] // 14) ** 2:
            features = features[:, 1:, :]

        # Fuse with secondary encoder
        if self.use_fused and self.fused_featurizer is not None:
            fused_features = self.fused_featurizer.forward_features(pixel_values)
            if fused_features.shape[1] > (pixel_values.shape[-1] // 14) ** 2:
                fused_features = fused_features[:, 1:, :]
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


class OpenVLAImagePixelInputs:
    """Schema for OpenVLA image pixel inputs."""

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: torch.Tensor  # Shape: (batch * num_images, 3, height, width)


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


class OpenVLADummyInputsBuilder(BaseDummyInputsBuilder[OpenVLAProcessingInfo]):
    """Builds dummy inputs for profiling OpenVLA."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<image>" * num_images

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
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=224,
                image_height=224,
            )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
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
            return "<image>"
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
                timm_model_ids=self.timm_model_ids,
                image_sizes=self.image_sizes,
                use_fused_vision_backbone=self.use_fused_vision_backbone,
            )
            # Projector will be initialized after vision backbone loads
            self.projector: Optional[PrismaticProjector] = None
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
        # Initialize vision backbone TIMM models
        if self.vision_backbone is not None:
            self.vision_backbone._init_timm_models()

            # Initialize projector after we know vision dimensions
            vision_dim = self.vision_backbone.embed_dim
            llm_dim = self.config.hidden_size if hasattr(self.config, "hidden_size") else 4096
            self.projector = PrismaticProjector(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                use_fused=self.use_fused_vision_backbone,
                prefix="projector",
            )

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
