# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict

import pytest
import torch

from vllm import EngineArgs, LLM, SamplingParams
from vllm.model_executor.models.openvla import OpenVLAForActionPrediction

from ....utils import large_gpu_test

MODEL_NAME = "openvla/openvla-7b"
ACTION_TOKEN_OFFSET = 32000
ACTION_TOKEN_MAX = 32255


@pytest.mark.core_model
@large_gpu_test(min_gb=24)
def test_openvla_inference(image_assets):
    pytest.importorskip("timm")

    images = [asset.pil_image for asset in image_assets][:1]

    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
    )
    llm = LLM(**(asdict(engine_args) | {"seed": 42}))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=7,
        min_tokens=7,
        ignore_eos=True,
    )

    outputs = llm.generate(
        {
            "prompt": "<image>",
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )

    token_ids = outputs[0].outputs[0].token_ids
    assert len(token_ids) == 7
    assert all(ACTION_TOKEN_OFFSET <= tok <= ACTION_TOKEN_MAX for tok in token_ids)

    actions = OpenVLAForActionPrediction.decode_action_tokens(
        torch.tensor([token_ids]),
        action_token_offset=ACTION_TOKEN_OFFSET,
    )
    assert actions.shape == (1, 7)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
