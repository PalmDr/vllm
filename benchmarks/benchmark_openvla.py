# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

import vllm.platforms as platforms
from vllm.assets.image import ImageAsset
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm import EngineArgs, LLM, SamplingParams

# OpenVLA action token range
# Formula: bin_index = vocab_size - token - 1, where vocab_size = 32000
# Bins are in range [0, 255], so tokens are in range [31744, 31999]
# See: https://huggingface.co/openvla/openvla-7b
VOCAB_SIZE = 32000
N_ACTION_BINS = 256
ACTION_TOKEN_MIN = VOCAB_SIZE - N_ACTION_BINS  # 31744
ACTION_TOKEN_MAX = VOCAB_SIZE - 1  # 31999


def _percentiles(values: list[float], percentiles: list[int]) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in percentiles}
    arr = np.array(values)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenVLA inference performance in vLLM."
    )
    parser.add_argument("--model", type=str, default="openvla/openvla-7b")
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "timm is required for OpenVLA benchmarking. "
            "Install with: pip install timm"
        ) from exc

    platforms.current_platform = (
        CudaPlatform() if torch.cuda.is_available() else CpuPlatform()
    )

    engine_args = EngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    llm = LLM(**(asdict(engine_args) | {"seed": 0}))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=7,
        min_tokens=7,
        ignore_eos=True,
    )

    image = ImageAsset("stop_sign").pil_image
    # OpenVLA expects a specific prompt format with the instruction
    # Image tokens are inserted at the prefix (before the text prompt)
    instruction = "pick up the object"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": [image]},
    }

    def run_once() -> tuple[float, dict[str, Any]]:
        start = time.perf_counter()
        outputs = llm.generate(request, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        metrics = output.metrics
        prompt_token_ids = output.prompt_token_ids or []
        prompt_tokens = len(prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        token_ids = output.outputs[0].token_ids

        if not all(ACTION_TOKEN_MIN <= tok <= ACTION_TOKEN_MAX for tok in token_ids):
            print(f"Output token IDs: {token_ids}")
            print(f"Expected tokens in range [{ACTION_TOKEN_MIN}, {ACTION_TOKEN_MAX}]")
            raise RuntimeError("OpenVLA output tokens are outside action range.")

        record = {
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft": metrics.first_token_latency if metrics else 0.0,
            "num_generation_tokens": metrics.num_generation_tokens if metrics else 0,
            "first_token_ts": metrics.first_token_ts if metrics else 0.0,
            "last_token_ts": metrics.last_token_ts if metrics else 0.0,
        }
        return elapsed, record

    for _ in range(args.num_warmup):
        run_once()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for _ in range(args.num_iters):
        elapsed, record = run_once()
        total_elapsed += elapsed
        records.append(record)

    ttfts = [r["ttft"] for r in records]
    tpot_values = []
    for r in records:
        num_tokens = r["num_generation_tokens"]
        if num_tokens > 1:
            tpot_values.append(
                (r["last_token_ts"] - r["first_token_ts"]) / (num_tokens - 1)
            )
        else:
            tpot_values.append(0.0)

    prompt_tokens = sum(r["prompt_tokens"] for r in records)
    output_tokens = sum(r["output_tokens"] for r in records)
    total_tokens = prompt_tokens + output_tokens
    request_throughput = len(records) / total_elapsed if total_elapsed else 0.0
    output_tok_throughput = output_tokens / total_elapsed if total_elapsed else 0.0
    total_tok_throughput = total_tokens / total_elapsed if total_elapsed else 0.0

    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    results = {
        "model": args.model,
        "num_iters": args.num_iters,
        "num_warmup": args.num_warmup,
        "request_throughput": request_throughput,
        "output_token_throughput": output_tok_throughput,
        "total_token_throughput": total_tok_throughput,
        "ttft_median": float(np.median(ttfts)) if ttfts else 0.0,
        "ttft_p99": _percentiles(ttfts, [99]).get(99, 0.0),
        "tpot_median": float(np.median(tpot_values)) if tpot_values else 0.0,
        "tpot_p99": _percentiles(tpot_values, [99]).get(99, 0.0),
        "peak_gpu_mem_gb": peak_mem_gb,
    }

    print(json.dumps(results, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
