# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenVLA consistency test - compares vLLM outputs against HuggingFace reference.

This test verifies that vLLM's OpenVLA implementation produces the same
action tokens as the reference HuggingFace Transformers implementation.

Reference results were generated with transformers==4.40.1 and are stored in:
    test_data/openvla_equivalence/transformers_results_4401.json

Usage:
    python tests/models/multimodal/test_openvla_consistency.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Add parent paths for imports
TRIESTE_ROOT = Path(__file__).parent.parent.parent.parent.parent
TEST_DATA_DIR = TRIESTE_ROOT / "test_data" / "openvla_equivalence"

# Reference results from HuggingFace transformers 4.40.1
REFERENCE_FILE = TEST_DATA_DIR / "transformers_results_4401.json"

# OpenVLA action token conversion
VOCAB_SIZE = 32000
N_ACTION_BINS = 256


def token_to_bin(token: int) -> int:
    """Convert action token to bin index."""
    return VOCAB_SIZE - token - 1


def bin_to_action(bin_idx: int) -> float:
    """Convert bin index to normalized action value."""
    return (2 * bin_idx + 1) / 256 - 1


def load_reference_results() -> list[dict[str, Any]]:
    """Load reference results from HuggingFace transformers."""
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE}\n"
            "Please run the HuggingFace baseline first."
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def load_test_samples() -> list[tuple[Image.Image, str, str]]:
    """Load test samples (images and instructions)."""
    samples = []
    for i in range(5):
        img_path = TEST_DATA_DIR / f"sample_{i:03d}.png"
        json_path = TEST_DATA_DIR / f"sample_{i:03d}.json"

        if not img_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Test sample {i} not found")

        image = Image.open(img_path).convert("RGB")
        with open(json_path) as f:
            meta = json.load(f)

        samples.append((image, meta["instruction"], meta["name"]))

    return samples


def run_vllm_inference(
    samples: list[tuple[Image.Image, str, str]]
) -> list[dict[str, Any]]:
    """Run inference using vLLM."""
    # Import vLLM here to avoid import errors when just checking the file
    from vllm import LLM, SamplingParams

    print("Initializing vLLM with OpenVLA model...")
    llm = LLM(
        model="openvla/openvla-7b",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=7,
        min_tokens=7,
        ignore_eos=True,
    )

    results = []
    for image, instruction, name in samples:
        # IMPORTANT: HF's OpenVLA uses trailing space which adds token 29871
        prompt = f"In: What action should the robot take to {instruction}?\nOut: "

        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": [image]},
        }

        outputs = llm.generate(request, sampling_params=sampling_params)
        token_ids = list(outputs[0].outputs[0].token_ids)

        # Convert to actions
        bin_indices = [token_to_bin(t) for t in token_ids]
        actions = [bin_to_action(b) for b in bin_indices]

        results.append({
            "name": name,
            "instruction": instruction,
            "prompt": prompt,
            "action_tokens": token_ids,
            "bin_indices": bin_indices,
            "normalized_actions": actions,
        })

        print(f"  {name}: tokens={token_ids}")

    return results


def compare_results(
    vllm_results: list[dict[str, Any]],
    reference_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare vLLM results against reference."""
    comparison = {
        "total_samples": len(reference_results),
        "exact_matches": 0,
        "token_matches": [],
        "samples": [],
    }

    for vllm_res, ref_res in zip(vllm_results, reference_results):
        vllm_tokens = vllm_res["action_tokens"]
        ref_tokens = ref_res["action_tokens"]

        token_diff = [v - r for v, r in zip(vllm_tokens, ref_tokens)]
        tokens_match = vllm_tokens == ref_tokens
        num_matching = sum(1 for v, r in zip(vllm_tokens, ref_tokens) if v == r)

        if tokens_match:
            comparison["exact_matches"] += 1

        comparison["token_matches"].append(num_matching)

        # Calculate action value differences
        vllm_actions = np.array(vllm_res["normalized_actions"])
        ref_actions = np.array(ref_res["normalized_actions"])
        action_diff = np.abs(vllm_actions - ref_actions)

        comparison["samples"].append({
            "name": vllm_res["name"],
            "instruction": vllm_res["instruction"],
            "vllm_tokens": vllm_tokens,
            "ref_tokens": ref_tokens,
            "token_diff": token_diff,
            "tokens_match": tokens_match,
            "num_matching_tokens": num_matching,
            "max_action_diff": float(action_diff.max()),
            "mean_action_diff": float(action_diff.mean()),
        })

    comparison["match_rate"] = comparison["exact_matches"] / comparison["total_samples"]
    comparison["avg_matching_tokens"] = np.mean(comparison["token_matches"])

    return comparison


def print_comparison(comparison: dict[str, Any]) -> None:
    """Print comparison results in a readable format."""
    print("\n" + "=" * 60)
    print("OpenVLA vLLM vs HuggingFace Consistency Test Results")
    print("=" * 60)

    print(f"\nTotal samples: {comparison['total_samples']}")
    print(f"Exact matches: {comparison['exact_matches']}/{comparison['total_samples']}")
    print(f"Match rate: {comparison['match_rate']*100:.1f}%")
    print(f"Avg matching tokens: {comparison['avg_matching_tokens']:.1f}/7")

    print("\nPer-sample results:")
    print("-" * 60)

    for sample in comparison["samples"]:
        status = "✓ EXACT" if sample["tokens_match"] else "✗ DIFF"
        print(f"\n{sample['name']}: {status}")
        print(f"  Instruction: {sample['instruction']}")
        print(f"  vLLM tokens:  {sample['vllm_tokens']}")
        print(f"  Ref tokens:   {sample['ref_tokens']}")
        print(f"  Token diff:   {sample['token_diff']}")
        print(f"  Matching: {sample['num_matching_tokens']}/7 tokens")
        if not sample["tokens_match"]:
            print(f"  Max action diff: {sample['max_action_diff']:.4f}")
            print(f"  Mean action diff: {sample['mean_action_diff']:.4f}")

    print("\n" + "=" * 60)
    if comparison["match_rate"] >= 0.8:
        print("PASS: >= 80% exact token match achieved")
    else:
        print("FAIL: < 80% exact token match")
    print("=" * 60)


def main():
    """Run the consistency test."""
    print("OpenVLA vLLM Consistency Test")
    print("-" * 40)

    # Check if test data exists
    if not TEST_DATA_DIR.exists():
        print(f"ERROR: Test data directory not found: {TEST_DATA_DIR}")
        return 1

    # Load reference results
    print("Loading reference results from HuggingFace transformers 4.40.1...")
    try:
        reference_results = load_reference_results()
        print(f"  Loaded {len(reference_results)} reference samples")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Load test samples
    print("\nLoading test samples...")
    try:
        samples = load_test_samples()
        print(f"  Loaded {len(samples)} test samples")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Run vLLM inference
    print("\nRunning vLLM inference...")
    vllm_results = run_vllm_inference(samples)

    # Compare results
    print("\nComparing results...")
    comparison = compare_results(vllm_results, reference_results)

    # Print results
    print_comparison(comparison)

    # Save results
    output_file = TEST_DATA_DIR / "vllm_consistency_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "vllm_results": vllm_results,
            "comparison": comparison,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Return exit code based on match rate
    return 0 if comparison["match_rate"] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
