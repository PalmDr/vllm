# OpenVLA

OpenVLA is a vision-language-action model that generates 7 action tokens from
an image-conditioned prompt. vLLM runs it as a multimodal generative model with
an image placeholder in the prompt.

## Performance Metrics (Default Settings)

Defaults: vLLM default engine settings unless noted. Model `openvla/openvla-7b`,
`limit_mm_per_prompt={"image": 1}`, single-image prompt, and greedy decoding for
7 action tokens.

| Metric | Value | Notes |
| --- | --- | --- |
| Request throughput (req/s) | TBD | Default settings |
| Output token throughput (tok/s) | TBD | 7 action tokens |
| Total token throughput (tok/s) | TBD | Prompt + output tokens |
| Median TTFT (s) | TBD | Default settings |
| Median TPOT (s) | TBD | Default settings |
| P99 TTFT (s) | TBD | Default settings |
| P99 TPOT (s) | TBD | Default settings |
| Peak GPU memory (GB) | TBD | Default settings |

Replace the TBD values after collecting the benchmark results on your default
hardware/software stack.
