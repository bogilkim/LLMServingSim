---
sidebar_position: 3
title: Speculative decoding
---

# Speculative decoding

## Independent draft model

Speculative decoding uses a draft model to propose tokens and a target model
to verify them. Both models execute the prompt prefill. The target prefill
samples the first output token, and the initial draft proposal completes before
that token is returned and defines TTFT. After each verification, rejected
draft KV is rolled back and the draft model executes the target-produced token
before proposing again.

Use the included two-instance example:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_speculative_instance.json \
  --dataset workloads/example_trace.jsonl \
  --num-reqs 1
```

Each participating instance sets `speculative_decoding: true` and has an explicit `speculative_role` of `draft` or `target`. For a two-instance configuration with `pd_type: null`, roles can also be inferred by order: draft first, target second. Explicit roles are recommended.

A single independent-draft instance is inferred as
`speculative_role: colocated`. Set `speculative_draft_model` to the draft model
identifier. The instance loads both weight sets and uses the target instance's
hardware and TP
degree to resolve both profile bundles. Draft and target batches remain
separate so each stage uses its own latency table and KV shape.

The target instance controls these settings:

- `speculative_draft_length`: maximum proposed tokens per iteration.
- `speculative_acceptance_model`: `constant`, `random`, or `trace`.
- `speculative_acceptance_rate`: value in `[0, 1]` for constant and random modes.
- `speculative_acceptance_trace`: CSV or JSONL path for trace replay.

Trace rows use `request_id`, `iteration`, and `accepted_tokens`. A missing request/iteration entry is an error; it is not interpreted as zero accepted tokens. Relative trace paths are resolved from the repository launch directory.

Speculative KV uses separate raw ownership for the draft and target models. Prefix caching is therefore disabled on speculative instances even if enabled globally; this avoids representing two model-specific KV copies in the single-request radix-cache state.

The target token budget must accommodate verification. The simulator caps the proposal length to `max_num_batched_tokens - 1`, reserving one position for the target-produced token. If even one verification request cannot fit in target KV memory, the run fails with a clear error instead of stalling.

### Interpreting TTFT and ITL

Independent-draft prompt prefills are currently serialized on the request's
critical path: draft prefill, then target prefill. Speculative decoding is a
steady-state decode optimization, so TTFT can be worse than autoregressive
decoding even when TPOT and mean ITL improve.

For proposal length `k`, a steady-state round costs approximately:

```text
k * draft_decode + target_verify(k + 1)
```

The round commits the accepted prefix plus one target token. A configuration
only speeds up decode when that cost is lower than the same number of target
autoregressive decode steps. A draft model that is too large, a low acceptance
rate, or an undersized request batch can therefore make speculative decoding
slower.

All tokens committed by one verification become visible at the same simulated
completion time. ITL records the elapsed interval for the first token in that
burst and zero for the remaining tokens; mean ITL and TPOT therefore retain
the burst's amortized latency.

## EAGLE3

EAGLE3 uses one colocated target instance and a target-specific EAGLE3
head. Its native vLLM verification, rejection-sampling, and next-proposal
pipeline is profiled as one aggregate GPU operation. The simulator uses the
normal target profile for prompt prefill and `tp1/eagle3.csv` for each
speculative iteration.

Inside the vLLM container, first create the normal target profile and then
supplement it with the EAGLE3 table:

```bash
python -m profiler profile meta-llama/Llama-3.1-8B \
  --hardware RTXPRO6000 \
  --tp 1

python -m profiler eagle3 meta-llama/Llama-3.1-8B \
  --hardware RTXPRO6000 \
  --tp 1 \
  --eagle-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --num-speculative-tokens 4
```

The second command preserves the existing target metadata and adds
`tp1/eagle3.csv` plus `meta.yaml::eagle3_profile`. The target and draft use
dummy weights during profiling; kernel shapes and the native vLLM execution
path are retained.

Run the included example in the simulator container:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_eagle3_instance.json \
  --dataset workloads/example_trace.jsonl \
  --num-reqs 1
```

The instance sets:

- `speculative_method: eagle3`
- `speculative_draft_model`: the exact head identifier used for profiling
- `speculative_draft_length`: the same value as profiler
  `--num-speculative-tokens`
- the same acceptance-model settings used by independent draft decoding

The current aggregate profile supports `tp_size: 1`, `pp_size: 1`, and NPU
execution without a `dp_group`. Attention offloading and sub-batch interleaving are rejected
because the native EAGLE3 timing cannot be decomposed into target attention
and draft-head work. Prefix caching is disabled for speculative instances.

Runtime proposal length must have an exact matching row group in
`eagle3.csv`; the simulator does not silently substitute another `k`.
