---
sidebar_position: 3
title: Speculative decoding
---

# Speculative decoding

Speculative decoding uses a draft instance to propose tokens and a target instance to verify them. Both instances execute the prompt prefill. After each verification, rejected draft KV is rolled back and the draft model executes the target-produced token before proposing again.

Use the included two-instance example:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_speculative_instance.json \
  --dataset workloads/example_trace.jsonl \
  --num-reqs 1
```

Each participating instance sets `speculative_decoding: true` and has an explicit `speculative_role` of `draft` or `target`. For a two-instance configuration with `pd_type: null`, roles can also be inferred by order: draft first, target second. Explicit roles are recommended.

The target instance controls these settings:

- `speculative_draft_length`: maximum proposed tokens per iteration.
- `speculative_acceptance_model`: `constant`, `random`, or `trace`.
- `speculative_acceptance_rate`: value in `[0, 1]` for constant and random modes.
- `speculative_acceptance_trace`: CSV or JSONL path for trace replay.

Trace rows use `request_id`, `iteration`, and `accepted_tokens`. A missing request/iteration entry is an error; it is not interpreted as zero accepted tokens. Relative trace paths are resolved from the repository launch directory.

Speculative KV uses separate raw ownership for the draft and target models. Prefix caching is therefore disabled on speculative instances even if enabled globally; this avoids representing two model-specific KV copies in the single-request radix-cache state.

The target token budget must accommodate verification. The simulator caps the proposal length to `max_num_batched_tokens - 1`, reserving one position for the target-produced token. If even one verification request cannot fit in target KV memory, the run fails with a clear error instead of stalling.
