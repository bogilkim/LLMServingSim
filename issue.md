# Colocated speculative decoding forms pathologically small batches

## Problem

With roughly 300 requests, autoregressive decoding builds useful continuous
batches, while colocated independent-draft speculative decoding repeatedly
processes only the first one or two requests. The resulting generation
throughput can be far below the autoregressive baseline even with a high
configured acceptance rate.

Speculative decoding is not unconditionally faster: for proposal length `k`,
one round costs approximately `k * draft_decode + target_verify(k + 1)`, and it
only wins when that cost is below the equivalent target-only decode work.
However, the observed one-or-two-request steady state is a scheduler bug, not
an expected speculative-decoding tradeoff.

## Root cause

The colocated scheduler stores draft- and target-stage requests in one FIFO
queue but must issue separate model batches. `schedule_base()` previously chose
the model from the oldest ready request and filtered out every request for the
other model.

Once that oldest request entered `draft`/`draft_sync`, it stayed at the front
through multiple draft steps and verification rounds. Newer requests that had
finished draft prefill and were ready for `target_prefill` could not run until
the old cohort completed its entire output. Consequently, the initial cohort
size (often one or two requests under an arrival stream) remained the effective
speculative batch size even when hundreds of requests were queued.

## Resolution

- Track the model used by the last colocated speculative batch.
- When both draft and target work are ready, select the other model.
- Preserve FIFO order within each model and continue keeping draft and target
  batches separate so they use the correct latency table and KV shape.
- Count target verification as target-model service for the alternation policy.

## Acceptance criteria

- A ready `target_prefill` request is scheduled after a draft-model batch even
  when an older request is still in the `draft` stage.
- A colocated batch never mixes target and draft model work.
- Existing speculative KV ownership, token accounting, TTFT, and acceptance
  behavior remain unchanged.
- The speculative scheduler regression suite passes.

