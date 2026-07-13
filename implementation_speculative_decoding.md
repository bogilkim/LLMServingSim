# Speculative Decoding Implementation Notes

This note records the speculative-decoding work currently implemented in `LLMServingSim`.

## What Was Added

- Runtime flags in `serving/__main__.py` for:
  - enabling/disabling speculative decoding
  - draft length
  - acceptance model selection
  - acceptance rate
  - acceptance trace replay
- Speculative state on each `Request` in `serving/core/request.py`:
  - draft/verify stage tracking
  - target and draft token counters
  - draft and target KV tracking
  - pinned draft/target instance ids
- Stage-aware speculative scheduling in `serving/core/scheduler.py`:
  - draft-stage requests stay on the draft scheduler
  - verify-stage requests are scheduled on the target scheduler
  - verify batches use the speculative acceptance model
  - target KV cache is rolled back after verification
  - draft KV cache is released when a request finishes on the target side
- Stage-aware routing in `serving/core/router.py`:
  - speculative draft requests are routed to draft schedulers
  - speculative verify requests are routed to target schedulers
  - per-request instance pinning is preserved when present
- Main-loop integration in `serving/__main__.py`:
  - completion handling now separates final completions from speculative transfers
  - speculative transfer requests are re-routed through the new router helper

## Behavior Modeled

- Draft phase:
  - generates `k` autoregressive speculative tokens
  - tracks draft KV growth separately from the target path
- Verify phase:
  - runs a mini-prefill verification step
  - samples accepted tokens from the configured acceptance model
  - rolls back unaccepted target KV state
- Acceptance models:
  - `constant`
  - `random`
  - `trace`

## Validation Performed

- Syntax-checked the edited modules in memory:
  - `serving/core/request.py`
  - `serving/core/scheduler.py`
  - `serving/core/router.py`
  - `serving/__main__.py`

## Current Limitations

- The implementation is wired through the simulator, but it has not yet been validated against a full speculative-decoding workload.
- Acceptance trace quality and rollback overhead still need end-to-end benchmarking against real vLLM / assisted-generation runs.

## Files Touched

- `[serving/__main__.py](/home/bogil/LLMServingSim/serving/__main__.py)`
- `[serving/core/request.py](/home/bogil/LLMServingSim/serving/core/request.py)`
- `[serving/core/router.py](/home/bogil/LLMServingSim/serving/core/router.py)`
- `[serving/core/scheduler.py](/home/bogil/LLMServingSim/serving/core/scheduler.py)`

