# Speculative Decoding Review Summary

## Scope

Reviewed the committed changes on `speculative_decoding` against `plan.md` and traced the speculative request lifecycle through:

- runtime configuration and scheduler construction
- draft and verification scheduling
- request routing between draft and target instances
- target and draft KV-cache allocation, rollback, and release
- acceptance-rate sampling and trace replay
- completion metrics and simulation-loop integration

No speculative-decoding implementation code was changed during this review.

## Findings

1. Prefix caching and speculative KV accounting use incompatible allocation paths. Draft KV can be freed without having been recorded, while target KV can be counted both as a raw allocation and as a radix-cache insertion.
2. The target model does not execute prompt prefill. The implementation allocates target prompt KV directly before verification, omitting target prefill latency, energy, and computation.
3. `--speculative-decoding` does not activate speculative behavior for the common `pd_type: null` topology. Role inference and routing remain tied to prefill/decode disaggregation rather than `speculative_role`.
4. Verification can stall permanently when `draft_length + 1` exceeds the target token budget or when verification KV memory is unavailable.
5. After verification, draft KV is resized to include target-produced tokens without executing the draft model to construct that KV state.
6. Missing acceptance-trace entries silently produce zero accepted tokens, and trace paths are not resolved consistently after the simulator changes into the `astra-sim/` directory.
7. The branch lacks a runnable speculative cluster/workload example and focused regression tests. Generated profiler CSVs are also committed despite repository guidance against large generated files.

## Validation Performed

- Compared `origin/main...HEAD` with the requirements in `plan.md`.
- Inspected the modified simulator, scheduler, router, request, memory, and acceptance-model code paths.
- Successfully compiled all modified Python modules with Python 3.
- Exercised the constant and random acceptance models in isolation.
- Ran focused Git diff hygiene checks.
- Did not run a full simulation because the ASTRA-Sim analytical binary is not built in this workspace.

## Resolution

The implementation was corrected after the initial review. The following changes were made.

### Runtime configuration and role resolution

Changed `serving/__main__.py`:

- Added `_resolve_speculative_roles()` to validate the speculative topology.
- Explicit `speculative_role: draft` and `speculative_role: target` values take precedence.
- Existing prefill/decode configurations retain backward-compatible inference: `prefill` maps to draft and `decode` maps to target.
- A two-instance topology with `pd_type: null` now infers the first instance as draft and the second as target.
- Ambiguous configurations fail during startup instead of silently running normal autoregressive scheduling.
- Added `_resolve_trace_paths()` so a relative acceptance-trace path is resolved against the directory where `python -m serving` was launched, before the simulator changes into `astra-sim/`.

### Speculative request lifecycle

Changed `serving/core/request.py` and `serving/core/scheduler.py` to use this lifecycle:

```text
draft_prefill
    -> target_prefill
    -> draft (generate up to k proposals)
    -> verify (target verifies k + 1 tokens)
    -> draft_sync (draft executes the target-produced token)
    -> draft / verify until completion
```

The concrete scheduling changes are:

- The draft model executes the complete prompt prefill and keeps its prompt KV.
- The request is transferred to the target model, which independently executes the complete prompt prefill and keeps target prompt KV.
- Prompt throughput is counted for the target prefill only, while both prefills contribute execution latency and energy through their generated traces.
- The draft model generates speculative tokens autoregressively.
- The target model verifies the proposals as a mini-prefill batch.
- Accepted draft tokens plus one target-produced token are committed.
- TTFT is recorded when the first target verification commits output, not when either prompt prefill finishes.
- Tail iterations with no room for a draft proposal use a one-token target verification.

### KV-cache ownership, rollback, and synchronization

Changed `serving/core/scheduler.py` and reused the raw-token resizing support in `serving/core/memory_model.py`:

- Draft and target KV sizes are tracked independently with `speculative_draft_kv_tokens` and `speculative_target_kv_tokens`.
- Speculative schedulers use raw KV accounting instead of inserting the same `Request` into a radix cache owned by two different models.
- Prefix caching is therefore disabled on speculative instances, even if it is enabled globally. This prevents raw allocation plus radix insertion from double-counting target KV and prevents freeing draft KV that was never recorded by that scheduler.
- Target verification temporarily grows target KV to the full verification window, then rolls it back to the committed length.
- Draft KV is rolled back from the complete proposal window to `base + accepted` tokens.
- The new `draft_sync` stage executes the target-produced corrective token on the draft model. Only after this execution does draft KV advance to the full committed length.
- Final completion frees target KV on the target scheduler and draft KV on the pinned draft scheduler.

### Verification progress guarantees

Changed `serving/core/scheduler.py`:

- Proposal length is capped to `target.max_num_batched_tokens - 1`, reserving one verification position for the target-produced token.
- Verification admission removes requests from the candidate batch until the remaining batch fits target KV memory.
- If no verification request can fit, the simulator raises a clear `RuntimeError` instead of returning `None` forever and deadlocking the simulation loop.

### Role-based routing

Changed `serving/core/router.py`:

- Added independent draft-role and target-role scheduler pools.
- New speculative requests start on a draft-role scheduler even when every instance has `pd_type: null`.
- `target_prefill` and `verify` stages route to the pinned target scheduler.
- `draft_prefill`, `draft`, and `draft_sync` stages route to the pinned draft scheduler.
- A workload `model_name` describes the served target model and no longer prevents initial admission to a differently named draft model.
- Missing or invalid role pools produce explicit routing errors.

### Acceptance models

Changed `serving/core/speculative.py`:

- Acceptance rates outside `[0, 1]` are rejected.
- Empty trace files are rejected.
- A missing `(request_id, iteration)` entry raises `KeyError` instead of silently returning zero accepted tokens.
- Global per-iteration trace rows still work as a fallback when the row omits `request_id`.
- Trace replay positions are tracked against the actual request-specific or global key used for the lookup.

### Examples, tests, documentation, and cleanup

Added:

- `configs/cluster/single_node_speculative_instance.json`: runnable two-instance draft/target example.
- `serving/tests/test_speculative.py`: acceptance validation, role inference, and routing tests.
- `serving/tests/test_speculative_scheduler.py`: target-prefill, target-KV rollback, and draft-sync lifecycle tests using the real scheduler completion logic.
- `docs/docs/examples/advanced/speculative-decoding.md`: user documentation and example command.
- A sidebar entry in `docs/sidebars.ts`.

Cleaned up:

- Removed the branch-only note from the root `README.md`; detailed feature documentation belongs on the website.
- Removed the generated GTX2080Ti Llama profiler CSV bundle from the resulting branch diff.
- Preserved the pre-existing untracked Qwen profiling output and `profiler/profile_example.sh`.

## Validation After the Fix

The following focused suite passes all 10 tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/spec-pycache \
  python3 -m unittest discover -s serving/tests -p 'test_speculative*.py' -v
```

The modified Python files compile successfully, the example cluster JSON passes `python3 -m json.tool`, and `git diff --check HEAD` reports no whitespace errors.

A full simulator smoke run could not be completed in this workspace because the ASTRA-Sim analytical binary is not built and the active host Python environment does not provide `pandas`. The example configuration is ready to run in the normal simulator container after building ASTRA-Sim.
