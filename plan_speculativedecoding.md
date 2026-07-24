# Implementing Speculative Decoding in LLMServingSim

## Objective

The objective of this project is to extend **LLMServingSim** to support **Speculative Decoding (SD)** and evaluate its performance on heterogeneous systems such as **xPU-only** and **xPU + PIM** architectures.

The implementation should preserve the existing architecture of LLMServingSim as much as possible while accurately modeling the execution flow of speculative decoding. The simulator should support both conventional autoregressive decoding and speculative decoding through a configurable runtime option.

---

# Background

LLMServingSim 2.0 consists of the following major components.

- Python-based continuous batching scheduler (vLLM style)
- Execution graph generator / operation mapper
- ASTRA-Sim backend for hardware simulation
- Layerwise latency/power profiler
- Memory subsystem including KV cache management

Current observations indicate that **LLMServingSim does not natively support Speculative Decoding**.

Therefore, the simulator must be extended to explicitly model the draft-verify execution pipeline.

---

# Overall Development Plan

## Phase 0. Environment Setup & Codebase Analysis

### Goal

Understand the execution flow of LLMServingSim before modifying the simulator.

### Tasks

1. Clone and build the repository.

```bash
git clone --recurse-submodules
```

2. Build the simulator using the provided Docker/compilation scripts.

3. Execute baseline workloads to verify that the simulator works correctly.

4. Analyze the following core modules.

- Batch Scheduler
- Graph Generator / Operation Mapper
- Memory Model (KV Cache)

5. Trace the complete execution flow of **autoregressive decoding**.

The trace should clearly identify

- request lifecycle
- iteration loop
- scheduling decisions
- graph generation
- hardware execution
- KV cache updates

6. Understand how the layerwise profiler generates latency and power profiles.

This profiler will later be used to profile both

- target model
- draft model

---

## Phase 1. Speculative Decoding Modeling

Before modifying the simulator, define the SD algorithm that will be simulated.

### 1. Draft Model

Initially implement the classic two-model speculative decoding.

- Target Model
- Small Draft Model

Self-speculative methods (e.g., EAGLE) are outside the initial scope.

---

### 2. Acceptance Rate Modeling

Support multiple acceptance-rate models.

#### Option A

Constant acceptance rate

```
acceptance_rate = α
```

Advantages

- deterministic
- fast parameter sweep

---

#### Option B

Replay an offline acceptance trace.

The trace should be collected using

- vLLM speculative decoding
- HuggingFace Assisted Generation

This provides much higher realism and should be used for validation.

---

#### Option C

Random sampling

Example distributions

- geometric distribution
- Bernoulli sampling

---

The simulator should support both

- deterministic mode
- trace replay mode

through configuration.

---

### 3. Draft Length

Expose

```
draft_length = k
```

as a simulator parameter.

---

## Phase 2. Simulator Extension

This is the primary implementation stage.

---

### Phase 2.1 Scheduler Extension

Modify the scheduler so that one speculative decoding iteration consists of two execution stages.

Current execution

```
Decode
↓

1 token
↓

next iteration
```

New execution

```
Draft Model

k autoregressive decoding steps

↓

Target Model

verify k+1 tokens

↓

accept / reject

↓

next SD iteration
```

The scheduler should explicitly distinguish

- draft iterations
- verification iterations

---

### Phase 2.2 Graph Generator / Operation Mapper

Generate execution graphs for

- Draft Model
- Target Model

independently.

The simulator should support different hardware mappings.

Examples

- Draft → xPU
- Draft → PIM
- Target → xPU
- Target → PIM

---

Important observation

Draft execution is primarily

- autoregressive decoding
- GEMV dominated
- memory bound

Verification execution resembles

- mini-prefill
- GEMM dominated
- compute bound

This distinction is expected to significantly influence PIM performance.

---

The memory model should also account for

- Draft Model weights
- Target Model weights
- Draft KV Cache
- Target KV Cache

---

### Phase 2.3 KV Cache Management

Extend KV cache handling.

The simulator must support

- Draft KV cache
- Target KV cache

After verification

Accepted tokens

- commit to target KV cache

Rejected tokens

- rollback draft KV cache

Rollback is a new operation that is currently absent in the simulator.

---

### Phase 2.4 Latency & Power Profiling

Profile

- Target Model
- Draft Model

using the existing layerwise profiler.

Generate latency and power tables for

- xPU
- PIM

Validate that very small draft models still produce accurate decode latency.

---

### Phase 2.5 Runtime Configuration

Support runtime switching.

Example

```yaml
speculative_decoding: true
```

or

```yaml
speculative_decoding: false
```

Both decoding modes should coexist in the same codebase.

---

## Phase 3. Validation

Simulation results must be validated against real execution.

Run identical

- Draft Model
- Target Model

using

- vLLM Speculative Decoding
- HuggingFace Assisted Generation

Compare

- throughput
- TTFT
- TPOT
- acceptance rate

with simulator output.

Target accuracy should be comparable to the original LLMServingSim validation (approximately within 15%).

If large discrepancies appear, perform ablation studies on

- acceptance model
- latency profile
- KV rollback overhead

---

## Phase 4. Experimental Evaluation

Evaluate multiple hardware configurations.

Examples

- xPU only
- xPU + PIM
- Draft on PIM / Target on xPU
- Draft on xPU / Target on PIM

Sweep the following parameters.

- draft length (k)
- acceptance rate (α)
- batch size
- draft model size
- target model size

Compare

- Baseline autoregressive decoding
- Speculative decoding

under identical workloads.

---

### Metrics

Measure

- Throughput (tokens/sec)
- TTFT
- TPOT
- p99 latency
- Power
- Energy per token

Use the existing reporting infrastructure whenever possible.

---

## Phase 5. Result Analysis

Analyze how speculative decoding changes the benefit of PIM.

One expected hypothesis is

- Draft stage is memory-bound
- Verification stage is compute-bound

Therefore,

Speculative Decoding may alter the relative advantage of PIM compared to conventional autoregressive decoding.

Additional analyses should include

- Energy/token tradeoff
- Sensitivity to acceptance rate
- Sensitivity to draft length
- Impact of draft model size

Present the results using figures and tables.

---

# Expected Timeline

| Phase | Duration |
|--------|----------|
| Phase 0 | 1 week |
| Phase 1 | 2 weeks |
| Phase 2 | 3–5 weeks |
| Phase 3 | 1–2 weeks |
| Phase 4 | 2 weeks |
| Phase 5 | 1–2 weeks |

Total estimated duration:

**10–13 weeks**

---

# High-Risk Components

The following components are expected to be the most technically challenging.

1. Acceptance-rate modeling
2. KV cache rollback implementation
3. Validation against real speculative decoding systems

Validation should begin as early as possible to avoid propagating modeling errors into later experiments.

---

# Expected Deliverables

- Fully functional speculative decoding support in LLMServingSim
- Runtime switch between AR decoding and speculative decoding
- Draft/Target execution graph support
- KV cache rollback mechanism
- Latency and power profiles for draft models
- Validation against real systems
- Experimental comparison of xPU-only and xPU+PIM systems
- Comprehensive performance and energy analysis
