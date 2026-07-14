"""vLLM worker extension.

Registered via ``worker_extension_cls="profiler.core.hooks.extension.Extension"``
when constructing the ``vllm.LLM``. vLLM instantiates one Extension per
TP-rank worker process and exposes its methods through
``llm.collective_rpc(method_name, args=...)``.

The sole public method here is ``fire()``: it takes a serialized Shot
plus a catalog slice (the subset of the layer map relevant to the
category being profiled), runs the synthetic batch through
``model_runner.execute_model`` under ``layerwise_profile``, and
returns per-layer CUDA timings.

Measurement protocol per shot:
    1 warmup forward (discarded) — amortises JIT / paged-buffer setup
    N timed forwards inside ``layerwise_profile`` — the hook aggregates
        ``cuda_time_us`` across invocations; ``extract_samples``
        divides by ``invocations`` to return the per-call mean.

N defaults to ``ProfileArgs.measurement_iterations`` (3). A single
timed sample can swing 15-25%% on large GEMMs due to DVFS / boost
jitter; averaging cuts that noise floor dramatically.
"""

from __future__ import annotations

from typing import Any

from profiler.core.hooks.batch import (
    Shot,
    assemble_eagle3_seed_output,
    assemble_eagle3_verify_output,
    assemble_scheduler_output,
)
from profiler.core.hooks.moe_hook import (
    ExpertRoute,
    force_moe_routing,
    single_moe_layer,
)
from profiler.core.hooks.timings import extract_samples


class Extension:
    """Worker-side profiling entry point.

    vLLM instantiates this class inside each TP worker process and
    injects ``self.model_runner`` via attribute assignment before any
    ``collective_rpc`` call.
    """

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """Run one profiling shot and return per-layer timings.

        Args:
            shot_dict: Serialized ``Shot``; rehydrated inside the worker.
            slice_: Serialized catalog slice
                ``{canonical_name: {"vllm": cls, "within": parent, ...}}``
                scoped to the category we're profiling (so timings for
                unrelated layers aren't returned).
            kind: One of ``"dense"``, ``"per_sequence"``, ``"attention"``,
                ``"moe"``. Used to decide whether to forge MoE routing.
            iterations: Number of timed forward passes (averaged via
                the hook's invocation count). Default 3.

        Returns:
            List of ``TimingSample`` as plain dicts (pickled back to host).
        """
        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))

        def _fresh_batch():
            # Rebuild the synthetic SchedulerOutput on every forward so
            # prior-iteration KV writes / request state don't bleed into
            # the next measurement.
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            return batch

        # -- warm-up run, result discarded -----------------------------
        # The first forward pays for JIT compilation, CUDA context
        # setup, paged-attention buffer allocation. We also call
        # sample_tokens to exercise the sampler path (if execute_model
        # returns None it means the scheduler consumed everything and
        # sample_tokens finalizes the step).
        warmup_out = self.model_runner.execute_model(_fresh_batch())
        if warmup_out is None:
            self.model_runner.sample_tokens(None)

        # -- optional MoE routing forge --------------------------------
        route: ExpertRoute | None = None
        if kind == "moe":
            if shot.experts is None or "activated" not in shot.experts:
                raise ValueError(
                    "moe shot missing experts.activated payload"
                )
            moe_layer = single_moe_layer(self.model_runner)
            num_tokens = sum(new for new, _ in shot.requests)
            route = ExpertRoute.forge(
                moe_layer,
                num_tokens=num_tokens,
                activated_experts=int(shot.experts["activated"]),
            )

        # -- measured runs (N iterations, averaged) -------------------
        # Local import so that profiler/__init__.py doesn't require
        # vllm.profiler to be importable at package-import time.
        #
        # vLLM's layerwise_profile hook accumulates ``cuda_time_us``
        # and ``invocations`` across every forward inside its context.
        # ``extract_samples`` divides one by the other, so running
        # execute_model N times here yields the per-call mean — the
        # cheap statistical fix for DVFS / boost-clock jitter that
        # single-sample measurements don't mitigate.
        from vllm.profiler.layerwise_profile import layerwise_profile

        with force_moe_routing(route):
            with layerwise_profile() as hook:
                for _ in range(iterations):
                    measured_out = self.model_runner.execute_model(_fresh_batch())
                    if measured_out is None:
                        self.model_runner.sample_tokens(None)

        stats = hook.results.convert_stats_to_dict()
        summary = stats["summary_stats"]

        samples = extract_samples(summary, slice_)
        return [s.as_dict() for s in samples]

    def fire_eagle3(
        self,
        shot_dict: dict[str, Any],
        iterations: int = 3,
    ) -> dict[str, float]:
        """Measure EAGLE3's initial proposal and steady-state iteration."""
        import torch

        spec_config = getattr(self.model_runner, "speculative_config", None)
        if spec_config is None or spec_config.method != "eagle3":
            raise RuntimeError("fire_eagle3 requires a native vLLM EAGLE3 engine")
        if not hasattr(self.model_runner, "drafter"):
            raise RuntimeError("vLLM model runner did not initialize an EAGLE3 drafter")

        batch_size = int(shot_dict["batch_size"])
        kv_len = int(shot_dict["kv_len"])
        num_speculative_tokens = int(shot_dict["num_speculative_tokens"])
        if num_speculative_tokens != int(spec_config.num_speculative_tokens):
            raise ValueError(
                "shot num_speculative_tokens must match the engine configuration"
            )
        iterations = max(1, int(iterations))

        sequence = int(getattr(self, "_eagle3_profile_sequence", 0))
        live_req_ids = list(getattr(self, "_eagle3_profile_live_req_ids", []))
        elapsed_ms = []
        seed_elapsed_ms = []

        def _execute(scheduler_output):
            output = self.model_runner.execute_model(scheduler_output)
            if output is None:
                output = self.model_runner.sample_tokens(None)
            return output

        # One extra iteration warms the verification query shape. Every
        # iteration gets a fresh synthetic request set so random acceptance
        # from dummy weights cannot alter the next measurement's KV length.
        for measurement_idx in range(iterations + 1):
            request_prefix = f"eagle3_profile_{sequence}"
            sequence += 1
            seed, req_ids = assemble_eagle3_seed_output(
                batch_size=batch_size,
                kv_len=kv_len,
                num_speculative_tokens=num_speculative_tokens,
                model_runner=self.model_runner,
                request_prefix=request_prefix,
                finished_req_ids=live_req_ids,
            )
            # The regular layerwise profile already models the target prefill
            # and sampler. Time only the native EAGLE proposal launched after
            # that sampler so the simulator can add the missing initial draft
            # without double-counting target-model work.
            proposal_times = []
            original_propose = self.model_runner.propose_draft_token_ids

            def _timed_propose(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = original_propose(*args, **kwargs)
                end.record()
                end.synchronize()
                proposal_times.append(float(start.elapsed_time(end)))
                return result

            self.model_runner.propose_draft_token_ids = _timed_propose
            try:
                _execute(seed)
            finally:
                self.model_runner.propose_draft_token_ids = original_propose
            if len(proposal_times) != 1:
                raise RuntimeError(
                    "Expected exactly one EAGLE3 proposal during the seed step, "
                    f"but observed {len(proposal_times)}."
                )
            draft_rows, draft_req_ids = self.model_runner._get_draft_token_ids_cpu()
            draft_by_req = {
                req_id: list(tokens[:num_speculative_tokens])
                for req_id, tokens in zip(draft_req_ids, draft_rows)
                if req_id in req_ids
            }
            if set(draft_by_req) != set(req_ids):
                raise RuntimeError(
                    "vLLM did not return one EAGLE3 draft row per synthetic request"
                )

            verify = assemble_eagle3_verify_output(
                req_ids=req_ids,
                kv_len=kv_len,
                draft_token_ids=draft_by_req,
                model_runner=self.model_runner,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _execute(verify)
            end.record()
            end.synchronize()
            measured_ms = float(start.elapsed_time(end))
            # Synchronize vLLM's asynchronous draft-token D2H copy before
            # the next request set reuses its staging buffer.
            self.model_runner._get_draft_token_ids_cpu()
            if measurement_idx > 0:
                seed_elapsed_ms.append(proposal_times[0])
                elapsed_ms.append(measured_ms)
            live_req_ids = req_ids

        self._eagle3_profile_sequence = sequence
        self._eagle3_profile_live_req_ids = live_req_ids
        return {
            "seed_microseconds": (
                sum(seed_elapsed_ms) * 1000.0 / len(seed_elapsed_ms)
            ),
            "microseconds": sum(elapsed_ms) * 1000.0 / len(elapsed_ms),
        }
