"""Native vLLM EAGLE3 profiling.

The regular profiler decomposes one target-model forward into canonical
layers. EAGLE3 is different: verification, rejection sampling, and the next
proposal are one version-specific vLLM pipeline. This module profiles that
whole native iteration and writes a compact table consumed atomically by the
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from profiler.core import logger as log
from profiler.core.config import ProfileArgs
from profiler.core.engine import probe_limits, spin_down, spin_up_eagle3
from profiler.core.writer import DedupSink, persist_meta


@dataclass(frozen=True)
class Eagle3Point:
    batch_size: int
    kv_len: int
    num_speculative_tokens: int
    microseconds: float


def _variant_root(out_root: Path, args: ProfileArgs) -> Path:
    model_as_path = Path(args.model)
    model_subpath = (
        model_as_path.name
        if model_as_path.exists() and model_as_path.is_dir()
        else args.model
    )
    return out_root / args.hardware / model_subpath / args.effective_variant


def _feasible(batch_size: int, kv_len: int, args: ProfileArgs, limits) -> bool:
    k = int(args.eagle3_num_speculative_tokens)
    if batch_size > limits.max_num_seqs:
        return False
    if batch_size * (k + 1) > limits.max_num_batched_tokens:
        return False
    if kv_len + k + 1 > limits.max_model_len:
        return False
    block_size = 16
    per_request = ((kv_len + k + 1 + block_size - 1) // block_size) * block_size
    return batch_size * per_request <= limits.num_cache_tokens


def run_eagle3(arch_path: Path, args: ProfileArgs, out_root: Path) -> None:
    """Profile native EAGLE3 iterations for one target/draft pair.

    EAGLE3 draft tensor parallelism is intentionally TP=1 in the first
    implementation. The target profile remains free to contain other TP
    degrees, but this native table is written under tp1.
    """
    if not args.eagle3_model:
        raise ValueError("eagle3 profiling requires --eagle-model")
    if args.tp_degrees != [1]:
        raise ValueError("eagle3 profiling currently requires --tp 1")
    if args.eagle3_num_speculative_tokens < 1:
        raise ValueError("--num-speculative-tokens must be positive")

    variant_root = _variant_root(out_root, args)
    tp_root = variant_root / "tp1"
    iteration_sink = DedupSink(
        tp_root / "eagle3.csv",
        ["batch_size", "kv_len", "num_speculative_tokens"],
    )
    seed_sink = DedupSink(
        tp_root / "eagle3_seed.csv",
        ["batch_size", "kv_len", "num_speculative_tokens"],
    )

    iteration_prior = set()
    seed_prior = set()
    if not args.force:
        iteration_sink.preload()
        seed_sink.preload()
        iteration_prior = iteration_sink.prior_shot_keys()
        seed_prior = seed_sink.prior_shot_keys()

    llm = None
    tmpdir = None
    engine_kwargs = {}
    try:
        with log.stage("booting native EAGLE3 vLLM engine"):
            llm, engine_kwargs, tmpdir = spin_up_eagle3(args)
            limits = probe_limits(llm)

        shots = [
            (int(batch_size), int(kv_len), int(args.eagle3_num_speculative_tokens))
            for batch_size in sorted(set(args.eagle3_batch_sizes))
            for kv_len in sorted(set(args.eagle3_kv_lengths))
            if _feasible(int(batch_size), int(kv_len), args, limits)
        ]
        shots = [
            shot for shot in shots
            if shot not in iteration_prior or shot not in seed_prior
        ]

        if not shots:
            log.info("eagle3: nothing to do (all feasible shots already measured)")
        else:
            with log.progress("TP=1  eagle3", total=len(shots)) as bar:
                for batch_size, kv_len, num_speculative_tokens in shots:
                    raw = llm.collective_rpc(
                        "fire_eagle3",
                        args=(
                            {
                                "batch_size": batch_size,
                                "kv_len": kv_len,
                                "num_speculative_tokens": num_speculative_tokens,
                            },
                            args.measurement_iterations,
                        ),
                    )
                    result = raw[0]
                    if (
                        batch_size, kv_len, num_speculative_tokens
                    ) not in iteration_prior:
                        iteration_sink.coalesce(
                            Eagle3Point(
                                batch_size=batch_size,
                                kv_len=kv_len,
                                num_speculative_tokens=num_speculative_tokens,
                                microseconds=float(result["microseconds"]),
                            )
                        )
                    if (
                        batch_size, kv_len, num_speculative_tokens
                    ) not in seed_prior:
                        seed_sink.coalesce(
                            Eagle3Point(
                                batch_size=batch_size,
                                kv_len=kv_len,
                                num_speculative_tokens=num_speculative_tokens,
                                microseconds=float(result["seed_microseconds"]),
                            )
                        )
                    bar.advance(1)
        iteration_sink.flush()
        seed_sink.flush()
        log.success("eagle3 iteration → %s", iteration_sink.path)
        log.success("eagle3 initial proposal → %s", seed_sink.path)
    finally:
        if llm is not None and tmpdir is not None:
            spin_down(llm, tmpdir)

    persist_meta(args, arch_path, engine_kwargs, variant_root)
