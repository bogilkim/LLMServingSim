import os
import sys
import tempfile
import types
import unittest
from unittest import mock


try:
    import pandas  # noqa: F401
except ImportError:
    sys.modules["pandas"] = types.ModuleType("pandas")

try:
    import msgspec  # noqa: F401
except ImportError:
    module = types.ModuleType("msgspec")

    class Struct:
        def __init_subclass__(cls, **kwargs):
            return super().__init_subclass__()

    module.Struct = Struct
    sys.modules["msgspec"] = module


from serving.core import trace_generator


def _eagle3_table(scale=1.0):
    return {
        "k_values": [4],
        "by_k": {
            4: {
                "batch_values": [1, 4],
                "rows": {
                    1: {
                        "keys": [128, 256],
                        "values": [100 * scale, 200 * scale],
                    },
                    4: {
                        "keys": [128, 256],
                        "values": [400 * scale, 800 * scale],
                    },
                },
            },
        },
    }


def _perf_db():
    return {
        "meta": {
            "eagle3_profile": {
                "enabled": True,
                "draft_model": "org/eagle3-head",
                "seed_table": "tp1/eagle3_seed.csv",
            },
        },
        "tables": {
            1: {
                "eagle3": _eagle3_table(),
                "eagle3_seed": _eagle3_table(0.5),
            },
        },
    }


class Eagle3LookupTest(unittest.TestCase):
    def test_interpolates_batch_and_kv_axes(self):
        latency = trace_generator._lookup_eagle3(
            _perf_db(), tp=1, batch_size=2, kv_len=192,
            num_speculative_tokens=4,
        )
        self.assertEqual(latency, 375)

    def test_looks_up_initial_proposal_table(self):
        latency = trace_generator._lookup_eagle3(
            _perf_db(), tp=1, batch_size=2, kv_len=192,
            num_speculative_tokens=4, table_name="eagle3_seed",
        )
        self.assertEqual(latency, 188)

    def test_requires_exact_profiled_proposal_length(self):
        with self.assertRaisesRegex(KeyError, "k=3"):
            trace_generator._lookup_eagle3(
                _perf_db(), tp=1, batch_size=1, kv_len=128,
                num_speculative_tokens=3,
            )


class Eagle3TraceTest(unittest.TestCase):
    def test_emits_native_aggregate_operation(self):
        batch = types.SimpleNamespace(
            batch_id=7,
            eagle3_batch_size=2,
            eagle3_kv_len=192,
            eagle3_num_speculative_tokens=4,
            eagle3_draft_model="org/eagle3-head",
        )
        ctx = types.SimpleNamespace(perf_db=_perf_db(), tp_size=1)

        with tempfile.NamedTemporaryFile(delete=False) as output:
            output_path = output.name
        try:
            with mock.patch.object(
                    trace_generator, "_build_trace_ctx", return_value=ctx):
                trace_generator._synthesize_eagle3_trace(
                    "GPU", "org/target", {}, 1, 1, 1, 1,
                    None, 0, 0, batch, output_path, {},
                    None, None, 2, "bf16",
                )
            with open(output_path, encoding="utf-8") as f:
                fields = f.readline().split()
            self.assertEqual(fields[0], "eagle3_native")
            self.assertEqual(fields[1], "375")
            self.assertEqual(fields[2], "REMOTE:0")
            self.assertEqual(fields[6], "REMOTE:0")
        finally:
            os.unlink(output_path)

    def test_generate_trace_dispatches_eagle3_batch(self):
        batch = types.SimpleNamespace(
            batch_id=9,
            model="org/target",
            load=0,
            evict=0,
            speculative_stage="eagle3",
        )

        def emit_native(*args, **kwargs):
            output_path = args[11]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(trace_generator.formatter(
                    "eagle3_native", "123", "REMOTE:0", "4",
                    "LOCAL", "0", "REMOTE:0", "4", "NONE", "0", "NONE",
                ))

        with tempfile.TemporaryDirectory() as inputs_root:
            with (
                mock.patch.object(
                    trace_generator,
                    "get_config",
                    return_value={
                        "max_position_embeddings": 4096,
                        "model_type": "llama",
                    },
                ),
                mock.patch.object(
                    trace_generator,
                    "_synthesize_eagle3_trace",
                    side_effect=emit_native,
                ) as native,
                mock.patch.object(trace_generator, "_synthesize_trace") as regular,
            ):
                trace_generator.generate_trace(
                    batch, "GPU", 1, 1, 1, 1,
                    dtype="bfloat16", inputs_root=inputs_root,
                )

            native.assert_called_once()
            regular.assert_not_called()
            output_path = trace_generator.input_path(
                inputs_root, "trace", "GPU", "org/target",
                "instance0_batch9.txt",
            )
            with open(output_path, encoding="utf-8") as f:
                trace = f.read()
            self.assertIn("eagle3_native_0", trace)


    def test_rejects_draft_head_mismatch(self):
        batch = types.SimpleNamespace(
            batch_id=7,
            eagle3_batch_size=1,
            eagle3_kv_len=128,
            eagle3_num_speculative_tokens=4,
            eagle3_draft_model="org/other-head",
        )
        ctx = types.SimpleNamespace(perf_db=_perf_db(), tp_size=1)

        with tempfile.NamedTemporaryFile(delete=False) as output:
            output_path = output.name
        try:
            with mock.patch.object(
                    trace_generator, "_build_trace_ctx", return_value=ctx):
                with self.assertRaisesRegex(ValueError, "draft mismatch"):
                    trace_generator._synthesize_eagle3_trace(
                        "GPU", "org/target", {}, 1, 1, 1, 1,
                        None, 0, 0, batch, output_path, {},
                        None, None, 2, "bf16",
                    )
        finally:
            os.unlink(output_path)


class FusedDraftVerifyTraceTest(unittest.TestCase):
    def test_appends_draft_steps_before_target_verify(self):
        request0 = types.SimpleNamespace(id=0)
        request1 = types.SimpleNamespace(id=1)
        batch = trace_generator.Batch(
            3, "org/target", 10, 20, [5, 5], [10, 10], 2, 0,
            [5, 5], [10, 10], [], 0, 0,
        )
        batch.requests.extend([request0, request1])
        batch.speculative_draft_model = "org/draft"
        batch.speculative_draft_steps = 2
        batch.speculative_draft_kv_before = {0: 10, 1: 20}

        emitted = []

        def build_ctx(hardware, model, *args, **kwargs):
            return types.SimpleNamespace(model=model)

        def build_batch_ctx(sub_batch, ctx):
            return types.SimpleNamespace(batch=sub_batch)

        def emit_body(ctx, bctx, sub_batch, config, block_mode_on, output):
            emitted.append((ctx.model, list(sub_batch.decode_k_list)))
            output.write(trace_generator.formatter(
                ctx.model.replace("/", "_"), "1", "LOCAL", "0",
                "LOCAL", "0", "LOCAL", "0", "NONE", "0", "NONE",
            ))

        with tempfile.NamedTemporaryFile(delete=False) as output:
            output_path = output.name
        try:
            with (
                mock.patch.object(
                    trace_generator, "_build_trace_ctx",
                    side_effect=build_ctx,
                ),
                mock.patch.object(
                    trace_generator, "_build_batch_ctx",
                    side_effect=build_batch_ctx,
                ),
                mock.patch.object(
                    trace_generator, "_emit_standard_trace_body",
                    side_effect=emit_body,
                ),
            ):
                trace_generator._synthesize_fused_draft_verify_trace(
                    "GPU", "org/target", {}, "org/draft", {},
                    1, 1, 1, 1, None, 0, 0, batch, output_path, {},
                    False, None, None, None, None, 2, "bf16", "fp16",
                )

            self.assertEqual(
                emitted,
                [
                    ("org/draft", [10, 20]),
                    ("org/draft", [11, 21]),
                    ("org/target", []),
                ],
            )
        finally:
            os.unlink(output_path)

    def test_generate_trace_dispatches_fused_batch(self):
        batch = types.SimpleNamespace(
            batch_id=11,
            model="org/target",
            load=0,
            evict=0,
            speculative_stage="draft_verify",
            speculative_draft_model="org/draft",
        )

        def config_for(model):
            return {
                "max_position_embeddings": 4096,
                "model_type": "llama",
                "torch_dtype": (
                    "float16" if model == "org/draft" else "bfloat16"),
            }

        def emit_fused(*args, **kwargs):
            output_path = args[13]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(trace_generator.formatter(
                    "draft_then_verify", "123", "REMOTE:0", "4",
                    "LOCAL", "0", "REMOTE:0", "4", "NONE", "0", "NONE",
                ))

        with tempfile.TemporaryDirectory() as inputs_root:
            with (
                mock.patch.object(
                    trace_generator, "get_config", side_effect=config_for),
                mock.patch.object(
                    trace_generator,
                    "_synthesize_fused_draft_verify_trace",
                    side_effect=emit_fused,
                ) as fused,
                mock.patch.object(
                    trace_generator, "_synthesize_trace") as regular,
            ):
                trace_generator.generate_trace(
                    batch, "GPU", 1, 1, 1, 1,
                    dtype="bfloat16", inputs_root=inputs_root,
                )

            fused.assert_called_once()
            regular.assert_not_called()
            output_path = trace_generator.input_path(
                inputs_root, "trace", "GPU", "org/target",
                "instance0_batch11.txt",
            )
            with open(output_path, encoding="utf-8") as f:
                trace = f.read()
            self.assertIn("draft_then_verify_0", trace)


if __name__ == "__main__":
    unittest.main()
