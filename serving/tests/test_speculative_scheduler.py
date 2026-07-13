import sys
import types
import unittest


def _stub_optional_dependencies():
    for name in ('pandas', 'numpy'):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    try:
        import msgspec  # noqa: F401
    except ImportError:
        module = types.ModuleType('msgspec')

        class Struct:
            def __init_subclass__(cls, **kwargs):
                return super().__init_subclass__()

        module.Struct = Struct
        sys.modules['msgspec'] = module


_stub_optional_dependencies()

from serving.core.request import Request
from serving.core.scheduler import Scheduler
from serving.core.speculative import SpeculativeAcceptanceModel


class _Memory:
    def __init__(self):
        self.adjustments = []

    def adjust_tokens(self, current_tokens, new_tokens, device):
        self.adjustments.append((current_tokens, new_tokens))


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _scheduler(role):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.speculative_decoding = True
    scheduler.speculative_role = role
    scheduler.speculative_draft_length = 4
    scheduler.speculative_acceptance_model = SpeculativeAcceptanceModel(
        'constant', acceptance_rate=0.5)
    scheduler.max_num_batched_tokens = 16
    scheduler.start_npu = 0
    scheduler.num_npus = 1
    scheduler.pd_type = None
    scheduler.enable_prefix_caching = False
    scheduler.prefix_storage = None
    scheduler.prioritize_prefill = False
    scheduler.request = []
    scheduler.done = []
    scheduler.memory = _Memory()
    scheduler.logger = _Logger()
    return scheduler


def _batch(request, stage=None):
    return types.SimpleNamespace(
        batch_id=0,
        end=[],
        requests=[request],
        speculative_stage=stage,
    )


class SpeculativeSchedulerTest(unittest.TestCase):
    def test_target_prefill_runs_before_first_draft(self):
        scheduler = _scheduler('target')
        request = Request(0, 'target', 10, 20, 0, 1)
        request.speculative_active = True
        request.speculative_stage = 'target_prefill'
        request.chunk_len = 10
        scheduler.inflight = [_batch(request)]

        prompt, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual((prompt, generated), (10, 0))
        self.assertEqual(final, [])
        self.assertEqual(transfer, [request])
        self.assertEqual(request.speculative_stage, 'draft')
        self.assertEqual(request.speculative_target_kv_tokens, 10)

    def test_verify_rolls_back_then_requests_draft_sync(self):
        scheduler = _scheduler('target')
        request = Request(0, 'target', 10, 20, 0, 1)
        request.speculative_active = True
        request.speculative_stage = 'verify'
        request.speculative_target_tokens = 10
        request.speculative_target_kv_tokens = 15
        request.speculative_draft_tokens = 4
        request.speculative_verify_tokens = 5
        request.num_computed_tokens = 10
        scheduler.inflight = [_batch(request, 'verify')]

        _, generated, _, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 3)
        self.assertEqual(scheduler.memory.adjustments, [(15, 13)])
        self.assertEqual(transfer, [request])
        self.assertEqual(request.speculative_stage, 'draft_sync')
        self.assertEqual(request.num_computed_tokens, 12)

    def test_draft_sync_executes_target_token(self):
        scheduler = _scheduler('draft')
        request = Request(0, 'draft', 10, 20, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_draft_kv_tokens = 12
        request.num_computed_tokens = 12
        scheduler.inflight = [_batch(request, 'draft')]

        _, generated, _, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 0)
        self.assertEqual(transfer, [])
        self.assertEqual(request.speculative_stage, 'draft')
        self.assertEqual(request.speculative_draft_kv_tokens, 13)


if __name__ == '__main__':
    unittest.main()
