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
    scheduler.max_num_seqs = 8
    scheduler.pp_size = 1
    scheduler.model = 'target'
    scheduler.speculative_draft_model = 'org/eagle3-head'
    scheduler.inflight = []
    scheduler.batch_ids = -1
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


def _batch(request, stage=None, scheduled_tokens=None):
    return types.SimpleNamespace(
        batch_id=0,
        end=[],
        requests=[request],
        speculative_stage=stage,
        scheduled_tokens=scheduled_tokens,
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

    def test_eagle3_prefill_transitions_to_seed(self):
        scheduler = _scheduler('eagle3')
        request = Request(0, 'target', 10, 30, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'eagle3_prefill'
        request.chunk_len = 10
        scheduler.inflight = [_batch(request)]

        prompt, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual((prompt, generated), (10, 0))
        self.assertEqual(final, [])
        self.assertEqual(transfer, [])
        self.assertEqual(request.speculative_target_tokens, 10)
        self.assertEqual(request.speculative_target_kv_tokens, 10)
        self.assertEqual(request.speculative_stage, 'eagle3_seed')
        self.assertIn(request, scheduler.request)

    def test_eagle3_seed_batch_uses_prompt_kv_length(self):
        scheduler = _scheduler('eagle3')
        request = Request(0, 'target', 128, 200, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'eagle3_seed'
        request.speculative_draft_tokens = 4
        request.num_computed_tokens = 128
        scheduler.request = [request]

        batch = scheduler.schedule_eagle3_seed(0, 0)

        self.assertIsNotNone(batch)
        self.assertEqual(batch.speculative_stage, 'eagle3_seed')
        self.assertEqual(batch.eagle3_batch_size, 1)
        self.assertEqual(batch.eagle3_kv_len, 128)
        self.assertEqual(batch.eagle3_num_speculative_tokens, 4)
        self.assertEqual(batch.eagle3_draft_model, 'org/eagle3-head')
        self.assertEqual(batch.kv_size, 0)
        self.assertEqual(scheduler.request, [])
        self.assertIn(batch, scheduler.inflight)

    def test_eagle3_seed_commits_first_token_and_sets_ttft(self):
        scheduler = _scheduler('eagle3')
        request = Request(0, 'target', 10, 30, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'eagle3_seed'
        request.speculative_target_tokens = 10
        request.speculative_target_kv_tokens = 10
        request.speculative_draft_tokens = 4
        request.speculative_verify_tokens = 5
        request.speculative_first_commit_pending = True
        request.num_computed_tokens = 10
        scheduler.inflight = [_batch(request, 'eagle3_seed')]

        _, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 1)
        self.assertEqual(final, [])
        self.assertEqual(transfer, [])
        self.assertEqual(scheduler.memory.adjustments, [])
        self.assertEqual(request.ttft, 100)
        self.assertEqual(request.speculative_target_tokens, 11)
        self.assertEqual(request.speculative_target_kv_tokens, 10)
        self.assertEqual(request.num_computed_tokens, 10)
        self.assertFalse(request.is_init)
        self.assertEqual(request.speculative_stage, 'eagle3')
        self.assertIn(request, scheduler.request)

    def test_eagle3_iteration_commits_and_requeues(self):
        scheduler = _scheduler('eagle3')
        request = Request(0, 'target', 10, 30, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'eagle3'
        request.speculative_target_tokens = 11
        request.speculative_target_kv_tokens = 10
        request.speculative_draft_tokens = 4
        request.speculative_verify_tokens = 5
        request.speculative_first_commit_pending = False
        request.num_computed_tokens = 10
        scheduler.inflight = [
            _batch(request, 'eagle3', scheduled_tokens={request.id: 5})
        ]

        _, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 3)
        self.assertEqual(final, [])
        self.assertEqual(transfer, [])
        self.assertEqual(scheduler.memory.adjustments, [(15, 13)])
        self.assertEqual(request.speculative_target_tokens, 14)
        self.assertEqual(request.speculative_target_kv_tokens, 13)
        self.assertEqual(request.num_computed_tokens, 13)
        self.assertEqual(request.speculative_draft_tokens, 4)
        self.assertEqual(request.speculative_stage, 'eagle3')
        self.assertIn(request, scheduler.request)



if __name__ == '__main__':
    unittest.main()
