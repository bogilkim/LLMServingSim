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
from serving.core.memory_model import MemoryModel
from serving.core.speculative import SpeculativeAcceptanceModel


class _Memory:
    def __init__(self):
        self.adjustments = []
        self.adjustment_models = []
        self.block_models = []
        self.allocations = []
        self.frees = []

    def adjust_tokens(
            self, current_tokens, new_tokens, device, model=None):
        self.adjustments.append((current_tokens, new_tokens))
        self.adjustment_models.append(model)

    def get_block_kv(
            self, batch_req, batch_len, scheduled_tokens=None, model=None):
        self.block_models.append(model)
        return 0

    def get_evict_kv(self, request, model=None):
        return 0

    def is_avail(self, size, device):
        return True

    def allocate(self, size, device):
        self.allocations.append(size)

    def free(self, size, device):
        self.frees.append(size)

    def get_total_kv_tokens(self, tokens, model=None):
        multiplier = 2 if model == 'org/eagle3-head' else 10
        return int(tokens) * multiplier


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
    scheduler.instance_id = 0
    scheduler.max_num_seqs = 8
    scheduler.pp_size = 1
    scheduler.model = 'target'
    scheduler.speculative_draft_model = 'org/eagle3-head'
    scheduler.inflight = []
    scheduler.batch_ids = -1
    scheduler._last_colocated_model = None
    scheduler._last_colocated_fused = False
    scheduler.num_npus = 1
    scheduler.pd_type = None
    scheduler.enable_prefix_caching = False
    scheduler.prefix_storage = None
    scheduler.prioritize_prefill = False
    scheduler.enable_chunked_prefill = False
    scheduler.long_prefill_token_threshold = 0
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
    def test_colocated_schedules_draft_and_verify_as_one_batch(self):
        scheduler = _scheduler('colocated')
        scheduler.max_num_batched_tokens = 64
        requests = []
        for request_id in range(2):
            request = Request(request_id, 'target', 10, 30, 0, 0)
            request.speculative_active = True
            request.speculative_stage = 'draft_sync'
            request.speculative_target_tokens = 11
            request.speculative_target_kv_tokens = 10
            request.speculative_draft_kv_tokens = 10
            request.speculative_draft_tokens = 4
            request.speculative_draft_generated = 0
            request.num_computed_tokens = 10
            requests.append(request)
        scheduler.request = requests.copy()

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.speculative_stage, 'draft_verify')
        self.assertEqual(batch.model, 'target')
        self.assertEqual(batch.speculative_draft_model, 'org/eagle3-head')
        self.assertEqual(batch.speculative_draft_steps, 4)
        self.assertEqual(batch.scheduled_tokens, {0: 5, 1: 5})
        self.assertEqual(batch.requests, requests)
        self.assertEqual(scheduler.request, [])
        self.assertEqual(scheduler.memory.allocations, [116])

    def test_colocated_fused_completion_applies_acceptance_once(self):
        scheduler = _scheduler('colocated')
        request = Request(0, 'target', 10, 30, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_target_tokens = 11
        request.speculative_target_kv_tokens = 10
        request.speculative_draft_kv_tokens = 10
        request.speculative_draft_tokens = 4
        request.speculative_draft_generated = 0
        request.speculative_first_commit_pending = True
        request.num_computed_tokens = 10

        batch = _batch(request, 'draft_verify', {request.id: 5})
        batch.speculative_target_kv_after = {request.id: 15}
        batch.speculative_draft_kv_after = {request.id: 14}
        scheduler.inflight = [batch]

        _, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 4)
        self.assertEqual(final, [])
        self.assertEqual(transfer, [])
        self.assertEqual(request.speculative_target_tokens, 14)
        self.assertEqual(request.speculative_target_kv_tokens, 13)
        self.assertEqual(request.speculative_draft_kv_tokens, 13)
        self.assertEqual(request.speculative_stage, 'draft_sync')
        self.assertEqual(request.speculative_iteration, 1)
        self.assertEqual(request.ttft, 100)
        self.assertEqual(request.itl, [0, 0, 0])
        self.assertEqual(
            scheduler.memory.adjustments,
            [(15, 13), (14, 13)],
        )
        self.assertEqual(
            scheduler.memory.adjustment_models,
            ['target', 'org/eagle3-head'],
        )

    def test_colocated_fused_completion_clears_released_draft_kv(self):
        scheduler = _scheduler('colocated')
        request = Request(0, 'target', 10, 14, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_target_tokens = 11
        request.speculative_target_kv_tokens = 10
        request.speculative_draft_kv_tokens = 10
        request.speculative_draft_tokens = 4
        request.speculative_draft_generated = 0
        request.num_computed_tokens = 10

        batch = _batch(request, 'draft_verify', {request.id: 5})
        batch.speculative_target_kv_after = {request.id: 15}
        batch.speculative_draft_kv_after = {request.id: 14}
        scheduler.inflight = [batch]

        _, _, final, _ = scheduler.add_done(1, 0, 100)

        self.assertEqual(final, [request])
        self.assertEqual(request.speculative_target_kv_tokens, 0)
        self.assertEqual(request.speculative_draft_kv_tokens, 0)
        self.assertEqual(len(scheduler.memory.adjustments), 2)

        scheduler.release_speculative_draft(request)

        self.assertEqual(len(scheduler.memory.adjustments), 2)

    def test_colocated_memory_accounts_for_both_models(self):
        target = 'Qwen/Qwen3-32B'
        draft = 'meta-llama/Llama-3.1-8B'
        memory = MemoryModel(
            target, 0, 0, 1, 1, 96, 512, 16, 16,
            False, False, None, None,
            additional_models=[draft],
        )

        self.assertEqual(
            memory.weight,
            memory.weight_by_model[target] + memory.weight_by_model[draft],
        )
        self.assertEqual(memory.npu_used, memory.weight)
        self.assertNotEqual(
            memory.get_kv(16),
            memory.get_kv(16, model=draft),
        )

    def test_colocated_batch_uses_draft_model_without_mixing_target(self):
        scheduler = _scheduler('colocated')
        draft = Request(0, 'target', 10, 20, 0, 0)
        draft.speculative_active = True
        draft.speculative_stage = 'draft'
        draft.speculative_target_tokens = 10
        draft.speculative_draft_kv_tokens = 10
        draft.num_computed_tokens = 10

        target = Request(1, 'target', 10, 20, 0, 0)
        target.speculative_active = True
        target.speculative_stage = 'target_prefill'
        scheduler.request = [draft, target]

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.model, 'org/eagle3-head')
        self.assertEqual(batch.speculative_stage, 'draft')
        self.assertEqual(batch.requests, [draft])
        self.assertEqual(scheduler.request, [target])
        self.assertEqual(
            set(scheduler.memory.block_models),
            {'org/eagle3-head'},
        )

    def test_colocated_target_prefill_does_not_starve_behind_older_draft(self):
        scheduler = _scheduler('colocated')
        scheduler.max_num_batched_tokens = 2048
        scheduler.max_num_seqs = 128
        draft = Request(0, 'target', 10, 20, 0, 0)
        draft.speculative_active = True
        draft.speculative_stage = 'draft'
        draft.speculative_target_tokens = 10
        draft.speculative_draft_kv_tokens = 10
        draft.num_computed_tokens = 10

        targets = [
            Request(i, 'target', 10, 20, 0, 0)
            for i in range(1, 65)
        ]
        for target in targets:
            target.speculative_active = True
            target.speculative_stage = 'target_prefill'
        scheduler.request = [draft] + targets
        scheduler._last_colocated_model = 'org/eagle3-head'

        batch = scheduler.schedule_base(0, 0)

        self.assertEqual(batch.model, 'target')
        self.assertEqual(batch.speculative_stage, 'target_prefill')
        self.assertEqual(batch.requests, targets)
        self.assertEqual(scheduler.request, [draft])
        self.assertEqual(scheduler._last_colocated_model, 'target')

    def test_colocated_transfer_rolls_back_draft_kv_with_draft_shape(self):
        scheduler = _scheduler('colocated')
        request = Request(0, 'target', 10, 20, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_draft_kv_tokens = 14
        request.num_computed_tokens = 12

        scheduler.add_decode(request)

        self.assertEqual(scheduler.memory.adjustments, [(14, 12)])
        self.assertEqual(
            scheduler.memory.adjustment_models,
            ['org/eagle3-head'],
        )
        self.assertIn(request, scheduler.request)

    def test_colocated_prefills_both_models_before_first_token(self):
        scheduler = _scheduler('colocated')
        request = Request(0, 'target', 10, 20, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_prefill'
        request.chunk_len = 10
        scheduler.inflight = [_batch(request, 'draft')]

        prompt, generated, final, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual((prompt, generated), (0, 0))
        self.assertEqual(final, [])
        self.assertEqual(transfer, [request])
        self.assertEqual(request.speculative_stage, 'target_prefill')
        self.assertEqual(request.speculative_draft_kv_tokens, 10)
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(request.ttft, -1)

        scheduler.add_decode(request)
        batch = scheduler.schedule_base(100, 0)

        self.assertEqual(batch.model, 'target')
        self.assertEqual(batch.speculative_stage, 'target_prefill')

        prompt, generated, final, transfer = scheduler.add_done(
            batch.batch_id + 1, 0, 200)

        self.assertEqual((prompt, generated), (10, 0))
        self.assertEqual(final, [])
        self.assertEqual(transfer, [request])
        self.assertEqual(request.ttft, -1)
        self.assertEqual(request.speculative_stage, 'draft_sync')
        self.assertEqual(request.speculative_target_tokens, 11)
        self.assertEqual(request.speculative_target_kv_tokens, 10)
        self.assertTrue(request.speculative_first_commit_pending)

    def test_draft_prefill_is_not_counted_as_prompt_throughput(self):
        scheduler = _scheduler('draft')
        request = Request(0, 'draft', 10, 20, 0, 0)
        request.speculative_stage = 'draft_prefill'
        request.chunk_len = 5
        scheduler.inflight = [_batch(request, 'draft')]

        prompt, _, _, _ = scheduler.add_done(1, 0, 100)

        self.assertEqual(prompt, 0)

    def test_target_prefill_waits_for_initial_draft_proposal(self):
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
        self.assertEqual(request.ttft, -1)
        self.assertEqual(request.speculative_stage, 'draft_sync')
        self.assertEqual(request.speculative_target_tokens, 11)
        self.assertEqual(request.speculative_target_kv_tokens, 10)
        self.assertEqual(request.num_computed_tokens, 10)
        self.assertTrue(request.speculative_first_commit_pending)

    def test_verify_rolls_back_then_requests_draft_sync(self):
        scheduler = _scheduler('target')
        request = Request(0, 'target', 10, 20, 0, 1)
        request.speculative_active = True
        request.speculative_stage = 'verify'
        request.speculative_target_tokens = 11
        request.speculative_target_kv_tokens = 10
        request.speculative_draft_tokens = 4
        request.speculative_verify_tokens = 5
        request.num_computed_tokens = 10
        request.set_ttft(20)
        scheduler.inflight = [_batch(request, 'verify')]

        _, generated, _, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 3)
        self.assertEqual(scheduler.memory.adjustments, [(15, 13)])
        self.assertEqual(transfer, [request])
        self.assertEqual(request.speculative_stage, 'draft_sync')
        self.assertEqual(request.speculative_target_tokens, 14)
        self.assertEqual(request.speculative_target_kv_tokens, 13)
        self.assertEqual(request.num_computed_tokens, 13)
        self.assertEqual(request.itl, [80, 0, 0])

    def test_draft_sync_executes_target_token(self):
        scheduler = _scheduler('draft')
        request = Request(0, 'draft', 10, 20, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_draft_kv_tokens = 12
        request.speculative_target_kv_tokens = 12
        request.speculative_draft_tokens = 4
        request.speculative_draft_generated = 0
        request.num_computed_tokens = 12
        scheduler.inflight = [_batch(request, 'draft')]

        _, generated, _, transfer = scheduler.add_done(1, 0, 100)

        self.assertEqual(generated, 0)
        self.assertEqual(transfer, [])
        self.assertEqual(request.speculative_stage, 'draft')
        self.assertEqual(request.speculative_draft_kv_tokens, 13)
        self.assertEqual(request.speculative_draft_generated, 1)

    def test_initial_draft_proposal_sets_ttft_before_verify(self):
        scheduler = _scheduler('draft')
        request = Request(0, 'draft', 10, 20, 0, 0)
        request.speculative_active = True
        request.speculative_stage = 'draft_sync'
        request.speculative_draft_kv_tokens = 10
        request.speculative_target_kv_tokens = 10
        request.speculative_target_tokens = 11
        request.speculative_draft_tokens = 1
        request.speculative_first_commit_pending = True
        request.num_computed_tokens = 10
        scheduler.inflight = [_batch(request, 'draft')]

        _, generated, _, transfer = scheduler.add_done(1, 0, 150)

        self.assertEqual(generated, 1)
        self.assertEqual(transfer, [request])
        self.assertEqual(request.ttft, 150)
        self.assertFalse(request.is_init)
        self.assertFalse(request.speculative_first_commit_pending)
        self.assertEqual(request.speculative_stage, 'verify')
        self.assertEqual(request.num_computed_tokens, 10)

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
