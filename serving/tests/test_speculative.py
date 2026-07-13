import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from serving.core.router import Router
from serving.core.speculative import SpeculativeAcceptanceModel


class _FakeScheduler:
    def __init__(self, instance_id, role, model):
        self.instance_id = instance_id
        self.speculative_decoding = True
        self.speculative_role = role
        self.pd_type = None
        self.model = model
        self.max_num_seqs = 16
        self.enable_prefix_caching = False
        self.request = []
        self.inflight = []

    def add_request(self, request, **kwargs):
        self.request.append(request)

    def add_decode(self, request):
        self.request.append(request)


class SpeculativeAcceptanceModelTest(unittest.TestCase):
    def test_constant_model(self):
        model = SpeculativeAcceptanceModel('constant', acceptance_rate=0.5)
        self.assertEqual(model.sample(draft_tokens=4), 2)

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            SpeculativeAcceptanceModel('constant', acceptance_rate=1.1)

    def test_missing_trace_entry_is_rejected(self):
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as trace:
            json.dump({'request_id': 3, 'iteration': 0, 'accepted_tokens': 2}, trace)
            trace.write('\n')
            path = trace.name
        try:
            model = SpeculativeAcceptanceModel('trace', trace_path=path)
            self.assertEqual(model.sample(request=3, iteration=0, draft_tokens=4), 2)
            with self.assertRaises(KeyError):
                model.sample(request=4, iteration=0, draft_tokens=4)
        finally:
            os.unlink(path)


class SpeculativeRoleInferenceTest(unittest.TestCase):
    def test_two_pd_null_instances_are_draft_then_target(self):
        source = Path('serving/__main__.py').read_text(encoding='utf-8')
        module = ast.parse(source)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == '_resolve_speculative_roles')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<roles>', 'exec'), namespace)
        instances = [{'pd_type': None}, {'pd_type': None}]
        configs = [
            {'speculative_decoding': True, 'speculative_role': None},
            {'speculative_decoding': True, 'speculative_role': None},
        ]
        namespace['_resolve_speculative_roles'](instances, configs)
        self.assertEqual(
            [config['speculative_role'] for config in configs],
            ['draft', 'target'],
        )


class SpeculativeRouterTest(unittest.TestCase):
    def setUp(self):
        self.draft = _FakeScheduler(0, 'draft', 'draft-model')
        self.target = _FakeScheduler(1, 'target', 'target-model')
        self.router = Router(2, [self.draft, self.target], req_num=0)

    def test_pd_null_request_starts_on_draft_role(self):
        selected = self.router._select_scheduler_for_request({
            'index': 0,
            'route_to': 'prefill',
            'model_name': 'target-model',
        })
        self.assertIs(selected, self.draft)

    def test_target_prefill_routes_by_speculative_role(self):
        request = type('RequestStub', (), {
            'id': 7,
            'speculative_stage': 'target_prefill',
            'speculative_target_instance_id': None,
            'speculative_draft_instance_id': 0,
        })()
        self.router.transfer_speculative_requests([request])
        self.assertEqual(request.speculative_target_instance_id, 1)
        self.assertIn(request, self.target.request)

    def test_draft_sync_returns_to_pinned_draft(self):
        request = type('RequestStub', (), {
            'id': 8,
            'speculative_stage': 'draft_sync',
            'speculative_target_instance_id': 1,
            'speculative_draft_instance_id': 0,
        })()
        self.router.transfer_speculative_requests([request])
        self.assertIn(request, self.draft.request)


if __name__ == '__main__':
    unittest.main()
