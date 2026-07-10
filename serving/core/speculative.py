import csv
import json
import random
from collections import defaultdict


class SpeculativeAcceptanceModel:
    def __init__(self, mode='constant', acceptance_rate=0.8, trace_path=None, seed=42):
        self.mode = (mode or 'constant').lower()
        self.acceptance_rate = float(acceptance_rate)
        self.trace_path = trace_path
        self._rnd = random.Random(seed)
        self._trace = defaultdict(list)
        self._trace_pos = defaultdict(int)

        if self.mode not in ('constant', 'random', 'trace'):
            raise ValueError(
                f"Unsupported speculative acceptance model {mode!r}. "
                "Supported: constant, random, trace"
            )
        if self.mode == 'trace':
            if not trace_path:
                raise ValueError('speculative acceptance trace mode requires trace_path')
            self._load_trace(trace_path)

    def _load_trace(self, path):
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._store_trace_row(row)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._store_trace_row(row)

    def _store_trace_row(self, row):
        request_id = row.get('request_id', row.get('id', row.get('request', None)))
        iteration = row.get('iteration', row.get('step', 0))
        accepted = row.get('accepted_tokens', row.get('accepted', row.get('accept', None)))
        if accepted is None:
            raise ValueError(
                'Trace rows must define accepted_tokens/accepted/accept.'
            )
        key = (None if request_id is None else int(request_id), int(iteration))
        self._trace[key].append(int(accepted))

    def _trace_sample(self, request_id, iteration, draft_tokens):
        key = (None if request_id is None else int(request_id), int(iteration))
        values = self._trace.get(key)
        if not values:
            values = self._trace.get((None, int(iteration)), [])
        if not values:
            return min(draft_tokens, 0)
        idx = self._trace_pos[key]
        self._trace_pos[key] = idx + 1
        if idx >= len(values):
            idx = len(values) - 1
        return min(draft_tokens, max(0, int(values[idx])))

    def sample(self, request=None, iteration=0, draft_tokens=1):
        draft_tokens = max(int(draft_tokens), 0)
        if draft_tokens == 0:
            return 0

        request_id = getattr(request, 'id', request)
        if self.mode == 'constant':
            accepted = int(round(self.acceptance_rate * draft_tokens))
            return max(0, min(draft_tokens, accepted))

        if self.mode == 'random':
            accepted = 0
            for _ in range(draft_tokens):
                if self._rnd.random() < self.acceptance_rate:
                    accepted += 1
                else:
                    break
            return accepted

        return self._trace_sample(request_id, iteration, draft_tokens)
