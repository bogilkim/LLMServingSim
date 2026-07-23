import bisect
import json
import random
from .logger import get_logger


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.speculative_draft_schedulers = [
            s for s in schedulers
            if s.speculative_decoding
            and s.speculative_role in ('draft', 'colocated')
        ]
        self.speculative_target_schedulers = [
            s for s in schedulers
            if s.speculative_decoding
            and s.speculative_role in ('target', 'colocated')
        ]
        self.eagle3_schedulers = [
            s for s in schedulers if s.speculative_decoding and s.speculative_role == 'eagle3']
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.request)
            running = sum(len(b.requests) for b in sched.inflight)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, role):
        raise NotImplementedError("Implement custom routing policy.")

    def _normalize_route_to(self, route_to):
        normalized = 'prefill' if route_to is None else str(route_to).lower()
        if normalized not in ('prefill', 'decode'):
            raise ValueError(
                f"Invalid route_to '{route_to}'. Supported values: prefill, decode"
            )
        return normalized

    def _make_request_data(self, source, req_id, arrival_time_ns, enable_prefix_caching,
                           defaults=None, session_id=None, sub_request_index=None):
        defaults = defaults or {}
        req_data = {
            'index': req_id,
            'input_toks': int(source['input_toks']),
            'output_toks': int(source['input_toks'] + source['output_toks']),
            'arrival_time_ns': int(arrival_time_ns),
        }

        model_name = source.get('model_name', defaults.get('model_name'))
        if model_name is not None:
            req_data['model_name'] = model_name

        route_to = source.get('route_to', defaults.get('route_to'))
        req_data['route_to'] = self._normalize_route_to(route_to)

        instance_id = source.get('instance_id', defaults.get('instance_id'))
        if instance_id is not None:
            req_data['instance_id'] = int(instance_id)

        if session_id is not None:
            req_data['session_id'] = session_id
        if sub_request_index is not None:
            req_data['sub_request_index'] = sub_request_index

        if enable_prefix_caching:
            req_data['input_hash_ids'] = source.get('input_tok_ids', [])
            req_data['output_hash_ids'] = source.get('output_tok_ids', [])
        return req_data

    def _select_scheduler_for_request(self, req_data):
        route_to = self._normalize_route_to(req_data.get('route_to'))
        if self.eagle3_schedulers:
            candidates = self.eagle3_schedulers
            instance_id = req_data.get('instance_id')
            model_name = req_data.get('model_name')
            if instance_id is not None:
                idx = int(instance_id)
                if idx < 0 or idx >= len(self.schedulers):
                    raise IndexError(
                        f"Requested instance_id {idx} for request "
                        f"{req_data.get('index')} is out of range"
                    )
                scheduler = self.schedulers[idx]
                if scheduler not in candidates:
                    raise ValueError(
                        f"Requested instance_id {idx} is not an EAGLE3 instance"
                    )
                candidates = [scheduler]
            if model_name is not None:
                candidates = [s for s in candidates if s.model == model_name]
            if not candidates:
                raise LookupError(f"No EAGLE3 scheduler matches model {model_name!r}")
            return candidates[self._select_instance(candidates, 'prefill')]
        if self.speculative_draft_schedulers and self.speculative_target_schedulers:
            # The workload's model_name describes the served target model;
            # speculative admission always begins with draft prefill.
            candidates = self.speculative_draft_schedulers
            if all(s.speculative_role == 'colocated' for s in candidates):
                instance_id = req_data.get('instance_id')
                model_name = req_data.get('model_name')
                if instance_id is not None:
                    idx = int(instance_id)
                    if idx < 0 or idx >= len(self.schedulers):
                        raise IndexError(
                            f"Requested instance_id {idx} for request "
                            f"{req_data.get('index')} is out of range"
                        )
                    scheduler = self.schedulers[idx]
                    if scheduler not in candidates:
                        raise ValueError(
                            f"Requested instance_id {idx} is not a colocated "
                            "speculative instance"
                        )
                    candidates = [scheduler]
                if model_name is not None:
                    candidates = [
                        scheduler for scheduler in candidates
                        if scheduler.model == model_name
                    ]
                if not candidates:
                    raise LookupError(
                        "No colocated speculative scheduler matches target "
                        f"model {model_name!r}"
                    )
            return candidates[self._select_instance(candidates, 'prefill')]
        candidates = self.prefill_schedulers if route_to == 'prefill' else self.decode_schedulers
        if not candidates:
            raise LookupError(
                f"No {route_to} schedulers are available for request {req_data.get('index')}"
            )

        instance_id = req_data.get('instance_id')
        model_name = req_data.get('model_name')

        if instance_id is not None:
            idx = int(instance_id)
            if idx < 0 or idx >= len(self.schedulers):
                raise IndexError(
                    f"Requested instance_id {idx} for request {req_data.get('index')} is out of range"
                )
            sched = self.schedulers[idx]
            if sched not in candidates:
                raise ValueError(
                    f"Requested instance_id {idx} for request {req_data.get('index')} does not belong to the {route_to} pool"
                )
            if model_name is not None and sched.model != model_name:
                raise ValueError(
                    f"Requested instance_id {idx} for request {req_data.get('index')} runs model {sched.model!r}, "
                    f"but the workload requested {model_name!r}"
                )
            return sched

        if model_name is not None:
            candidates = [sched for sched in candidates if sched.model == model_name]
            if not candidates:
                raise LookupError(
                    f"No {route_to} scheduler matches model {model_name!r} for request {req_data.get('index')}"
                )

        instance_idx = self._select_instance(candidates, route_to)
        return candidates[instance_idx]

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        Requests and sub-requests may optionally carry routing metadata:
        ``model_name`` to target a specific scheduler model, ``route_to``
        to pick the prefill/decode instance pool, and ``instance_id`` to
        pin a specific scheduler. This is what lets speculative-decoding
        traces alternate draft/verifier model calls.

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = self._make_request_data(row, req_id, row['arrival_time_ns'], enable_prefix_caching)
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])
        session_defaults = {
            'model_name': row.get('model_name'),
            'route_to': row.get('route_to'),
            'instance_id': row.get('instance_id'),
        }

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
            'defaults': session_defaults,
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = self._make_request_data(
            first, base_id, arrival_ns, enable_prefix_caching,
            defaults=session_defaults, session_id=session_id, sub_request_index=0,
        )
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            sched = self._select_scheduler_for_request(req_data)

            if sched.enable_prefix_caching:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                    req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                ], is_init=self._is_init,
                   session_id=req_data.get('session_id'),
                   sub_request_index=req_data.get('sub_request_index'))
            else:
                sched.add_request([
                    req_data['index'], sched.model,
                    req_data['input_toks'], req_data['output_toks'],
                    req_data['arrival_time_ns'], sched.instance_id,
                ], is_init=self._is_init,
                   session_id=req_data.get('session_id'),
                   sub_request_index=req_data.get('sub_request_index'))

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def request_status_counts(self, current_time_ns):
        """Return counts for requests that have not reached a scheduler yet."""
        pending = self._pending_requests[self._pending_idx:]
        awaiting_arrival = sum(
            req['arrival_time_ns'] > current_time_ns for req in pending
        )
        awaiting_routing = len(pending) - awaiting_arrival
        dependency_blocked = sum(
            len(session['sub_requests']) - session['next_index']
            for session in self._deferred_sessions.values()
        )
        return {
            "awaiting_arrival": awaiting_arrival,
            "awaiting_routing": awaiting_routing,
            "dependency_blocked": dependency_blocked,
            "total": self._next_request_id,
        }

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']
        session_defaults = session.get('defaults', {})

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = self._make_request_data(
                next_sub, next_id, release_time_ns, self._enable_prefix_caching,
                defaults=session_defaults, session_id=session_id, sub_request_index=next_idx,
            )
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete - all sub-requests have been released
            del self._deferred_sessions[session_id]

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self._select_instance(self.decode_schedulers, "decode")
            self.decode_schedulers[instance_id].add_decode(req)

    def transfer_speculative_requests(self, requests):
        for req in requests:
            stage = getattr(req, 'speculative_stage', None)
            if stage in ('verify', 'target_prefill'):
                candidates = self.speculative_target_schedulers
                pinned = getattr(req, 'speculative_target_instance_id', None)
                role = 'target'
            elif stage in ('draft_prefill', 'draft', 'draft_sync'):
                candidates = self.speculative_draft_schedulers
                pinned = getattr(req, 'speculative_draft_instance_id', None)
                role = 'draft'
            else:
                self.transfer_prefill_request([req])
                continue

            if not candidates:
                raise LookupError(f'No speculative {role} scheduler is configured.')
            if pinned is not None:
                pinned = int(pinned)
                if pinned < 0 or pinned >= len(self.schedulers):
                    raise IndexError(
                        f'Request {req.id} requested speculative instance {pinned}, '
                        'but that instance id is out of range.'
                    )
                sched = self.schedulers[pinned]
                if sched not in candidates:
                    raise ValueError(
                        f'Request {req.id} requested speculative instance {pinned}, '
                        f'but it is not a {role} scheduler.'
                    )
            else:
                sched = candidates[self._select_instance(candidates, role)]

            if stage in ('verify', 'target_prefill'):
                req.speculative_target_instance_id = sched.instance_id
            else:
                req.speculative_draft_instance_id = sched.instance_id
            sched.add_decode(req)
