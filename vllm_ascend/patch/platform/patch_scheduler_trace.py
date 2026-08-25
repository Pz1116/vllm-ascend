# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in diagnostics for asynchronous KV scheduling."""

from contextvars import ContextVar
from functools import wraps
from typing import Any

import vllm.v1.core.sched.scheduler as scheduler_module
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
)

_active_scheduler: ContextVar[Any | None] = ContextVar(
    "ascend_scheduler_trace", default=None
)
_load_steps: dict[str, int] = {}


def _block_counts(scheduler: Any, req_id: str) -> list[int]:
    return [
        len(group)
        for group in scheduler.kv_cache_manager.get_blocks(req_id).blocks
    ]


def _log_state(
    scheduler: Any,
    phase: str,
    step: int,
    budget: int,
    scheduled: dict[str, int] | None = None,
) -> None:
    deferred_statuses: dict[str, int] = {}
    remote: list[str] = []
    for req in scheduler.skipped_waiting:
        status = req.status.name
        deferred_statuses[status] = deferred_statuses.get(status, 0) + 1
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            load_step = _load_steps.get(req.request_id)
            age = step - load_step if load_step is not None else -1
            ready = req.request_id in scheduler.finished_recving_kv_req_ids
            remote.append(
                f"{req.request_id}:ready={int(ready)}:age={age}:"
                f"tokens={req.num_computed_tokens}/{req.num_tokens}:"
                f"blocks={_block_counts(scheduler, req.request_id)}"
            )

    block_pool = scheduler.kv_cache_manager.block_pool
    logger.info(
        "[SCHED-TRACE] phase=%s step=%d budget=%d scheduled=%s "
        "running=%d waiting=%d deferred=%d deferred_statuses=%s "
        "free_blocks=%d inflight_prefills=%d reserved_blocks=%d "
        "max_running=%d remote=%s",
        phase,
        step,
        budget,
        scheduled or {},
        len(scheduler.running),
        len(scheduler.waiting),
        len(scheduler.skipped_waiting),
        deferred_statuses,
        block_pool.get_num_free_blocks(),
        len(scheduler._inflight_prefills),
        scheduler._inflight_prefill_reserved_blocks(),
        scheduler.max_num_running_reqs,
        remote,
    )


def _patch_schedule(cls: type[Any]) -> None:
    original = cls.__dict__.get("schedule")
    if original is None or getattr(original, "_ascend_scheduler_trace", False):
        return

    @wraps(original)
    def traced(self: Any, *args: Any, **kwargs: Any):
        if _active_scheduler.get() is self:
            return original(self, *args, **kwargs)

        next_step = self.current_step + 1
        initial_budget = self.max_num_scheduled_tokens
        _log_state(self, "begin", next_step, initial_budget)
        token = _active_scheduler.set(self)
        try:
            output = original(self, *args, **kwargs)
        finally:
            _active_scheduler.reset(token)
        _log_state(
            self,
            "end",
            self.current_step,
            initial_budget - output.total_num_scheduled_tokens,
            output.num_scheduled_tokens,
        )
        return output

    traced._ascend_scheduler_trace = True
    cls.schedule = traced


def _patch_method(name: str, wrapper):
    for cls in scheduler_module.Scheduler.__mro__:
        original = cls.__dict__.get(name)
        if original is not None:
            setattr(cls, name, wrapper(original))
            return


def _trace_promote(original):
    @wraps(original)
    def traced(self: Any, request):
        old_status = request.status
        ready = request.request_id in self.finished_recving_kv_req_ids
        promoted = original(self, request)
        if old_status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            load_step = _load_steps.get(request.request_id)
            logger.info(
                "[SCHED-TRACE] event=%s step=%d req=%s ready_signal=%s "
                "load_age_steps=%d status_after=%s free_blocks=%d",
                "remote_promote" if promoted else "remote_still_blocked",
                self.current_step,
                request.request_id,
                ready,
                self.current_step - load_step if load_step is not None else -1,
                request.status.name,
                self.kv_cache_manager.block_pool.get_num_free_blocks(),
            )
            if promoted:
                _load_steps.pop(request.request_id, None)
        return promoted

    return traced


def _trace_update_finished(original):
    @wraps(original)
    def traced(self: Any, kv_connector_output):
        if (
            kv_connector_output.finished_recving
            or kv_connector_output.finished_sending
            or kv_connector_output.invalid_block_ids
        ):
            logger.info(
                "[SCHED-TRACE] event=scheduler_recv_transfer_finished "
                "step=%d recv=%s send=%s invalid_blocks=%s",
                self.current_step,
                sorted(kv_connector_output.finished_recving or ()),
                sorted(kv_connector_output.finished_sending or ()),
                sorted(kv_connector_output.invalid_block_ids),
            )
        return original(self, kv_connector_output)

    return traced


_original_allocate_slots = KVCacheManager.allocate_slots


@wraps(_original_allocate_slots)
def _traced_allocate_slots(
    self: KVCacheManager,
    request,
    num_new_tokens: int,
    *args: Any,
    **kwargs: Any,
):
    free_before = self.block_pool.get_num_free_blocks()
    result = _original_allocate_slots(
        self, request, num_new_tokens, *args, **kwargs
    )
    load_async = kwargs.get("delay_cache_blocks", False)
    if load_async or result is None:
        scheduler = _active_scheduler.get()
        step = scheduler.current_step if scheduler is not None else -1
        if result is None:
            event = (
                "running_allocate_fail"
                if request.status == RequestStatus.RUNNING
                else "waiting_allocate_fail"
            )
        else:
            event = "async_load_admit"
        logger.info(
            "[SCHED-TRACE] event=%s step=%d req=%s status=%s "
            "new_tokens=%d external_tokens=%d free_before=%d free_after=%d "
            "reserved_blocks=%d allocated_blocks=%s",
            event,
            step,
            request.request_id,
            request.status.name,
            num_new_tokens,
            kwargs.get("num_external_computed_tokens", 0),
            free_before,
            self.block_pool.get_num_free_blocks(),
            kwargs.get("reserved_blocks", 0),
            None if result is None else [len(group) for group in result.blocks],
        )
        if event == "async_load_admit":
            _load_steps[request.request_id] = step
    return result


_original_set_finished = KVTransferThread.set_finished_request


@wraps(_original_set_finished)
def _traced_set_finished(self: KVTransferThread, req_id: str):
    result = _original_set_finished(self, req_id)
    logger.info(
        "[SCHED-TRACE] event=worker_transfer_finished "
        "thread=%s tp_rank=%d req=%s",
        self.name,
        self.tp_rank,
        req_id,
    )
    return result


for scheduler_cls in scheduler_module.Scheduler.__mro__:
    _patch_schedule(scheduler_cls)
_patch_method("_try_promote_blocked_waiting_request", _trace_promote)
_patch_method("_update_from_kv_xfer_finished", _trace_update_finished)
KVCacheManager.allocate_slots = _traced_allocate_slots
KVTransferThread.set_finished_request = _traced_set_finished
