import importlib
import logging
import math
import threading
from collections.abc import Generator

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    CacheGroupLayout,
    CacheLayout,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    get_cache_family_granularity,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)

backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.model_type = getattr(hf_text_config, "model_type", None) or getattr(hf_config, "model_type", None)
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.use_layerwise = use_layerwize
        self.kv_cache_config = kv_cache_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
        self.block_size = self.grouped_block_size[0]
        self.hash_block_size = vllm_config.cache_config.block_size * cp_scale
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        self.current_layer = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        if self.use_mla or self.use_sparse:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        self.metadata = KeyMetadata(
            model_config.model.rstrip("/").split("/")[-1],
            self.head_or_tp_rank,
            self.pcp_rank,
            self.dcp_rank,
            self.pp_rank,
        )
        self.num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups) if self.kv_cache_config is not None else 1
        self.kv_cache_group_families = self._infer_group_families()
        self.state_cache_group_families = self.kv_cache_group_families.copy() if self._requires_state_groups() else []
        self.group_uses_align_state = [False] * self.num_kv_cache_groups
        self.state_group_uses_align_state: list[bool] = []
        self.state_group_ids: list[int] = []
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        if self.kv_cache_config is not None:
            for group_id, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
                kv_cache_spec = kv_cache_group.kv_cache_spec
                if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                    kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
                if isinstance(kv_cache_spec, MambaSpec) and kv_cache_spec.mamba_cache_mode != "align":
                    raise NotImplementedError(
                        "AscendStore hybrid linear-attention support currently requires mamba_cache_mode='align'."
                    )
                self.group_uses_align_state[group_id] = (
                    isinstance(kv_cache_spec, MambaSpec) and kv_cache_spec.mamba_cache_mode == "align"
                )
        if self.use_layerwise and self.num_kv_cache_groups > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")

        partitions = None
        if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "prefill_pp_layer_partition", None
            )
            prefill_pp_size = int(vllm_config.kv_transfer_config.kv_connector_extra_config.get("prefill_pp_size", 1))

            if partition_list_str is not None:
                try:
                    partitions = [int(layer) for layer in partition_list_str.split(",")]
                except ValueError as err:
                    raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
                if len(partitions) != prefill_pp_size:
                    raise ValueError(f"{len(partitions)=} does not match {prefill_pp_size=}.")
                if sum(partitions) != num_hidden_layers:
                    raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
            else:
                layers_per_partition = num_hidden_layers // prefill_pp_size
                partitions = [layers_per_partition for _ in range(prefill_pp_size)]

                if remaining_layers := num_hidden_layers % prefill_pp_size:
                    for i in range(2, remaining_layers + 2):
                        partitions[-i] += 1

        self.token_database = ChunkedTokenDatabase(
            self.metadata,
            self.grouped_block_size,
            partitions,
            self.use_hybrid,
            self.hash_block_size,
        )

        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config
        )
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

    def _infer_group_families(self) -> list[str]:
        kv_cache_groups = self.kv_cache_config.kv_cache_groups if self.kv_cache_config is not None else None
        return infer_group_cache_families(kv_cache_groups, self.compress_ratios)

    def _requires_state_groups(self) -> bool:
        return self.model_type == "deepseek_v4" and self.use_compress

    @staticmethod
    def _uses_hybrid_kv_cache(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> bool:
        if kv_cache_config is None:
            return False
        if getattr(vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", False):
            return False
        return len(kv_cache_config.kv_cache_groups) > 1 and any(
            not isinstance(group.kv_cache_spec, FullAttentionSpec) for group in kv_cache_config.kv_cache_groups
        )

    def _infer_group_block_sizes(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        if kv_cache_config is None or not self.use_hybrid:
            return [vllm_config.cache_config.block_size]

        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    def _get_group_block_size(self, group_id: int) -> int:
        if group_id >= len(self.grouped_block_size):
            return self.grouped_block_size[0]
        return self.grouped_block_size[group_id]

    @staticmethod
    def _get_group_family(families: list[str], group_id: int) -> str:
        if group_id >= len(families):
            return "default"
        return families[group_id]

    def _infer_cache_transfer_granularity(self) -> int:
        granularities = [self.lcm_block_size]
        for group_id in range(self.num_kv_cache_groups):
            granularities.append(
                get_cache_family_granularity(
                    self._get_group_block_size(group_id),
                    self._get_group_family(self.kv_cache_group_families, group_id),
                )
            )
        for group_id in self.state_group_ids:
            granularities.append(
                get_cache_family_granularity(
                    self._get_group_block_size(group_id),
                    self._get_group_family(self.state_cache_group_families, group_id),
                )
            )
        return math.lcm(*granularities)

    def _build_cache_layout(
        self,
        kv_states: dict[str, torch.Tensor] | None,
        kv_layout: CacheLayout | None,
    ) -> CacheLayout:
        if kv_layout is not None:
            return kv_layout

        if self.kv_cache_config is None:
            kv_groups = [CacheGroupLayout(group_id=0, layer_names=[])]
        else:
            group_families = self._infer_group_families()
            kv_groups = [
                CacheGroupLayout(
                    group_id=group_id,
                    layer_names=list(group_spec.layer_names),
                    cache_role="kv",
                    cache_family=group_families[group_id],
                    skip_null_blocks=self.group_uses_align_state[group_id],
                )
                for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups)
            ]

        state_groups: list[CacheGroupLayout] = []
        if kv_states:
            if kv_groups:
                for group in kv_groups:
                    state_groups.append(
                        CacheGroupLayout(
                            group_id=group.group_id,
                            layer_names=list(group.layer_names),
                            cache_role="state",
                            cache_family=group.cache_family,
                            skip_null_blocks=group.skip_null_blocks,
                        )
                    )
            else:
                state_groups = [CacheGroupLayout(group_id=0, layer_names=list(kv_states), cache_role="state")]
        return CacheLayout(kv_groups=kv_groups, state_groups=state_groups)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_states: dict[str, torch.Tensor] | None = None,
        kv_layout: CacheLayout | None = None,
    ):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        if not isinstance(first_kv_cache_tuple, (list, tuple)):
            first_kv_cache_tuple = (first_kv_cache_tuple,)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)
        block_rank = 3
        self.block_len = []
        self.block_size_scale = []
        if self.use_mla or self.use_sparse:
            for i in range(len(first_kv_cache_tuple)):
                block_shape = first_kv_cache_tuple[i].shape[-block_rank:]
                logger.info("block_shape: %s", block_shape)
                self.block_len.append(first_kv_cache_tuple[i].element_size() * math.prod(block_shape))
                tensor_num_blocks = first_kv_cache_tuple[i].shape[0]
                assert tensor_num_blocks % self.num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                self.block_size_scale.append(tensor_num_blocks // self.num_blocks)
        else:
            # [num_block, block_size, num_head, hidden_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            logger.info("block_shape: %s", block_shape)
            self.block_len = [first_kv_cache.element_size() * math.prod(block_shape)]
            assert first_kv_cache.shape[0] % self.num_blocks == 0, (
                "The external block size must be an integer multiple of the kernel block size."
            )
            self.block_size_scale = [first_kv_cache.shape[0] // self.num_blocks]

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        self.kv_caches = kv_caches
        self.kv_states = kv_states or {}
        self.cache_layout = self._build_cache_layout(self.kv_states, kv_layout)
        self.kv_caches_base_addr = []
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_kv_block_size_scale: dict[int, list[int]] = {}
        self.group_kv_cache_families: dict[int, str] = {
            group.group_id: group.cache_family for group in self.cache_layout.kv_groups
        }
        self.group_state_caches_base_addr: dict[int, list[int]] = {}
        self.group_state_block_len: dict[int, list[int]] = {}
        self.group_state_block_size_scale: dict[int, list[int]] = {}
        self.group_state_cache_families: dict[int, str] = {
            group.group_id: group.cache_family for group in self.cache_layout.state_groups
        }
        self.state_group_ids = [group.group_id for group in self.cache_layout.state_groups]
        self.state_group_uses_align_state = [False] * (max(self.state_group_ids) + 1) if self.state_group_ids else []
        ptrs = []
        lengths = []
        seen_ptrs: set[int] = set()
        for layer_name, cache_or_caches in kv_caches.items():
            if not isinstance(cache_or_caches, (list, tuple)):
                cache_or_caches = (cache_or_caches,)
            # Normalize to always be a list of caches
            for cache in cache_or_caches:
                base_addr = cache.data_ptr()
                block_len = cache[0].numel() * cache.element_size()
                tensor_num_blocks = cache.shape[0]
                assert tensor_num_blocks % self.num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_size_scale = tensor_num_blocks // self.num_blocks
                region_len = tensor_num_blocks * block_len
                self.kv_caches_base_addr.append(base_addr)
                if base_addr not in seen_ptrs:
                    ptrs.append(base_addr)
                    lengths.append(region_len)
                    seen_ptrs.add(base_addr)

        if self.cache_layout.kv_groups:
            for group_spec in self.cache_layout.kv_groups:
                group_id = group_spec.group_id
                group_addrs: list[int] = []
                group_block_lens: list[int] = []
                group_block_size_scales: list[int] = []
                seen_group_ptrs: set[int] = set()
                for layer_name in group_spec.layer_names:
                    cache_or_caches = kv_caches[layer_name]
                    if not isinstance(cache_or_caches, (list, tuple)):
                        cache_or_caches = (cache_or_caches,)
                    for cache in cache_or_caches:
                        base_addr = cache.data_ptr()
                        if base_addr in seen_group_ptrs:
                            continue
                        group_addrs.append(base_addr)
                        block_len = cache[0].numel() * cache.element_size()
                        tensor_num_blocks = cache.shape[0]
                        assert tensor_num_blocks % self.num_blocks == 0, (
                            "The external block size must be an integer multiple of the kernel block size."
                        )
                        group_block_lens.append(block_len)
                        group_block_size_scales.append(tensor_num_blocks // self.num_blocks)
                        seen_group_ptrs.add(base_addr)
                self.group_kv_caches_base_addr[group_id] = group_addrs
                self.group_block_len[group_id] = group_block_lens
                self.group_kv_block_size_scale[group_id] = group_block_size_scales

        for group_spec in self.cache_layout.state_groups:
            group_id = group_spec.group_id
            group_addrs: list[int] = []
            group_block_lens: list[int] = []
            group_block_size_scales: list[int] = []
            seen_group_ptrs: set[int] = set()
            for layer_name in group_spec.layer_names:
                cache_or_caches = self.kv_states.get(layer_name, ())
                if isinstance(cache_or_caches, torch.Tensor):
                    cache_or_caches = (cache_or_caches,)
                for cache in cache_or_caches:
                    base_addr = cache.data_ptr()
                    if base_addr in seen_group_ptrs:
                        continue
                    block_len = cache[0].numel() * cache.element_size()
                    tensor_num_blocks = cache.shape[0]
                    assert tensor_num_blocks % self.num_blocks == 0, (
                        "The external block size must be an integer multiple of the kernel block size."
                    )
                    block_size_scale = tensor_num_blocks // self.num_blocks
                    region_len = tensor_num_blocks * block_len
                    group_addrs.append(base_addr)
                    group_block_lens.append(block_len)
                    group_block_size_scales.append(block_size_scale)
                    if base_addr not in seen_ptrs:
                        ptrs.append(base_addr)
                        lengths.append(region_len)
                        seen_ptrs.add(base_addr)
                    seen_group_ptrs.add(base_addr)
            self.group_state_caches_base_addr[group_id] = group_addrs
            self.group_state_block_len[group_id] = group_block_lens
            self.group_state_block_size_scale[group_id] = group_block_size_scales
            if group_spec.skip_null_blocks and group_id < len(self.state_group_uses_align_state):
                self.state_group_uses_align_state[group_id] = True

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        self.token_database.set_block_len(self.block_len)
        self.token_database.set_block_size_scale(self.block_size_scale)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_kv_block_size_scale,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
        )
        self.token_database.set_group_buffers(
            self.group_state_caches_base_addr,
            self.group_state_block_len,
            self.group_state_block_size_scale,
            cache_role="state",
            group_cache_families=self.group_state_cache_families,
        )

        if self.use_layerwise:
            self.get_event = threading.Event()
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    ready_event_sending,
                    self.num_layers,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self.get_event,
            )
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    ready_event,
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        for request in metadata.requests:
            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.skip_null_state_blocks_by_group = self.state_group_uses_align_state
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load =0
                continue
            token_len = request.token_len_chunk
            if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
                load_spec.kvpool_cached_tokens == token_len - 1
            ):
                token_len = request.load_spec.kvpool_cached_tokens + 1
            else:
                token_len = request.load_spec.kvpool_cached_tokens
            request.load_spec.token_len = token_len
            if self.use_layerwise:
                layerwise_retriever = self.retrieve_layer(request)
                next(layerwise_retriever)  # first layer load
                self.layerwise_retrievers.append(layerwise_retriever)
            else:
                if self.load_async:
                    self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                        request,
                    )
                else:
                    addr_list = []
                    size_list = []
                    key_list = []
                    for cache_role, group_ids, block_ids_by_group, skip_null_blocks in (
                        (
                            "kv",
                            request.kv_cache_group_ids or [0],
                            request.block_ids_by_group,
                            self.group_uses_align_state,
                        ),
                        (
                            "state",
                            request.state_group_ids or [],
                            request.state_block_ids_by_group or [],
                            self.state_group_uses_align_state,
                        ),
                    ):
                        for group_id in group_ids:
                            block_ids = block_ids_by_group[group_id]
                            group_block_size = self.grouped_block_size[group_id]
                            mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
                            skip_null = group_id < len(skip_null_blocks) and skip_null_blocks[group_id]
                            for start, end, key, _ in self.token_database.process_tokens_with_block_ids(
                                token_len,
                                request.block_hashes,
                                block_ids,
                                mask_num,
                                kv_cache_group_id=group_id,
                                skip_null_blocks=skip_null,
                                cache_role=cache_role,
                            ):
                                addr, size, _ = self.token_database.prepare_value(
                                    start,
                                    end,
                                    block_ids,
                                    kv_cache_group_id=group_id,
                                    cache_role=cache_role,
                                )
                                key_list.append(key.to_string())
                                addr_list.append(addr)
                                size_list.append(size)
                    if not key_list:
                        continue
                    key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                    addr_list_c = (
                        addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                    )
                    size_list_c = (
                        size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                    )
                    self.m_store.get(key_list_c, addr_list_c, size_list_c)

    def wait_for_layer_load(self) -> None:
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)
            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.debug("Retrieved %s tokens", num_retrieved_tokens)

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        if self.current_layer == 0:
            self.layerwise_storers = []
            current_event = None
            for request in connector_metadata.requests:
                can_save = request.can_save
                if can_save is None or not can_save:
                    continue
                request.skip_null_blocks_by_group = self.group_uses_align_state
                current_event = torch.npu.Event()
                current_event.record()
                break
            for request in connector_metadata.requests:
                can_save = request.can_save
                if can_save is None or not can_save:
                    continue

                request.skip_null_blocks_by_group = self.group_uses_align_state
                layerwise_storer = self.store_layer(request, current_event)
                self.layerwise_storers.append(layerwise_storer)
        for layerwise_storer in self.layerwise_storers:
            try:
                next(layerwise_storer)
            except Exception:
                raise
        self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        current_event = None
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            if hasattr(current_event, "record"):
                current_event.record()
            break

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.skip_null_state_blocks_by_group = self.state_group_uses_align_state
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )

    def retrieve_layer(
        self,
        request: ReqMeta,
    ) -> Generator[torch.Tensor | None, None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the KV transfer which
            will be passed into the npu_transfer.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration.
        """
        token_len = request.token_len_chunk
        mask_num = (
            request.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.grouped_block_size[0]
            * self.grouped_block_size[0]
        )
        num_required_tokens = token_len - mask_num

        ret_mask = torch.zeros(token_len, dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []
        first_flag = True
        for start, end, key in self.token_database.process_tokens(token_len, request.block_hashes, mask_num):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys = [list(row) for row in zip(*keys)]  # [num_layer,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                if not first_flag:
                    is_finish = self.get_event.wait(timeout=3)  # try---cache
                    if not is_finish:
                        logger.info("Layerwise get failed")
                self.get_event.clear()
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id,
                    keys_multi_chunk,
                    starts,
                    ends,
                    request.block_ids_by_group,
                    layer_id,
                )
                self.kv_recv_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                first_flag = False
                yield None
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        retrieved_tokens = torch.sum(ret_mask)
        logger.debug("Retrieved %s out of %s out of total %s tokens", retrieved_tokens, num_required_tokens, token_len)

        yield ret_mask

    def store_layer(
        self,
        request: ReqMeta,
        current_event: torch.npu.Event | None,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(request.token_len_chunk, request.block_hashes):
            keys_multi_layer = key.split_layers(self.num_layers)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)  # [block_num,layer_num]

        if keys:
            keys = [list(row) for row in zip(*keys)]  # [layer_num,block_num]
            for layer_id, keys_multi_chunk in enumerate(keys):
                req_meta = LayerMultiBlockReqMeta(
                    request.req_id,
                    keys_multi_chunk,
                    starts,
                    ends,
                    request.block_ids_by_group,
                    layer_id,
                    request.is_last_chunk,
                    current_event,
                )
                self.kv_send_thread.add_request(  # type: ignore[union-attr, call-arg]
                    req_meta
                )  # type: ignore[union-attr, call-arg, arg-type]
                yield
        else:
            for layer_id in range(self.num_layers):
                yield

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        done_sending = (
            self.get_and_clear_finished_requests(
                finished_req_ids,
                meta,  # type: ignore[union-attr]
            )
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.load_async
            else set()
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
                len(done_sending),
                len(done_recving),
                self.tp_rank,
            )
        return done_sending, done_recving

    def get_and_clear_finished_requests(self, finished_req_ids, meta: AscendConnectorMetadata) -> set[str]:
        finished_sending = set()
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                req_id
            )
        for req_id in self.kv_send_thread.stored_requests.copy(  # type: ignore[union-attr]
        ):
            if (
                self.kv_send_thread.stored_requests[  # type: ignore[union-attr]
                    req_id
                ]
                == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(  # type: ignore[union-attr]
                req_id
            )
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        state_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            cache_specs: list[tuple[str, int, list[bool]]] = [
                ("kv", group_id, self.group_uses_align_state) for group_id in kv_cache_group_ids
            ]
            cache_specs.extend(
                ("state", group_id, self.state_group_uses_align_state) for group_id in (state_group_ids or [])
            )
            for cache_role, group_id, skip_null_groups in cache_specs:
                end = 0
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                    cache_role=cache_role,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    hits.append(0)
                    continue

                res = self.m_store.exists(keys)  # type: ignore[assignment]

                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                if group_id < len(skip_null_groups) and skip_null_groups[group_id]:
                    hit_end = 0
                    for index in range(len(ends) - 1, -1, -1):
                        if res[index] == 1 and ends[index] % self.cache_transfer_granularity == 0:  # type: ignore[index]
                            hit_end = ends[index]
                            break
                else:
                    hit_end = end
                    for index, value in enumerate(res):  # type: ignore[arg-type]
                        if value != 1:
                            hit_end = 0
                            for hit_index in range(index, 0, -1):
                                if starts[hit_index] % self.cache_transfer_granularity == 0:
                                    hit_end = starts[hit_index]
                                    break
                            break
                hits.append(hit_end)
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0
        return min(hits) if hits else 0

    def _get_group_num_kv_heads(self, group_id: int, cache_role: str) -> int:
        if cache_role == "state" or self.use_mla or self.use_sparse:
            return 1
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            return 1
        return self.num_kv_head

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        state_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            cache_specs: list[tuple[str, int, list[bool]]] = [
                ("kv", group_id, self.group_uses_align_state) for group_id in kv_cache_group_ids
            ]
            cache_specs.extend(
                ("state", group_id, self.state_group_uses_align_state) for group_id in (state_group_ids or [])
            )
            for cache_role, group_id, skip_null_groups in cache_specs:
                end = 0
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                    cache_role=cache_role,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    hits.append(0)
                    continue

                multi_tp_keys = keys[:]
                group_num_kv_head = self._get_group_num_kv_heads(group_id, cache_role)
                group_tp_size = min(self.tp_size, group_num_kv_head)
                for i in range(1, group_tp_size):
                    for item in keys:
                        new_str = item.replace(  # type: ignore[attr-defined]
                            "@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1
                        )
                        multi_tp_keys.append(new_str)

                pp_base_keys = multi_tp_keys.copy()
                for i in range(1, self.pp_size):
                    for item in pp_base_keys:
                        new_str = item.replace(  # type: ignore[attr-defined]
                            "@pp_rank:0", f"@pp_rank:{i}", 1
                        )
                        multi_tp_keys.append(new_str)

                res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
                num_block = len(keys)
                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                    num_block = len(keys) // self.num_layers
                multi_tp_values = [
                    res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                    for i in range(group_tp_size * self.pp_size)
                ]
                if group_id < len(skip_null_groups) and skip_null_groups[group_id]:
                    exists_by_block = [all(values[idx] == 1 for values in multi_tp_values) for idx in range(num_block)]
                    hit_end = 0
                    for index in range(num_block - 1, -1, -1):
                        if exists_by_block[index] and ends[index] % self.cache_transfer_granularity == 0:
                            hit_end = ends[index]
                            break
                    hits.append(hit_end)
                else:
                    index = self.find_min_first_non_one_index(multi_tp_values)
                    if index == -1:
                        hits.append(end)
                    else:
                        hit_end = 0
                        for hit_index in range(index, 0, -1):
                            if starts[hit_index] % self.cache_transfer_granularity == 0:
                                hit_end = starts[hit_index]
                                break
                        hits.append(hit_end)
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return 0
        return min(hits) if hits else 0

    def check_all_layers_exists(self, res: list[int], num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    def find_min_first_non_one_index(self, arr):
        try:
            return min(idx for row in arr for idx, val in enumerate(row) if val != 1)
        except ValueError:
            return -1

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []
