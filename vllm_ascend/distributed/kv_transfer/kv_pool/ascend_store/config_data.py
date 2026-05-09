from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional, cast

import regex as re
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import NewRequestData


# Parameters related to the key
@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current prefill context model parallel rank """
    pcp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int
    """ Initialize the current kv cache group id """
    kv_cache_group_id: int = 0
    """ Differentiate kv/state keys that share the same chunk hash """
    cache_role: str = "kv"
    """ Family name for compress-aware hybrid cache layouts """
    cache_family: str = "default"


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["LayerPoolKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                )
            )
        return keys


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.kv_cache_group_id,
                self.key_metadata.cache_role,
                self.key_metadata.cache_family,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@group:{self.key_metadata.kv_cache_group_id}"
            f"@cache_role:{self.key_metadata.cache_role}"
            f"@cache_family:{self.key_metadata.cache_family}"
            f"@{self.chunk_hash}@{self.layer_id}"
        )


@dataclass
class CacheGroupLayout:
    group_id: int
    layer_names: list[str]
    cache_role: str = "kv"
    cache_family: str = "default"
    skip_null_blocks: bool = False


@dataclass
class CacheLayout:
    kv_groups: list[CacheGroupLayout]
    state_groups: list[CacheGroupLayout]


def extract_layer_idx(layer_name: str) -> int:
    match = re.search(r"\.(\d+)\.", layer_name)
    if match is None:
        raise ValueError(f"Can not find layer_idx in layer name: {layer_name}")
    return int(match.group(1))


def infer_cache_family_from_ratio(compress_ratio: int | None) -> str:
    if compress_ratio is None:
        return "default"
    return f"c{compress_ratio}"


def infer_group_cache_families(
    kv_cache_groups: Sequence[object] | None,
    compress_ratios: Sequence[int] | None,
) -> list[str]:
    if kv_cache_groups is None:
        return ["default"]

    families: list[str] = []
    for group in kv_cache_groups:
        layer_names = list(getattr(group, "layer_names", []))
        if compress_ratios is None or not layer_names:
            families.append("default")
            continue

        group_ratios = {compress_ratios[extract_layer_idx(layer_name)] for layer_name in layer_names}
        assert len(group_ratios) == 1, (
            f"All layers in one KV cache group must share the same compress ratio, "
            f"got {sorted(group_ratios)} for layers {layer_names}"
        )
        families.append(infer_cache_family_from_ratio(next(iter(group_ratios))))
    return families


class ChunkedTokenDatabase:
    def __init__(self, metadata: KeyMetadata, block_size: int, partitions: list[int] | None):
        self.metadata = metadata
        self.block_size = block_size
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.block_size_scale: list[int] = []
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_kv_block_size_scale: dict[int, list[int]] = {}
        self.group_state_caches_base_addr: dict[int, list[int]] = {}
        self.group_state_block_len: dict[int, list[int]] = {}
        self.group_state_block_size_scale: dict[int, list[int]] = {}
        self.group_cache_families: dict[str, dict[int, str]] = {
            "kv": {},
            "state": {},
        }
        self.partitions = partitions

    def _make_key_by_hash(
        self,
        chunk_hash: str,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
        layer_id: int | None = None,
    ):
        assert self.metadata is not None
        if cache_family is None:
            cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
        return PoolKey(
            KeyMetadata(
                model_name=self.metadata.model_name,
                head_or_tp_rank=self.metadata.head_or_tp_rank,
                pcp_rank=self.metadata.pcp_rank,
                dcp_rank=self.metadata.dcp_rank,
                pp_rank=self.metadata.pp_rank,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                cache_family=cache_family,
            ),
            chunk_hash,
        )

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len

    def set_block_size_scale(self, block_size_scale: list[int]):
        self.block_size_scale = block_size_scale

    def set_group_buffers(
        self,
        group_kv_caches_base_addr: dict[int, list[int]],
        group_block_len: dict[int, list[int]],
        group_block_size_scale: dict[int, list[int]] | None = None,
        cache_role: str = "kv",
        group_cache_families: dict[int, str] | None = None,
    ) -> None:
        if cache_role == "state":
            self.group_state_caches_base_addr = group_kv_caches_base_addr
            self.group_state_block_len = group_block_len
            self.group_state_block_size_scale = group_block_size_scale or {}
        else:
            self.group_kv_caches_base_addr = group_kv_caches_base_addr
            self.group_block_len = group_block_len
            self.group_kv_block_size_scale = group_block_size_scale or {}
        if group_cache_families is not None:
            self.group_cache_families[cache_role] = group_cache_families.copy()

    def _get_group_buffers(self, kv_cache_group_id: int, cache_role: str) -> tuple[list[int], list[int], list[int]]:
        if cache_role == "state":
            return (
                self.group_state_caches_base_addr.get(kv_cache_group_id, []),
                self.group_state_block_len.get(kv_cache_group_id, []),
                self.group_state_block_size_scale.get(kv_cache_group_id, []),
            )
        return (
            self.group_kv_caches_base_addr.get(kv_cache_group_id, self.kv_caches_base_addr),
            self.group_block_len.get(kv_cache_group_id, self.block_len),
            self.group_kv_block_size_scale.get(kv_cache_group_id, self.block_size_scale),
        )

    def prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        addr_list = []
        size_list = []
        block_id = block_ids[start // self.block_size]
        group_addrs, group_block_len, group_block_size_scale = self._get_group_buffers(kv_cache_group_id, cache_role)
        length = len(group_block_len)
        for index, base_addr in enumerate(group_addrs):
            block_len = group_block_len[index % length]
            block_size_scale = group_block_size_scale[index % length] if group_block_size_scale else 1
            addr = base_addr + block_id * block_len * block_size_scale
            size = int(block_len * block_size_scale / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    def prepare_state_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        state_group_id: int = 0,
    ):
        return self.prepare_value(
            start,
            end,
            block_ids,
            kv_cache_group_id=state_group_id,
            cache_role="state",
        )

    def prepare_value_layer(self, start: int, end: int, block_ids: list[int], layer_id: int):
        block_id = block_ids[start // self.block_size]
        addr_list = []
        size_list = []
        length = len(self.block_len)
        for i in range(length):
            addr = self.kv_caches_base_addr[layer_id * length] + block_id * self.block_len[i]
            size = int(self.block_len[i] / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [
                h.hex()  # type: ignore[union-attr]
                for h in block_hashes
            ]
        start_idx = 0
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            else:
                yield (
                    start_idx,
                    end_idx,
                    self._make_key_by_hash(
                        hash_val,
                        kv_cache_group_id=kv_cache_group_id,
                        cache_role=cache_role,
                        cache_family=cache_family,
                    ),
                )

    def process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
        cache_family: str | None = None,
    ) -> Iterable[tuple[int, int, PoolKey, int]]:
        for start_idx, end_idx, key in self.process_tokens(
            token_len,
            block_hashes,
            mask_num,
            kv_cache_group_id=kv_cache_group_id,
            cache_role=cache_role,
            cache_family=cache_family,
        ):
            block_idx = start_idx // self.block_size
            if block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            yield start_idx, end_idx, key, block_id

    def decode_adaptor_prefill_pp(self, key, addr, size):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            start = 0
            for j, part in enumerate(self.partitions):
                # part * 2 because addr and size contain both k and v
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * 2
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size


def normalize_block_ids_by_group(block_ids: tuple[list[int], ...] | list[int] | list[list[int]]) -> list[list[int]]:
    if isinstance(block_ids, tuple):
        return [group.copy() for group in block_ids]
    if isinstance(block_ids, list):
        if not block_ids:
            return [[]]
        if isinstance(block_ids[0], list):
            grouped_block_ids = cast(list[list[int]], block_ids)
            return [group.copy() for group in grouped_block_ids]
        flat_block_ids = cast(list[int], block_ids)
        return [flat_block_ids.copy()]
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass(init=False)
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids_by_group: list[list[int]]
    # Optional state block ids. If absent, state save/load falls back to KV
    # block ids for the same group.
    allocated_state_block_ids_by_group: list[list[int]] | None = None

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    def __init__(
        self,
        req_id: str,
        token_len: int,
        allocated_block_ids_by_group: list[list[int]] | None = None,
        allocated_state_block_ids_by_group: list[list[int]] | None = None,
        num_saved_tokens: int = 0,
        token_ids: list[int] | None = None,
        allocated_block_ids: list[int] | None = None,
    ):
        self.req_id = req_id
        self.token_len = token_len
        if allocated_block_ids_by_group is None:
            allocated_block_ids_by_group = normalize_block_ids_by_group(allocated_block_ids or [])
        self.allocated_block_ids_by_group = allocated_block_ids_by_group
        self.allocated_state_block_ids_by_group = allocated_state_block_ids_by_group
        self.num_saved_tokens = num_saved_tokens
        self.token_ids = token_ids

    @property
    def allocated_block_ids(self) -> list[int]:
        return self.allocated_block_ids_by_group[0] if self.allocated_block_ids_by_group else []

    @allocated_block_ids.setter
    def allocated_block_ids(self, block_ids: list[int]) -> None:
        self.allocated_block_ids_by_group = normalize_block_ids_by_group(block_ids)

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=normalize_block_ids_by_group(new_request.block_ids),
            num_saved_tokens=0,
        )

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """
        normalized = normalize_block_ids_by_group(new_block_ids)
        if len(normalized) > len(self.allocated_block_ids_by_group):
            self.allocated_block_ids_by_group.extend(
                [[] for _ in range(len(normalized) - len(self.allocated_block_ids_by_group))]
            )
        for group_id, ids in enumerate(normalized):
            self.allocated_block_ids_by_group[group_id].extend(ids)


@dataclass(init=False)
class ReqMeta:
    # Request id
    req_id: str
    # Number of tokens in this chunk
    token_len_chunk: int

    block_ids_by_group: list[list[int]]

    block_hashes: list[BlockHash]

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None
    kv_cache_group_ids: list[int] | None = None
    kv_cache_families_by_group: list[str] | None = None
    state_group_ids: list[int] | None = None
    state_cache_families_by_group: list[str] | None = None
    state_block_ids_by_group: list[list[int]] | None = None
    skip_null_blocks_by_group: list[bool] | None = None
    skip_null_state_blocks_by_group: list[bool] | None = None

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: int | None = None

    def __init__(
        self,
        req_id: str,
        token_len_chunk: int,
        block_ids_by_group: list[list[int]] | None = None,
        block_hashes: list[BlockHash] | None = None,
        can_save: bool | None = None,
        load_spec: LoadSpec | None = None,
        is_last_chunk: bool | None = None,
        current_event: torch.npu.Event | None = None,
        kv_cache_group_ids: list[int] | None = None,
        kv_cache_families_by_group: list[str] | None = None,
        state_group_ids: list[int] | None = None,
        state_cache_families_by_group: list[str] | None = None,
        state_block_ids_by_group: list[list[int]] | None = None,
        skip_null_blocks_by_group: list[bool] | None = None,
        skip_null_state_blocks_by_group: list[bool] | None = None,
        token_ids: list[int] | None = None,
        original_block_size: int | None = None,
        block_ids: list[int] | None = None,
    ):
        self.req_id = req_id
        self.token_len_chunk = token_len_chunk
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.block_hashes = block_hashes or []
        self.can_save = can_save
        self.load_spec = load_spec
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event
        self.kv_cache_group_ids = kv_cache_group_ids
        self.kv_cache_families_by_group = kv_cache_families_by_group
        self.state_group_ids = state_group_ids
        self.state_cache_families_by_group = state_cache_families_by_group
        self.state_block_ids_by_group = state_block_ids_by_group
        self.skip_null_blocks_by_group = skip_null_blocks_by_group
        self.skip_null_state_blocks_by_group = skip_null_state_blocks_by_group
        self.token_ids = token_ids
        self.original_block_size = original_block_size

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: int | None = None,
        kv_cache_group_families: list[str] | None = None,
        state_group_ids: list[int] | None = None,
        state_cache_group_families: list[str] | None = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM scheduler and AscendConnector.
                If context parallelism is enabled, block_size = block_size * pcp_size * dcp_size.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.
            original_block_size (int | None): the block size in vLLM worker. This is only used for kv event generation.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size if discard_partial_chunks else 0
        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (input_token_len // block_size * block_size) if discard_partial_chunks else input_token_len

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # Get the token ids for kv event generation in kv_transfer
        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        # # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None
        logger.debug("request:%s, meta save spec:%s, meta load spec:%s", tracker.req_id, not skip_save, load_spec)
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids_by_group=tracker.allocated_block_ids_by_group,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
            kv_cache_group_ids=list(range(len(tracker.allocated_block_ids_by_group))),
            kv_cache_families_by_group=kv_cache_group_families,
            state_group_ids=state_group_ids,
            state_cache_families_by_group=state_cache_group_families,
            state_block_ids_by_group=(
                tracker.allocated_state_block_ids_by_group
                if tracker.allocated_state_block_ids_by_group is not None
                else [group.copy() for group in tracker.allocated_block_ids_by_group]
            ),
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(self, unfinished_request_ids, preempted_req_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass(init=False)
class LayerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids_by_group: list[list[int]]
    layer_id: int
    is_last_chunk: bool | None = True
    current_event: torch.npu.Event | None = None

    def __init__(
        self,
        req_id: str,
        keys: list[LayerPoolKey],
        starts: list[int],
        ends: list[int],
        block_ids_by_group: list[list[int]] | None = None,
        layer_id: int = 0,
        is_last_chunk: bool | None = True,
        current_event: torch.npu.Event | None = None,
        block_ids: list[int] | None = None,
    ):
        self.req_id = req_id
        self.keys = keys
        self.starts = starts
        self.ends = ends
        if block_ids_by_group is None:
            block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
        self.block_ids_by_group = block_ids_by_group
        self.layer_id = layer_id
        self.is_last_chunk = is_last_chunk
        self.current_event = current_event

    @property
    def block_ids(self) -> list[int]:
        return self.block_ids_by_group[0] if self.block_ids_by_group else []

    @block_ids.setter
    def block_ids(self, block_ids: list[int]) -> None:
        self.block_ids_by_group = normalize_block_ids_by_group(block_ids)
