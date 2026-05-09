import threading
import unittest
from types import SimpleNamespace

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)


class _FakeKey:
    def __init__(self, value: str):
        self._value = value

    def to_string(self) -> str:
        return self._value


class _FakeStore:
    def __init__(self, exists_result: list[int]):
        self.exists_result = exists_result
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
        self.get_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []

    def set_device(self):
        return None

    def exists(self, keys: list[str]) -> list[int]:
        # Return exact number of states for requested keys.
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class _FakeTokenDatabase:
    def __init__(self):
        self.decode_calls: list[list[str]] = []

    def process_tokens(
        self, token_len, block_hashes, mask_num=0, kv_cache_group_id=0, cache_role="kv", cache_family=None
    ):
        for i, _ in enumerate(block_hashes):
            yield i * 16, (i + 1) * 16, _FakeKey(f"{cache_role}{kv_cache_group_id}-k{i}")

    def process_tokens_with_block_ids(
        self,
        token_len,
        block_hashes,
        block_ids,
        mask_num=0,
        kv_cache_group_id=0,
        skip_null_blocks=False,
        cache_role="kv",
        cache_family=None,
    ):
        for i, _ in enumerate(block_hashes):
            block_id = block_ids[i]
            if skip_null_blocks and block_id <= 0:
                continue
            yield i * 16, (i + 1) * 16, _FakeKey(f"{cache_role}{kv_cache_group_id}-k{i}"), block_id

    def prepare_value(self, start, end, block_ids, kv_cache_group_id=0, cache_role="kv"):
        block_id = start // 16
        base = (1000 if cache_role == "kv" else 5000) + kv_cache_group_id * 100
        return [base + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = start // 16
        return [2000 + layer_id * 100 + block_id], [end - start]

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        self.decode_calls.append(list(keys))
        return [f"pp-{key}" for key in keys], addrs, sizes


class TestKVTransferMissingKeyPut(unittest.TestCase):
    def test_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-1",
            token_len_chunk=64,
            block_ids_by_group=[[0, 1, 2, 3]],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0],
            current_event=None,
        )
        thread.add_stored_request("req-1")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["kv0-k1", "kv0-k3"])
        self.assertEqual(put_addrs, [[1001], [1003]])
        self.assertEqual(put_sizes, [[16], [16]])

    def test_sending_thread_puts_state_groups_too(self):
        store = _FakeStore(exists_result=[0, 0, 0, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-state",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1]],
            state_block_ids_by_group=[[2, 3]],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0],
            state_group_ids=[0],
            current_event=None,
        )
        thread.add_stored_request("req-state")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 2)
        kv_put_keys, kv_put_addrs, _ = store.put_calls[0]
        state_put_keys, state_put_addrs, _ = store.put_calls[1]
        self.assertEqual(kv_put_keys, ["kv0-k0", "kv0-k1"])
        self.assertEqual(state_put_keys, ["state0-k0", "state0-k1"])
        self.assertEqual(kv_put_addrs, [[1000], [1001]])
        self.assertEqual(state_put_addrs, [[5000], [5001]])

    def test_sending_thread_applies_pp_adaptor_to_state_groups(self):
        store = _FakeStore(exists_result=[0, 0, 0, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_consumer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-state-pp",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1]],
            state_block_ids_by_group=[[2, 3]],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0],
            state_group_ids=[0],
            current_event=None,
        )
        thread.add_stored_request("req-state-pp")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(token_db.decode_calls, [["kv0-k0", "kv0-k1"], ["state0-k0", "state0-k1"]])
        self.assertEqual(store.put_calls[0][0], ["pp-kv0-k0", "pp-kv0-k1"])
        self.assertEqual(store.put_calls[1][0], ["pp-state0-k0", "pp-state0-k1"])

    def test_layer_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=2,
            enable_kv_event=False,
        )

        req_meta = LayerMultiBlockReqMeta(
            req_id="req-2",
            keys=[_FakeKey("k0"), _FakeKey("k1"), _FakeKey("k2"), _FakeKey("k3")],  # type: ignore[arg-type]
            starts=[0, 16, 32, 48],
            ends=[16, 32, 48, 64],
            block_ids_by_group=[[0, 1, 2, 3]],
            layer_id=1,
            is_last_chunk=False,
            current_event=None,
        )
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[2101], [2103]])
        self.assertEqual(put_sizes, [[16], [16]])

    def test_sending_thread_handles_two_kv_groups(self):
        store = _FakeStore(exists_result=[0, 0, 0, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-groups",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1], [2, 3]],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0, 1],
            current_event=None,
        )
        thread.add_stored_request("req-groups")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 2)
        self.assertEqual(store.put_calls[0][0], ["kv0-k0", "kv0-k1"])
        self.assertEqual(store.put_calls[0][1], [[1000], [1001]])
        self.assertEqual(store.put_calls[1][0], ["kv1-k0", "kv1-k1"])
        self.assertEqual(store.put_calls[1][1], [[1100], [1101]])

    def test_recving_thread_loads_kv_and_state_groups(self):
        store = _FakeStore(exists_result=[])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )

        req_meta = ReqMeta(
            req_id="req-load",
            token_len_chunk=32,
            block_ids_by_group=[[0, 1]],
            state_block_ids_by_group=[[2, 3]],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0],
            state_group_ids=[0],
            load_spec=SimpleNamespace(token_len=32, vllm_cached_tokens=0),
        )
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.get_calls), 1)
        keys, addrs, sizes = store.get_calls[0]
        self.assertEqual(keys, ["kv0-k0", "kv0-k1", "state0-k0", "state0-k1"])
        self.assertEqual(addrs, [[1000], [1001], [5000], [5001]])
        self.assertEqual(sizes, [[16], [16], [16], [16]])

    def test_recving_thread_handles_empty_key_list(self):
        store = _FakeStore(exists_result=[])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
        )

        req_meta = ReqMeta(
            req_id="req-empty-load",
            token_len_chunk=0,
            block_ids_by_group=[[]],
            block_hashes=[],
            kv_cache_group_ids=[0],
            load_spec=SimpleNamespace(token_len=0, vllm_cached_tokens=0),
        )
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(store.get_calls, [])


if __name__ == "__main__":
    unittest.main()
