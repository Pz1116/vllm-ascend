import threading
import unittest
from types import SimpleNamespace

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
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

    def set_device(self):
        return None

    def exists(self, keys: list[str]) -> list[int]:
        # Return exact number of states for requested keys.
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


class _FakeTokenDatabase:
    hash_block_size = 16

    def process_tokens(self, token_len, block_hashes):
        for i, _ in enumerate(block_hashes):
            yield i * 16, (i + 1) * 16, _FakeKey(f"k{i}")

    def process_tokens_with_block_ids(
        self,
        token_len,
        block_hashes,
        block_ids,
        mask_num=0,
        kv_cache_group_id=0,
        skip_null_blocks=False,
        cache_role="kv",
    ):
        for start, end, key in self.process_tokens(token_len, block_hashes):
            block_idx = start // 16
            if start < mask_num or block_idx >= len(block_ids):
                continue
            block_id = block_ids[block_idx]
            if skip_null_blocks and block_id <= 0:
                continue
            yield start, end, key, block_id

    def prepare_value(self, start, end, block_ids, kv_cache_group_id=0, cache_role="kv"):
        block_id = start // 16
        return [1000 + block_id], [end - start], block_id

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes, kv_cache_group_id=0, cache_role="kv"):
        return keys, addrs, sizes

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = start // 16
        return [2000 + layer_id * 100 + block_id], [end - start]


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
            current_event=None,
        )
        thread.add_stored_request("req-1")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[1001], [1003]])
        self.assertEqual(put_sizes, [[16], [16]])

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


class TestHybridKVPoolMetadata(unittest.TestCase):
    def test_group_family_is_encoded_in_key(self):
        token_db = ChunkedTokenDatabase(
            metadata=KeyMetadata("test-model", 0, 0, 0, 0),
            block_size=[16, 128],
            partitions=None,
            use_hybrid=True,
            hash_block_size=16,
        )
        token_db.set_group_buffers(
            group_kv_caches_base_addr={1: [4096]},
            group_block_len={1: [256]},
            group_block_stride={1: [512]},
            group_cache_families={1: "c128"},
            group_num_layers={1: 1},
        )

        _, _, key = next(
            token_db.process_tokens(
                token_len=128,
                block_hashes=["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"],
                kv_cache_group_id=1,
            )
        )

        self.assertIn("@group:1", key.to_string())
        self.assertIn("@cache_role:kv", key.to_string())
        self.assertIn("@cache_family:c128", key.to_string())

    def test_string_hashes_are_grouped_by_target_block_size(self):
        token_db = ChunkedTokenDatabase(
            metadata=KeyMetadata("test-model", 0, 0, 0, 0),
            block_size=[16, 64],
            partitions=None,
            use_hybrid=True,
            hash_block_size=16,
        )
        token_db.set_group_buffers(
            group_kv_caches_base_addr={1: [4096]},
            group_block_len={1: [256]},
            group_block_stride={1: [512]},
            group_cache_families={1: "c4"},
            group_num_layers={1: 1},
        )

        chunks = list(
            token_db.process_tokens(
                token_len=128,
                block_hashes=["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"],
                kv_cache_group_id=1,
            )
        )

        self.assertEqual([start for start, _, _ in chunks], [0, 64])
        self.assertTrue(chunks[0][2].to_string().endswith("@h0h1h2h3"))
        self.assertTrue(chunks[1][2].to_string().endswith("@h4h5h6h7"))

    def test_prepare_value_uses_group_stride_not_block_len(self):
        token_db = ChunkedTokenDatabase(
            metadata=KeyMetadata("test-model", 0, 0, 0, 0),
            block_size=[16],
            partitions=None,
            use_hybrid=True,
        )
        token_db.set_group_buffers(
            group_kv_caches_base_addr={0: [1000]},
            group_block_len={0: [128]},
            group_block_stride={0: [512]},
            group_cache_families={0: "c1"},
            group_num_layers={0: 1},
        )

        addrs, sizes, block_id = token_db.prepare_value(
            start=16,
            end=32,
            block_ids=[3, 5],
            kv_cache_group_id=0,
        )

        self.assertEqual(block_id, 5)
        self.assertEqual(addrs, [1000 + 5 * 512])
        self.assertEqual(sizes, [128])

    def test_group_cache_family_requires_same_compress_ratio(self):
        groups = [
            SimpleNamespace(layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn"]),
            SimpleNamespace(layer_names=["model.layers.2.self_attn"]),
        ]

        self.assertEqual(infer_group_cache_families(groups, [4, 4, 128]), ["c4", "c128"])
        with self.assertRaises(AssertionError):
            infer_group_cache_families(groups, [4, 128, 128])


if __name__ == "__main__":
    unittest.main()
