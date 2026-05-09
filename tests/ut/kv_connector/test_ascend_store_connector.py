import unittest
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
    RequestTracker,
    infer_group_cache_families,
    normalize_block_ids_by_group,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class _FakeBackend:
    def __init__(self):
        self.registered_ptrs: list[int] = []
        self.registered_lengths: list[int] = []

    def register_buffer(self, ptrs, lengths):
        self.registered_ptrs = list(ptrs)
        self.registered_lengths = list(lengths)


class TestAscendStoreGroupAwareConfig(unittest.TestCase):
    def test_normalize_block_ids_by_group(self):
        self.assertEqual(normalize_block_ids_by_group([1, 2]), [[1, 2]])
        self.assertEqual(normalize_block_ids_by_group([]), [[]])
        self.assertEqual(
            normalize_block_ids_by_group(([1, 2], [3, 4])),
            [[1, 2], [3, 4]],
        )
        self.assertEqual(normalize_block_ids_by_group(tuple()), [])
        self.assertEqual(
            normalize_block_ids_by_group([[5, 6], [7, 8]]),
            [[5, 6], [7, 8]],
        )

    def test_group_id_is_part_of_cache_key(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        key_group0 = next(token_db.process_tokens(16, ["hash0"], kv_cache_group_id=0))[2]
        key_group1 = next(token_db.process_tokens(16, ["hash0"], kv_cache_group_id=1))[2]

        self.assertNotEqual(key_group0.to_string(), key_group1.to_string())
        self.assertIn("@group:0@cache_role:kv@cache_family:default@", key_group0.to_string())
        self.assertIn("@group:1@cache_role:kv@cache_family:default@", key_group1.to_string())

    def test_state_key_uses_role_and_family(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        token_db.set_group_buffers({0: [1000]}, {0: [64]}, cache_role="state", group_cache_families={0: "c4"})
        key = next(token_db.process_tokens(16, ["hash0"], kv_cache_group_id=0, cache_role="state"))[2]

        self.assertIn("@cache_role:state@", key.to_string())
        self.assertIn("@cache_family:c4@", key.to_string())

    def test_prepare_value_uses_group_specific_buffers(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        token_db.set_kv_caches_base_addr([1000])
        token_db.set_block_len([64])
        token_db.set_group_buffers(
            {1: [2000, 3000]},
            {1: [128, 256]},
        )

        addrs, sizes, block_id = token_db.prepare_value(
            start=16,
            end=32,
            block_ids=[10, 11],
            kv_cache_group_id=1,
        )

        self.assertEqual(block_id, 11)
        self.assertEqual(addrs, [2000 + 11 * 128, 3000 + 11 * 256])
        self.assertEqual(sizes, [128, 256])

    def test_prepare_value_respects_block_size_scale(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        token_db.set_group_buffers(
            {0: [4096]},
            {0: [64]},
            {0: [2]},
        )

        addrs, sizes, block_id = token_db.prepare_value(
            start=16,
            end=32,
            block_ids=[5, 7],
            kv_cache_group_id=0,
        )

        self.assertEqual(block_id, 7)
        self.assertEqual(addrs, [4096 + 7 * 64 * 2])
        self.assertEqual(sizes, [64 * 2])

    def test_prepare_state_value_uses_state_buffers(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)
        token_db.set_group_buffers(
            {0: [5000, 6000]},
            {0: [32, 48]},
            cache_role="state",
            group_cache_families={0: "c128"},
        )

        addrs, sizes, block_id = token_db.prepare_state_value(start=0, end=16, block_ids=[7], state_group_id=0)

        self.assertEqual(block_id, 7)
        self.assertEqual(addrs, [5000 + 7 * 32, 6000 + 7 * 48])
        self.assertEqual(sizes, [32, 48])

    def test_process_tokens_with_block_ids_skips_null_blocks(self):
        metadata = KeyMetadata(
            model_name="test-model",
            head_or_tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
        )
        token_db = ChunkedTokenDatabase(metadata, block_size=16, partitions=None)

        chunks = list(
            token_db.process_tokens_with_block_ids(
                token_len=48,
                block_hashes=["h0", "h1", "h2"],
                block_ids=[0, 0, 9],
                kv_cache_group_id=1,
                skip_null_blocks=True,
            )
        )

        self.assertEqual(len(chunks), 1)
        start, end, key, block_id = chunks[0]
        self.assertEqual((start, end, block_id), (32, 48, 9))
        self.assertIn("@group:1@", key.to_string())

    def test_req_meta_tracks_all_groups(self):
        tracker = RequestTracker(
            req_id="req-1",
            token_len=32,
            allocated_block_ids_by_group=[[1, 2], [10, 11]],
            token_ids=[1] * 32,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            block_hashes=["h0", "h1"],
            skip_save=False,
            discard_partial_chunks=True,
        )

        assert req_meta is not None
        self.assertEqual(req_meta.block_ids_by_group, [[1, 2], [10, 11]])
        self.assertEqual(req_meta.kv_cache_group_ids, [0, 1])

    def test_req_meta_tracks_state_groups(self):
        tracker = RequestTracker(
            req_id="req-2",
            token_len=32,
            allocated_block_ids_by_group=[[1, 2], [10, 11]],
            allocated_state_block_ids_by_group=[[21, 22], [31, 32]],
            token_ids=[1] * 32,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=16,
            block_hashes=["h0", "h1"],
            skip_save=False,
            discard_partial_chunks=True,
            kv_cache_group_families=["c1", "c4"],
            state_group_ids=[0, 1],
            state_cache_group_families=["c1", "c4"],
        )

        assert req_meta is not None
        self.assertEqual(req_meta.kv_cache_families_by_group, ["c1", "c4"])
        self.assertEqual(req_meta.state_group_ids, [0, 1])
        self.assertEqual(req_meta.state_cache_families_by_group, ["c1", "c4"])
        self.assertEqual(req_meta.state_block_ids_by_group, [[21, 22], [31, 32]])

    def test_infer_group_cache_families_asserts_mixed_ratios(self):
        groups = [
            SimpleNamespace(layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn"]),
        ]
        with self.assertRaises(AssertionError):
            infer_group_cache_families(groups, [1, 4])

    def test_request_finished_all_groups_ignores_empty_groups(self):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.kv_role = "kv_producer"
        scheduler.consumer_is_to_put = False
        scheduler.use_hybrid = False
        scheduler._request_trackers = {"req-1": SimpleNamespace(num_saved_tokens=16)}

        request = SimpleNamespace(request_id="req-1")
        delay_free, _ = scheduler.request_finished_all_groups(request, ([], [1, 2], []))
        self.assertTrue(delay_free)

        delay_free_empty, _ = scheduler.request_finished_all_groups(request, ([], [], []))
        self.assertFalse(delay_free_empty)

    def test_get_sw_clipped_blocks_only_clips_sliding_window_groups(self):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.use_hybrid = True
        scheduler.num_swa_blocks = [2, 0, 1]

        clipped = scheduler.get_sw_clipped_blocks(([1, 2, 3], [4, 5], [6, 7, 8]))

        self.assertEqual(clipped, ([2, 3], [4, 5], [8]))

    def _build_worker(self, compress_ratios=None):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.use_mla = False
        worker.use_sparse = False
        worker.model_type = "deepseek_v4" if compress_ratios is not None else "test"
        worker.compress_ratios = compress_ratios
        worker.use_compress = compress_ratios is not None
        worker.use_layerwise = False
        worker.kv_role = "disabled"
        worker.consumer_is_to_put = False
        worker.load_async = False
        worker.enable_kv_events = False
        worker.tp_rank = 0
        worker.dcp_size = 1
        worker.put_step = 1
        worker.num_layers = 1
        worker.group_uses_align_state = [False]
        worker.state_group_uses_align_state = []
        worker.state_group_ids = []
        worker.m_store = _FakeBackend()
        worker.token_database = ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0),
            block_size=16,
            partitions=None,
        )
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            head_size_v=8,
            dtype=torch.float16,
        )
        worker.kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["model.layers.0.self_attn", "model.layers.1.self_attn"],
                    kv_cache_spec=kv_cache_spec,
                )
            ],
        )
        return worker

    def test_register_kv_caches_sets_state_groups_only_when_states_exist(self):
        worker = self._build_worker(compress_ratios=[4, 4])
        shared_k = torch.zeros((2, 16, 1, 8), dtype=torch.float16)
        shared_v = torch.zeros((2, 16, 1, 8), dtype=torch.float16)
        kv_caches = {
            "model.layers.0.self_attn": (shared_k, shared_v),
            "model.layers.1.self_attn": (shared_k, shared_v),
        }

        worker.register_kv_caches(kv_caches)
        self.assertEqual(worker.state_group_ids, [])
        self.assertEqual(len(worker.m_store.registered_ptrs), 2)

        worker = self._build_worker(compress_ratios=[4, 4])
        state_tensor = torch.zeros((2, 16, 8), dtype=torch.float32)
        kv_states = {
            "model.layers.0.self_attn": (state_tensor,),
            "model.layers.1.self_attn": (state_tensor,),
        }
        worker.register_kv_caches(kv_caches, kv_states=kv_states)
        self.assertEqual(worker.state_group_ids, [0])
        self.assertEqual(len(worker.m_store.registered_ptrs), 3)

    def test_register_kv_caches_uses_kernel_num_blocks_and_scale(self):
        worker = self._build_worker()
        scaled_k = torch.zeros((4, 16, 1, 8), dtype=torch.float16)
        scaled_v = torch.zeros((4, 16, 1, 8), dtype=torch.float16)
        kv_caches = {
            "model.layers.0.self_attn": (scaled_k, scaled_v),
            "model.layers.1.self_attn": (scaled_k, scaled_v),
        }

        worker.register_kv_caches(kv_caches)

        self.assertEqual(worker.num_blocks, 2)
        self.assertEqual(worker.block_size_scale, [2])
        self.assertEqual(worker.group_kv_block_size_scale[0], [2, 2])
        expected_region_len = scaled_k.shape[0] * scaled_k[0].numel() * scaled_k.element_size()
        self.assertEqual(worker.m_store.registered_lengths, [expected_region_len, expected_region_len])

    def test_register_kv_caches_uses_per_tensor_dtype_for_flat_block_len(self):
        worker = self._build_worker()
        worker.use_sparse = True
        k_cache = torch.zeros((2, 16, 1, 8), dtype=torch.float16)
        scale_cache = torch.zeros((2, 16, 1, 8), dtype=torch.int8)
        kv_caches = {
            "model.layers.0.self_attn": (k_cache, scale_cache),
            "model.layers.1.self_attn": (k_cache, scale_cache),
        }

        worker.register_kv_caches(kv_caches)

        self.assertEqual(
            worker.block_len,
            [
                k_cache[0].numel() * k_cache.element_size(),
                scale_cache[0].numel() * scale_cache.element_size(),
            ],
        )

    def test_group_num_kv_heads_treats_mamba_and_state_as_replicated(self):
        worker = self._build_worker()
        worker.num_kv_head = 8
        worker.group_uses_align_state = [False, True]
        worker.use_mla = False
        worker.use_sparse = False

        self.assertEqual(worker._get_group_num_kv_heads(0, "kv"), 8)
        self.assertEqual(worker._get_group_num_kv_heads(1, "kv"), 1)
        self.assertEqual(worker._get_group_num_kv_heads(0, "state"), 1)

        worker.use_sparse = True
        self.assertEqual(worker._get_group_num_kv_heads(0, "kv"), 1)

    def test_register_kv_caches_accepts_single_tensor_values(self):
        worker = self._build_worker()
        single_cache = torch.zeros((2, 16, 1, 8), dtype=torch.float16)
        kv_caches = {
            "model.layers.0.self_attn": single_cache,
            "model.layers.1.self_attn": single_cache,
        }

        worker.register_kv_caches(kv_caches)

        self.assertEqual(worker.group_block_len[0], [single_cache[0].numel() * single_cache.element_size()])
        self.assertEqual(worker.group_kv_block_size_scale[0], [1])


if __name__ == "__main__":
    unittest.main()
