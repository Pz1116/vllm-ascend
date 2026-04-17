import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.models.deepseek_compressor import CompressorStateCache


def __init__(
    self,
    state_dim: int,
    dtype: torch.dtype,
    compress_ratio: int,
    block_size: int,
    prefix: str,
):
    super().__init__()
    self.state_dim = state_dim
    self.dtype = dtype
    self.prefix = prefix
    self.kv_cache = torch.tensor([])
    compilation_config = get_current_vllm_config().compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self

    assert self.dtype == torch.float32
    assert compress_ratio in [4, 128]
    coff = 1 + (compress_ratio == 4)
    self.sliding_window = coff * compress_ratio

    # ======================GPU======================
    # Block size is constrained by tensor sharing between compressor states
    # and KV blocks. Since compressor states share the same physical tensor
    # as KV blocks, they must use the same page size.
    # The KV block shape [256//4, head_dim] = [64, 584] determines:
    # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
    # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
    # TODO(yifan): make block size automatically determined and configurable.
    # if compress_ratio == 4:
    #     self.block_size = 4
    # elif compress_ratio == 128:
    #     self.block_size = 8
    # else:
    #     raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    self.block_size = block_size


CompressorStateCache.__init__ = __init__