import math

try:
    # When run as a module within the project
    from src.transformer_calculator import TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
except ImportError:
    # Fallback when tests are run from repo root directly
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from transformer_calculator import TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType


def make_base_config(seq_len: int = 2048, batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=8192,
        vocab_size=50257,
        model_type=ModelType.DENSE,
        sequence_length=seq_len,
        batch_size=batch_size,
        decode_len=1,
        dtype=DataType.BF16
    )


def get_kv_bytes(memory_components) -> float:
    return memory_components.kv_cache


def test_kv_cache_increases_with_seq_len_in_decode():
    # Same decode_len, increasing seq_len should increase KV cache
    cfg_small = make_base_config(seq_len=1024, batch_size=2)
    cfg_small.decode_len = 128
    kv_small = get_kv_bytes(TransformerCalculator(cfg_small).calculate_memory(OperationMode.DECODE))

    cfg_large = make_base_config(seq_len=4096, batch_size=2)
    cfg_large.decode_len = 128
    kv_large = get_kv_bytes(TransformerCalculator(cfg_large).calculate_memory(OperationMode.DECODE))

    assert kv_large > kv_small, f"KV cache should grow with seq_len: {kv_small} -> {kv_large}"


def test_kv_cache_increases_with_decode_len_in_decode():
    # Same seq_len, increasing decode_len should increase KV cache (last-token effective length)
    cfg_short = make_base_config(seq_len=2048, batch_size=2)
    cfg_short.decode_len = 64
    kv_short = get_kv_bytes(TransformerCalculator(cfg_short).calculate_memory(OperationMode.DECODE))

    cfg_long = make_base_config(seq_len=2048, batch_size=2)
    cfg_long.decode_len = 1024
    kv_long = get_kv_bytes(TransformerCalculator(cfg_long).calculate_memory(OperationMode.DECODE))

    assert kv_long > kv_short, f"KV cache should grow with decode_len: {kv_short} -> {kv_long}"


