"""
Example usage of the Transformer Calculator

This script demonstrates how to use the transformer calculator with different models
and configurations.
"""

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode,
    create_qwen3_omni_config
)


def example_qwen3_omni():
    """Example with Qwen3-Omni-30B-A3B-Instruct model"""
    print("=" * 80)
    print("QWEN3-OMNI-30B-A3B-INSTRUCT ANALYSIS")
    print("=" * 80)
    
    # Create configuration
    config = create_qwen3_omni_config()
    calculator = TransformerCalculator(config)
    
    # Analyze different operation modes
    modes = [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]
    
    for mode in modes:
        print(f"\n{mode.value.upper()} MODE ANALYSIS")
        print("-" * 40)
        
        # Memory analysis
        print(calculator.generate_table(mode, "memory"))
        
        # FLOPS analysis
        print(calculator.generate_table(mode, "flops"))


def example_decode_len_analysis():
    """Example showing decode length impact on memory and FLOPS"""
    print("\n" + "=" * 80)
    print("DECODE LENGTH IMPACT ANALYSIS")
    print("=" * 80)
    
    base_config = ModelConfig(
        num_layers=12,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=50257,
        model_type=ModelType.DENSE,
        sequence_length=1024,
        batch_size=1,
        dtype_bytes=2
    )
    
    decode_lens = [32, 64, 128, 256, 512]
    
    print(f"{'Decode Length':<15} {'Memory (MB)':<15} {'FLOPS':<20}")
    print("-" * 55)
    
    for decode_len in decode_lens:
        config = ModelConfig(
            num_layers=base_config.num_layers,
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            vocab_size=base_config.vocab_size,
            model_type=base_config.model_type,
            sequence_length=base_config.sequence_length,
            batch_size=base_config.batch_size,
            decode_len=decode_len,
            dtype_bytes=base_config.dtype_bytes
        )
        
        calculator = TransformerCalculator(config)
        
        # Get memory and FLOPS for decode
        memory = calculator.calculate_memory(OperationMode.DECODE)
        flops = calculator.calculate_flops(OperationMode.DECODE)
        
        memory_mb = memory.total / (1024 * 1024)
        flops_str = f"{flops.total:.2e}"
        
        print(f"{decode_len:<15} {memory_mb:<15.2f} {flops_str:<20}")


def example_custom_dense_model():
    """Example with custom dense model"""
    print("\n" + "=" * 80)
    print("CUSTOM DENSE MODEL ANALYSIS")
    print("=" * 80)
    
    # Create custom configuration for a dense model
    config = ModelConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,  # Dense attention
        intermediate_size=8192,
        vocab_size=50257,
        model_type=ModelType.DENSE,
        sequence_length=4096,
        batch_size=4,
        dtype_bytes=2  # BF16
    )
    
    calculator = TransformerCalculator(config)
    
    # Analyze prefill mode
    print(calculator.generate_table(OperationMode.PREFILL, "memory"))
    print(calculator.generate_table(OperationMode.PREFILL, "flops"))


def example_custom_moe_model():
    """Example with custom MoE model"""
    print("\n" + "=" * 80)
    print("CUSTOM MOE MODEL ANALYSIS")
    print("=" * 80)
    
    # Create custom MoE configuration
    config = ModelConfig(
        num_layers=16,
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,  # GQA
        intermediate_size=12288,
        vocab_size=32000,
        model_type=ModelType.MOE,
        num_experts=16,
        expert_capacity_factor=1.25,
        top_k=4,
        sequence_length=8192,
        batch_size=2,
        dtype_bytes=2  # BF16
    )
    
    calculator = TransformerCalculator(config)
    
    # Analyze decode mode
    print(calculator.generate_table(OperationMode.DECODE, "memory"))
    print(calculator.generate_table(OperationMode.DECODE, "flops"))


def example_parameter_sweep():
    """Example showing parameter sweep for different sequence lengths"""
    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH SWEEP ANALYSIS")
    print("=" * 80)
    
    base_config = ModelConfig(
        num_layers=12,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=50257,
        model_type=ModelType.DENSE,
        batch_size=1,
        dtype_bytes=2
    )
    
    sequence_lengths = [512, 1024, 2048, 4096, 8192]
    
    print(f"{'Seq Length':<12} {'Memory (MB)':<15} {'FLOPS':<20}")
    print("-" * 50)
    
    for seq_len in sequence_lengths:
        config = ModelConfig(
            num_layers=base_config.num_layers,
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            vocab_size=base_config.vocab_size,
            model_type=base_config.model_type,
            sequence_length=seq_len,
            batch_size=base_config.batch_size,
            dtype_bytes=base_config.dtype_bytes
        )
        
        calculator = TransformerCalculator(config)
        
        # Get memory and FLOPS for prefill
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        flops = calculator.calculate_flops(OperationMode.PREFILL)
        
        memory_mb = memory.total / (1024 * 1024)
        flops_str = f"{flops.total:.2e}"
        
        print(f"{seq_len:<12} {memory_mb:<15.2f} {flops_str:<20}")


def example_batch_size_analysis():
    """Example showing batch size impact"""
    print("\n" + "=" * 80)
    print("BATCH SIZE IMPACT ANALYSIS")
    print("=" * 80)
    
    base_config = ModelConfig(
        num_layers=8,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        vocab_size=50257,
        model_type=ModelType.DENSE,
        sequence_length=1024,
        dtype_bytes=2
    )
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'FLOPS':<20} {'Memory/Token':<15}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        config = ModelConfig(
            num_layers=base_config.num_layers,
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            vocab_size=base_config.vocab_size,
            model_type=base_config.model_type,
            sequence_length=base_config.sequence_length,
            batch_size=batch_size,
            dtype_bytes=base_config.dtype_bytes
        )
        
        calculator = TransformerCalculator(config)
        
        # Get memory and FLOPS for prefill
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        flops = calculator.calculate_flops(OperationMode.PREFILL)
        
        memory_mb = memory.total / (1024 * 1024)
        flops_str = f"{flops.total:.2e}"
        memory_per_token = memory_mb / (batch_size * base_config.sequence_length)
        
        print(f"{batch_size:<12} {memory_mb:<15.2f} {flops_str:<20} {memory_per_token:<15.4f}")


if __name__ == "__main__":
    # Run all examples
    example_qwen3_omni()
    example_decode_len_analysis()
    example_custom_dense_model()
    example_custom_moe_model()
    example_parameter_sweep()
    example_batch_size_analysis()
