"""
Validation Script for Transformer Calculator

This script validates all calculations against the equations in Math_Used.md
and identifies any discrepancies between the code and documentation.
"""

import sys
from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
)


def validate_attention_memory():
    """Validate attention memory calculations"""
    print("🔍 Validating Attention Memory Calculations")
    print("=" * 50)
    
    # Test configuration
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    
    # Expected values from Math_Used.md
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    dtype_bytes = config.dtype_bytes
    
    # Math_Used.md equation: seq_len² × batch_size × num_heads × dtype_bytes
    expected_attention_scores = seq_len * seq_len * batch_size * num_heads * dtype_bytes
    expected_attention_output = seq_len * batch_size * hidden_size * dtype_bytes
    expected_attention_memory = expected_attention_scores + expected_attention_output
    
    print(f"Configuration:")
    print(f"  seq_len: {seq_len}")
    print(f"  batch_size: {batch_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  dtype_bytes: {dtype_bytes}")
    
    print(f"\nExpected (from Math_Used.md):")
    print(f"  Attention Scores: {expected_attention_scores/1e9:.2f} GB")
    print(f"  Attention Output: {expected_attention_output/1e9:.2f} GB")
    print(f"  Total Attention: {expected_attention_memory/1e9:.2f} GB")
    
    print(f"\nActual (from code):")
    print(f"  Attention Memory: {memory.attention/1e9:.2f} GB")
    
    # Check if they match
    if abs(memory.attention - expected_attention_memory) < 1e6:  # Within 1MB tolerance
        print("✅ Attention memory calculation matches Math_Used.md")
    else:
        print("❌ Attention memory calculation does NOT match Math_Used.md")
        print(f"   Difference: {abs(memory.attention - expected_attention_memory)/1e9:.2f} GB")
    
    return abs(memory.attention - expected_attention_memory) < 1e6


def validate_projection_memory():
    """Validate projection memory calculations"""
    print("\n🔍 Validating Projection Memory Calculations")
    print("=" * 50)
    
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    
    # Math_Used.md equation: 4 × seq_len × batch_size × hidden_size × dtype_bytes
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    dtype_bytes = config.dtype_bytes
    
    expected_projection_memory = 4 * seq_len * batch_size * hidden_size * dtype_bytes
    
    print(f"Expected (from Math_Used.md):")
    print(f"  Projection Memory: {expected_projection_memory/1e9:.2f} GB")
    
    print(f"Actual (from code):")
    print(f"  Projection Memory: {memory.projections/1e9:.2f} GB")
    
    if abs(memory.projections - expected_projection_memory) < 1e6:
        print("✅ Projection memory calculation matches Math_Used.md")
    else:
        print("❌ Projection memory calculation does NOT match Math_Used.md")
        print(f"   Difference: {abs(memory.projections - expected_projection_memory)/1e9:.2f} GB")
    
    return abs(memory.projections - expected_projection_memory) < 1e6


def validate_mlp_memory():
    """Validate MLP memory calculations"""
    print("\n🔍 Validating MLP Memory Calculations")
    print("=" * 50)
    
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    
    # Math_Used.md equation: seq_len × batch_size × (intermediate_size + hidden_size) × dtype_bytes
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    dtype_bytes = config.dtype_bytes
    
    expected_mlp_memory = seq_len * batch_size * (intermediate_size + hidden_size) * dtype_bytes
    
    print(f"Expected (from Math_Used.md):")
    print(f"  MLP Memory: {expected_mlp_memory/1e9:.2f} GB")
    
    print(f"Actual (from code):")
    print(f"  MLP Memory: {memory.mlp/1e9:.2f} GB")
    
    if abs(memory.mlp - expected_mlp_memory) < 1e6:
        print("✅ MLP memory calculation matches Math_Used.md")
    else:
        print("❌ MLP memory calculation does NOT match Math_Used.md")
        print(f"   Difference: {abs(memory.mlp - expected_mlp_memory)/1e9:.2f} GB")
    
    return abs(memory.mlp - expected_mlp_memory) < 1e6


def validate_embedding_memory():
    """Validate embedding memory calculations"""
    print("\n🔍 Validating Embedding Memory Calculations")
    print("=" * 50)
    
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    
    # Math_Used.md equation: input_embeddings + positional_embeddings
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    dtype_bytes = config.dtype_bytes
    
    input_embeddings = seq_len * batch_size * hidden_size * dtype_bytes
    positional_embeddings = seq_len * hidden_size * dtype_bytes
    expected_embedding_memory = input_embeddings + positional_embeddings
    
    print(f"Expected (from Math_Used.md):")
    print(f"  Input Embeddings: {input_embeddings/1e9:.2f} GB")
    print(f"  Positional Embeddings: {positional_embeddings/1e9:.2f} GB")
    print(f"  Total Embedding: {expected_embedding_memory/1e9:.2f} GB")
    
    print(f"Actual (from code):")
    print(f"  Embedding Memory: {memory.embeddings/1e9:.2f} GB")
    
    if abs(memory.embeddings - expected_embedding_memory) < 1e6:
        print("✅ Embedding memory calculation matches Math_Used.md")
    else:
        print("❌ Embedding memory calculation does NOT match Math_Used.md")
        print(f"   Difference: {abs(memory.embeddings - expected_embedding_memory)/1e9:.2f} GB")
    
    return abs(memory.embeddings - expected_embedding_memory) < 1e6


def validate_layer_norm_memory():
    """Validate layer norm memory calculations"""
    print("\n🔍 Validating Layer Norm Memory Calculations")
    print("=" * 50)
    
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    
    # Math_Used.md equation: 2 × seq_len × batch_size × hidden_size × dtype_bytes
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    dtype_bytes = config.dtype_bytes
    
    expected_layer_norm_memory = 2 * seq_len * batch_size * hidden_size * dtype_bytes
    
    print(f"Expected (from Math_Used.md):")
    print(f"  Layer Norm Memory: {expected_layer_norm_memory/1e9:.2f} GB")
    
    print(f"Actual (from code):")
    print(f"  Layer Norm Memory: {memory.layer_norm/1e9:.2f} GB")
    
    if abs(memory.layer_norm - expected_layer_norm_memory) < 1e6:
        print("✅ Layer norm memory calculation matches Math_Used.md")
    else:
        print("❌ Layer norm memory calculation does NOT match Math_Used.md")
        print(f"   Difference: {abs(memory.layer_norm - expected_layer_norm_memory)/1e9:.2f} GB")
    
    return abs(memory.layer_norm - expected_layer_norm_memory) < 1e6


def validate_flops_calculations():
    """Validate FLOPS calculations"""
    print("\n🔍 Validating FLOPS Calculations")
    print("=" * 50)
    
    config = ModelConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        model_type=ModelType.DENSE,
        dtype=DataType.BF16,
        sequence_length=2048,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    flops = calculator.calculate_flops(OperationMode.PREFILL)
    
    # Math_Used.md equations
    seq_len = config.sequence_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_per_head = hidden_size // num_heads
    
    # Attention FLOPS
    attention_scores_flops = seq_len * seq_len * batch_size * num_heads * hidden_per_head
    attention_output_flops = seq_len * batch_size * hidden_size * hidden_size
    expected_attention_flops = attention_scores_flops + attention_output_flops
    
    # Projection FLOPS
    expected_projection_flops = 4 * seq_len * batch_size * hidden_size * hidden_size
    
    # MLP FLOPS
    up_proj_flops = seq_len * batch_size * hidden_size * intermediate_size
    down_proj_flops = seq_len * batch_size * intermediate_size * hidden_size
    expected_mlp_flops = up_proj_flops + down_proj_flops
    
    # Layer Norm FLOPS
    expected_layer_norm_flops = 2 * seq_len * batch_size * hidden_size
    
    print(f"Expected (from Math_Used.md):")
    print(f"  Attention FLOPS: {expected_attention_flops/1e12:.2f} TFLOP")
    print(f"  Projection FLOPS: {expected_projection_flops/1e12:.2f} TFLOP")
    print(f"  MLP FLOPS: {expected_mlp_flops/1e12:.2f} TFLOP")
    print(f"  Layer Norm FLOPS: {expected_layer_norm_flops/1e12:.2f} TFLOP")
    
    print(f"Actual (from code):")
    print(f"  Attention FLOPS: {flops.attention/1e12:.2f} TFLOP")
    print(f"  Projection FLOPS: {flops.projections/1e12:.2f} TFLOP")
    print(f"  MLP FLOPS: {flops.mlp/1e12:.2f} TFLOP")
    print(f"  Layer Norm FLOPS: {flops.layer_norm/1e12:.2f} TFLOP")
    
    # Check each component
    results = []
    for name, expected, actual in [
        ("Attention", expected_attention_flops, flops.attention),
        ("Projection", expected_projection_flops, flops.projections),
        ("MLP", expected_mlp_flops, flops.mlp),
        ("Layer Norm", expected_layer_norm_flops, flops.layer_norm)
    ]:
        if abs(actual - expected) < 1e9:  # Within 1 GFLOP tolerance
            print(f"✅ {name} FLOPS calculation matches Math_Used.md")
            results.append(True)
        else:
            print(f"❌ {name} FLOPS calculation does NOT match Math_Used.md")
            print(f"   Difference: {abs(actual - expected)/1e12:.2f} TFLOP")
            results.append(False)
    
    return all(results)


def main():
    """Run all validation checks"""
    print("🧪 Transformer Calculator Validation")
    print("=" * 60)
    print("Validating calculations against Math_Used.md equations...")
    
    results = []
    
    # Run all validations
    results.append(validate_attention_memory())
    results.append(validate_projection_memory())
    results.append(validate_mlp_memory())
    results.append(validate_embedding_memory())
    results.append(validate_layer_norm_memory())
    results.append(validate_flops_calculations())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All calculations match Math_Used.md!")
    elif passed >= total * 0.8:
        print("✅ Most calculations match Math_Used.md!")
    else:
        print("⚠️  Some calculations do not match Math_Used.md!")
        print("   Review the discrepancies above and update either code or documentation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
