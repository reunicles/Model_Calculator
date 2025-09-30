#!/usr/bin/env python3
"""
Simple test script to verify the transformer calculator works correctly
without requiring Flask or other optional dependencies.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_calculator():
    """Test basic calculator functionality"""
    print("🧪 Testing Basic Calculator Functionality")
    print("=" * 50)
    
    try:
        from transformer_calculator import (
            TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
        )
        
        # Test 1: Dense model
        print("\n📊 Test 1: Dense Model")
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            model_type=ModelType.DENSE,
            dtype=DataType.BF16,
            sequence_length=2048,
            batch_size=1
        )
        
        calculator = TransformerCalculator(config)
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        flops = calculator.calculate_flops(OperationMode.PREFILL)
        reuse = calculator.calculate_reuse(OperationMode.PREFILL)
        
        print(f"   Memory: {memory.total/1e9:.2f} GB")
        print(f"   FLOPS: {flops.total/1e12:.2f} TFLOP")
        print(f"   Reuse: {reuse.total:.1f} FLOPS/byte")
        print("   ✅ Dense model calculation successful")
        
        # Test 2: MoE model
        print("\n📊 Test 2: MoE Model")
        moe_config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            expert_capacity_factor=1.0,
            dtype=DataType.BF16,
            sequence_length=2048,
            batch_size=1
        )
        
        moe_calculator = TransformerCalculator(moe_config)
        moe_memory = moe_calculator.calculate_memory(OperationMode.PREFILL)
        moe_flops = moe_calculator.calculate_flops(OperationMode.PREFILL)
        moe_reuse = moe_calculator.calculate_reuse(OperationMode.PREFILL)
        
        print(f"   Memory: {moe_memory.total/1e9:.2f} GB")
        print(f"   FLOPS: {moe_flops.total/1e12:.2f} TFLOP")
        print(f"   Reuse: {moe_reuse.total:.1f} FLOPS/byte")
        print("   ✅ MoE model calculation successful")
        
        # Test 3: Data types
        print("\n📊 Test 3: Data Types")
        for dtype in [DataType.FP32, DataType.FP16, DataType.BF16, DataType.INT8, DataType.INT4]:
            test_config = ModelConfig(
                num_layers=16,
                hidden_size=2048,
                num_attention_heads=16,
                intermediate_size=5504,
                vocab_size=32000,
                model_type=ModelType.DENSE,
                dtype=dtype,
                sequence_length=1024,
                batch_size=1
            )
            
            test_calc = TransformerCalculator(test_config)
            test_memory = test_calc.calculate_memory(OperationMode.PREFILL)
            
            print(f"   {dtype.value.upper()}: {test_memory.total/1e9:.2f} GB")
        
        print("   ✅ Data type calculations successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_advanced_features():
    """Test advanced features"""
    print("\n🔬 Testing Advanced Features")
    print("=" * 50)
    
    try:
        # Test MoE optimizations
        from moe_optimizations import AdvancedMoECalculator, RoutingStrategy
        from transformer_calculator import ModelConfig, ModelType, DataType, CalculationContext, OperationMode
        
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            expert_capacity_factor=1.0,
            dtype=DataType.BF16,
            sequence_length=2048,
            batch_size=1
        )
        
        # Test MoE calculator
        from transformer_calculator import MoECalculator
        moe_calc = MoECalculator(config)
        
        # Create MemoryContext for MoE calculations
        from context import MemoryContext
        memory_context = MemoryContext(
            seq_len=config.sequence_length,
            batch_size=config.batch_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            dtype_bytes=config.dtype_bytes,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            model_type=config.model_type.value,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_capacity_factor=config.expert_capacity_factor
        )
        
        hbm_memory = moe_calc.calculate_hbm_memory(memory_context)
        gpu_memory = moe_calc.calculate_gpu_memory(memory_context)
        
        print(f"   MoE HBM Memory: {hbm_memory/1e9:.2f} GB")
        print(f"   MoE GPU Memory: {gpu_memory/1e9:.2f} GB")
        print("   ✅ MoE features working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced features error: {e}")
        return False


def test_benchmark_suite():
    """Test benchmark suite"""
    print("\n📈 Testing Benchmark Suite")
    print("=" * 50)
    
    try:
        from benchmark_suite import BenchmarkSuite
        
        suite = BenchmarkSuite()
        suite.create_standard_validation_cases()
        
        # Run a quick test with just one case
        if suite.validation_cases:
            result = suite.run_validation_case(suite.validation_cases[0])
            print(f"   Test: {result.model_name}")
            print(f"   Accuracy: {result.accuracy_score:.2f}")
            print(f"   Memory: {result.memory.total/1e9:.2f} GB")
            print(f"   FLOPS: {result.flops.total/1e12:.2f} TFLOP")
            print("   ✅ Benchmark suite working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Benchmark suite error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Transformer Calculator Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Calculator", test_basic_calculator),
        ("Advanced Features", test_advanced_features),
        ("Benchmark Suite", test_benchmark_suite)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Calculator is working perfectly!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed! Calculator is working well!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
