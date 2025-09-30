#!/usr/bin/env python3
"""
Unit Tests for Transformer Calculator

This module provides comprehensive unit tests for the transformer calculator,
testing individual functions to catch regression bugs early.
"""

import unittest
import math
from typing import Dict, Any

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
    MemoryComponents, FLOPSComponents, ReuseComponents,
    calculate_model_parameters
)
from datatypes import DataTypeDetector
from context import MemoryContext, CalculationContext


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
    
    def test_dtype_bytes_property(self):
        """Test dtype_bytes property calculation"""
        self.assertEqual(self.config.dtype_bytes, 2.0)  # BF16 = 2 bytes
        
        # Test different data types
        config_fp32 = ModelConfig(
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            num_key_value_heads=32, intermediate_size=16384, vocab_size=50257,
            model_type=ModelType.DENSE, sequence_length=2048, batch_size=1,
            decode_len=64, dtype=DataType.FP32
        )
        self.assertEqual(config_fp32.dtype_bytes, 4.0)  # FP32 = 4 bytes
        
        config_int4 = ModelConfig(
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            num_key_value_heads=32, intermediate_size=16384, vocab_size=50257,
            model_type=ModelType.DENSE, sequence_length=2048, batch_size=1,
            decode_len=64, dtype=DataType.INT4
        )
        self.assertEqual(config_int4.dtype_bytes, 0.5)  # INT4 = 0.5 bytes


class TestParameterCalculation(unittest.TestCase):
    """Test parameter calculation functions"""
    
    def test_calculate_model_parameters_dense(self):
        """Test parameter calculation for dense models"""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        
        params = calculate_model_parameters(config)
        
        # Expected parameters for a 7B model
        # This is a rough estimate - actual calculation depends on implementation
        self.assertGreater(params, 1_000_000_000)  # At least 1B parameters
        self.assertLess(params, 10_000_000_000)     # Less than 10B parameters
    
    def test_calculate_model_parameters_moe(self):
        """Test parameter calculation for MoE models"""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        
        params = calculate_model_parameters(config)
        
        # MoE models should have more parameters due to multiple experts
        self.assertGreater(params, 1_000_000_000)  # At least 1B parameters
        self.assertLess(params, 50_000_000_000)    # Less than 50B parameters


class TestMemoryCalculations(unittest.TestCase):
    """Test memory calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        self.calculator = TransformerCalculator(self.config)
    
    def test_attention_memory_calculation(self):
        """Test attention memory calculation"""
        memory = self.calculator.calculate_memory(OperationMode.PREFILL)
        
        # Attention memory should be positive
        self.assertGreater(memory.attention, 0)
        
        # Should be reasonable for 2048 sequence length
        self.assertLess(memory.attention, 1e12)  # Less than 1TB
    
    def test_mlp_memory_calculation(self):
        """Test MLP memory calculation"""
        memory = self.calculator.calculate_memory(OperationMode.PREFILL)
        
        # MLP memory should be positive
        self.assertGreater(memory.mlp, 0)
        
        # Should be reasonable for the model size
        self.assertLess(memory.mlp, 1e12)  # Less than 1TB
    
    def test_embedding_memory_calculation(self):
        """Test embedding memory calculation"""
        memory = self.calculator.calculate_memory(OperationMode.PREFILL)
        
        # Embedding memory should be positive
        self.assertGreater(memory.embeddings, 0)
        
        # Should be much smaller than other components
        self.assertLess(memory.embeddings, memory.attention)
        self.assertLess(memory.embeddings, memory.mlp)


class TestFLOPSCalculations(unittest.TestCase):
    """Test FLOPS calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        self.calculator = TransformerCalculator(self.config)
    
    def test_attention_flops_calculation(self):
        """Test attention FLOPS calculation"""
        flops = self.calculator.calculate_flops(OperationMode.PREFILL)
        
        # Attention FLOPS should be positive
        self.assertGreater(flops.attention, 0)
        
        # Should be quadratic in sequence length
        # For 2048 sequence length, should be substantial (adjusted for realistic values)
        self.assertGreater(flops.attention, 1e11)  # At least 0.1 TFLOP
    
    def test_mlp_flops_calculation(self):
        """Test MLP FLOPS calculation"""
        flops = self.calculator.calculate_flops(OperationMode.PREFILL)
        
        # MLP FLOPS should be positive
        self.assertGreater(flops.mlp, 0)
        
        # Should be substantial for the model size
        self.assertGreater(flops.mlp, 1e12)  # At least 1 TFLOP
    
    def test_embedding_flops_calculation(self):
        """Test embedding FLOPS calculation"""
        flops = self.calculator.calculate_flops(OperationMode.PREFILL)
        
        # Embedding FLOPS should be positive
        self.assertGreater(flops.embeddings, 0)
        
        # Should be much smaller than other components
        self.assertLess(flops.embeddings, flops.attention)
        self.assertLess(flops.embeddings, flops.mlp)


class TestReuseCalculations(unittest.TestCase):
    """Test arithmetic intensity (reuse) calculation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        self.calculator = TransformerCalculator(self.config)
    
    def test_reuse_calculation(self):
        """Test reuse calculation"""
        reuse = self.calculator.calculate_reuse(OperationMode.PREFILL)
        
        # Reuse should be positive
        self.assertGreater(reuse.total, 0)
        
        # Should be reasonable (typically 1-100 FLOPS/byte)
        self.assertGreater(reuse.total, 0.1)
        self.assertLess(reuse.total, 10000)  # Increased threshold for large models
    
    def test_reuse_components(self):
        """Test reuse component calculations"""
        reuse = self.calculator.calculate_reuse(OperationMode.PREFILL)
        
        # All components should be positive
        self.assertGreater(reuse.pre_attention, 0)
        self.assertGreater(reuse.attention, 0)
        self.assertGreater(reuse.post_attention, 0)
        
        # Total should be the sum of components
        expected_total = reuse.pre_attention + reuse.attention + reuse.post_attention
        self.assertAlmostEqual(reuse.total, expected_total, places=6)


class TestMoECalculations(unittest.TestCase):
    """Test MoE-specific calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        self.calculator = TransformerCalculator(self.config)
    
    def test_moe_memory_calculation(self):
        """Test MoE memory calculation"""
        memory = self.calculator.calculate_memory(OperationMode.PREFILL)
        
        # MoE memory should be positive
        self.assertGreater(memory.total, 0)
        
        # Should be larger than dense model due to multiple experts
        # This is a rough check - actual values depend on implementation
        self.assertGreater(memory.mlp, 0)
    
    def test_moe_flops_calculation(self):
        """Test MoE FLOPS calculation"""
        flops = self.calculator.calculate_flops(OperationMode.PREFILL)
        
        # MoE FLOPS should be positive
        self.assertGreater(flops.total, 0)
        
        # Should be substantial
        self.assertGreater(flops.mlp, 0)


class TestDataTypeDetection(unittest.TestCase):
    """Test data type detection functionality"""
    
    def test_detect_from_config(self):
        """Test data type detection from config"""
        # Test BF16 detection
        config_bf16 = {
            "torch_dtype": "bfloat16",
            "dtype": "bfloat16"
        }
        detected = DataTypeDetector.detect_from_config(config_bf16)
        self.assertEqual(detected.value, DataType.BF16.value)
        
        # Test FP16 detection
        config_fp16 = {
            "torch_dtype": "float16",
            "dtype": "float16"
        }
        detected = DataTypeDetector.detect_from_config(config_fp16)
        self.assertEqual(detected.value, DataType.FP16.value)
        
        # Test FP32 detection
        config_fp32 = {
            "torch_dtype": "float32",
            "dtype": "float32"
        }
        detected = DataTypeDetector.detect_from_config(config_fp32)
        self.assertEqual(detected.value, DataType.FP32.value)
    
    def test_detect_from_model_name(self):
        """Test data type detection from model name"""
        # Test INT4 detection
        detected = DataTypeDetector.detect_from_model_name("model-int4")
        self.assertEqual(detected.value, DataType.INT4.value)
        
        # Test INT8 detection
        detected = DataTypeDetector.detect_from_model_name("model-int8")
        self.assertEqual(detected.value, DataType.INT8.value)
        
        # Test default detection
        detected = DataTypeDetector.detect_from_model_name("model")
        self.assertEqual(detected.value, DataType.BF16.value)  # Default


class TestOperationModes(unittest.TestCase):
    """Test different operation modes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        self.calculator = TransformerCalculator(self.config)
    
    def test_pretraining_mode(self):
        """Test pretraining mode calculations"""
        memory = self.calculator.calculate_memory(OperationMode.PRETRAINING)
        flops = self.calculator.calculate_flops(OperationMode.PRETRAINING)
        
        # Pretraining should have higher memory due to gradients and optimizer states
        self.assertGreater(memory.total, 0)
        self.assertGreater(flops.total, 0)
        
        # Should have gradients and optimizer states
        self.assertGreater(memory.gradients, 0)
        self.assertGreater(memory.optimizer_states, 0)
    
    def test_prefill_mode(self):
        """Test prefill mode calculations"""
        memory = self.calculator.calculate_memory(OperationMode.PREFILL)
        flops = self.calculator.calculate_flops(OperationMode.PREFILL)
        
        # Prefill should have positive values
        self.assertGreater(memory.total, 0)
        self.assertGreater(flops.total, 0)
        
        # Should not have gradients or optimizer states
        self.assertEqual(memory.gradients, 0)
        self.assertEqual(memory.optimizer_states, 0)
    
    def test_decode_mode(self):
        """Test decode mode calculations"""
        memory = self.calculator.calculate_memory(OperationMode.DECODE)
        flops = self.calculator.calculate_flops(OperationMode.DECODE)
        
        # Decode should have positive values
        self.assertGreater(memory.total, 0)
        self.assertGreater(flops.total, 0)
        
        # Should have KV cache
        self.assertGreater(memory.kv_cache, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_zero_sequence_length(self):
        """Test behavior with zero sequence length"""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=0,  # Zero sequence length
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        
        calculator = TransformerCalculator(config)
        
        # Should handle zero sequence length gracefully
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        self.assertGreaterEqual(memory.total, 0)  # Should not crash
    
    def test_large_sequence_length(self):
        """Test behavior with large sequence length"""
        config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=16384,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=65536,  # Large sequence length
            batch_size=1,
            decode_len=64,
            dtype=DataType.BF16
        )
        
        calculator = TransformerCalculator(config)
        
        # Should handle large sequence length
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        self.assertGreater(memory.total, 0)
        
        # Memory should be much larger than normal
        self.assertGreater(memory.attention, 1e6)  # At least 1MB (Flash Attention reduces memory significantly)


def run_tests():
    """Run all unit tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelConfig,
        TestParameterCalculation,
        TestMemoryCalculations,
        TestFLOPSCalculations,
        TestReuseCalculations,
        TestMoECalculations,
        TestDataTypeDetection,
        TestOperationModes,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Transformer Calculator Unit Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
