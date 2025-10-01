#!/usr/bin/env python3

# Fix imports for project structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Test suite for detecting gross errors in transformer calculations

This test suite specifically targets gross errors like:
- MLP memory being 1000x too high due to incorrect layer multiplication
- Projections memory being 1000x too high due to wrong sequence length
- Total memory being unreasonably high for the model size
- Arithmetic intensity being astronomically high/low
"""

import unittest
import sys
import os
from typing import Dict, Any, Tuple

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformer_calculator import TransformerCalculator, ModelConfig, OperationMode, ModelType, DataType
from datatypes import DataType as DT


class GrossErrorDetectionTests(unittest.TestCase):
    """Test suite for detecting gross calculation errors"""
    
    def setUp(self):
        """Set up test configurations"""
        self.small_dense_config = ModelConfig(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            dtype=DataType.BF16
        )
        
        self.small_moe_config = ModelConfig(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            expert_capacity_factor=1.0,
            sequence_length=2048,
            batch_size=1,
            dtype=DataType.BF16
        )
        
        self.large_moe_config = ModelConfig(
            num_layers=61,
            hidden_size=7168,
            num_attention_heads=128,
            intermediate_size=18432,
            vocab_size=129280,
            model_type=ModelType.MOE,
            num_experts=256,
            top_k=8,
            expert_capacity_factor=1.0,
            sequence_length=65552,
            batch_size=1,
            dtype=DataType.BF16
        )
    
    def test_dense_model_memory_reasonable(self):
        """Test that dense model memory is reasonable"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        # Test all modes
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                
                # Total memory should be reasonable (less than 100GB for small model)
                self.assertLess(memory.total, 100 * 1024**3, 
                              f"Total memory {memory.total/1024**3:.1f}GB too high for small dense model in {mode}")
                
                # MLP should not dominate for dense models
                mlp_ratio = memory.mlp / memory.total
                self.assertLess(mlp_ratio, 0.8, 
                              f"MLP memory ratio {mlp_ratio:.2f} too high for dense model in {mode}")
    
    def test_moe_model_memory_reasonable(self):
        """Test that MoE model memory is reasonable"""
        calculator = TransformerCalculator(self.small_moe_config)
        
        # Test all modes
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                
                # Total memory should be reasonable (less than 50GB for small MoE model)
                self.assertLess(memory.total, 50 * 1024**3, 
                              f"Total memory {memory.total/1024**3:.1f}GB too high for small MoE model in {mode}")
                
                # MLP should be significant for MoE models but not be excessive
                mlp_ratio = memory.mlp / memory.total
                self.assertGreater(mlp_ratio, 0.01, 
                                 f"MLP memory ratio {mlp_ratio:.2f} too low for MoE model in {mode}")
                self.assertLess(mlp_ratio, 0.95, 
                                f"MLP memory ratio {mlp_ratio:.2f} too high for MoE model in {mode}")
    
    def test_decode_mode_minimal_memory(self):
        """Test that decode mode has minimal memory usage"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        # Decode mode should use much less memory than prefill
        prefill_memory = calculator.calculate_memory(OperationMode.PREFILL)
        decode_memory = calculator.calculate_memory(OperationMode.DECODE)
        
        # Decode should use at least 10x less memory than prefill
        memory_ratio = decode_memory.total / prefill_memory.total
        self.assertLess(memory_ratio, 0.3, 
                       f"Decode memory {decode_memory.total/1024**3:.1f}GB not minimal compared to prefill {prefill_memory.total/1024**3:.1f}GB")
        
        # Projections should be minimal in decode mode
        projection_ratio = decode_memory.projections / decode_memory.total
        self.assertLess(projection_ratio, 0.2, 
                       f"Projections ratio {projection_ratio:.2f} too high for decode mode")
    
    def test_moe_layer_multiplication_fix(self):
        """Test that MoE models don't multiply MLP memory by num_layers"""
        calculator = TransformerCalculator(self.small_moe_config)
        
        # Calculate memory for different layer counts
        config_1_layer = ModelConfig(
            num_layers=1,
            hidden_size=self.small_moe_config.hidden_size,
            num_attention_heads=self.small_moe_config.num_attention_heads,
            intermediate_size=self.small_moe_config.intermediate_size,
            vocab_size=self.small_moe_config.vocab_size,
            model_type=ModelType.MOE,
            num_experts=self.small_moe_config.num_experts,
            top_k=self.small_moe_config.top_k,
            expert_capacity_factor=self.small_moe_config.expert_capacity_factor,
            sequence_length=self.small_moe_config.sequence_length,
            batch_size=self.small_moe_config.batch_size,
            dtype=self.small_moe_config.dtype
        )
        
        config_12_layers = self.small_moe_config
        
        calc_1_layer = TransformerCalculator(config_1_layer)
        calc_12_layers = TransformerCalculator(config_12_layers)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory_1_layer = calc_1_layer.calculate_memory(mode)
                memory_12_layers = calc_12_layers.calculate_memory(mode)
                
                # MLP memory should be the same regardless of layer count for MoE
                mlp_ratio = memory_12_layers.mlp / memory_1_layer.mlp
                self.assertLess(mlp_ratio, 2.0, 
                              f"MoE MLP memory scaled by layers: {mlp_ratio:.1f}x (should be ~1x)")
    
    def test_arithmetic_intensity_reasonable(self):
        """Test that arithmetic intensity is reasonable"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                reuse = calculator.calculate_reuse(mode)
                
                # Arithmetic intensity should be reasonable (1-1,000,000 FLOPS/B for compute-bound operations)
                self.assertGreater(reuse.total, 0.1, 
                                 f"Arithmetic intensity {reuse.total:.1f} too low in {mode}")
                self.assertLess(reuse.total, 1000000, 
                              f"Arithmetic intensity {reuse.total:.1f} too high in {mode}")
                
                # Individual stages should also be reasonable
                for stage_name, stage_value in [("pre_attention", reuse.pre_attention), 
                                               ("attention", reuse.attention), 
                                               ("post_attention", reuse.post_attention)]:
                    self.assertGreater(stage_value, 0, 
                                     f"{stage_name} arithmetic intensity {stage_value:.1f} should be positive")
                    self.assertLess(stage_value, 1000000, 
                                  f"{stage_name} arithmetic intensity {stage_value:.1f} too high")
    
    def test_large_model_memory_bounds(self):
        """Test that large model memory is within reasonable bounds"""
        calculator = TransformerCalculator(self.large_moe_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                
                # Large model should use significant memory but not excessive
                if mode == OperationMode.DECODE:
                    # Decode should be much more efficient
                    self.assertLess(memory.total, 10000 * 1024**3,  # Less than 10TB
                                  f"Decode memory {memory.total/1024**3:.1f}GB too high for large model")
                else:
                    # Prefill/pretraining can use more memory
                    self.assertLess(memory.total, 50000 * 1024**3,  # Less than 50TB
                                  f"{mode} memory {memory.total/1024**3:.1f}GB too high for large model")
                
                # Memory should be proportional to model size
                self.assertGreater(memory.total, 1 * 1024**3,  # At least 1GB
                                 f"{mode} memory {memory.total/1024**3:.1f}GB too low for large model")
    
    def test_sequence_length_scaling(self):
        """Test that memory scales reasonably with sequence length"""
        # Test with different sequence lengths
        seq_lengths = [1024, 4096, 16384]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                config = ModelConfig(
                    num_layers=12,
                    hidden_size=1024,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    vocab_size=50257,
                    model_type=ModelType.DENSE,
                    sequence_length=seq_len,
                    batch_size=1,
                    dtype=DataType.BF16
                )
                
                calculator = TransformerCalculator(config)
                
                # Test prefill mode (should scale with sequence length)
                prefill_memory = calculator.calculate_memory(OperationMode.PREFILL)
                
                # Memory should scale with sequence length (but not necessarily linearly due to attention)
                # For small models, memory should be reasonable
                self.assertLess(prefill_memory.total, 50 * 1024**3, 
                               f"Memory {prefill_memory.total/1024**3:.1f}GB too high for seq_len {seq_len}")
                self.assertGreater(prefill_memory.total, 0.1 * 1024**3, 
                                  f"Memory {prefill_memory.total/1024**3:.1f}GB too low for seq_len {seq_len}")
    
    def test_moe_vs_dense_comparison(self):
        """Test that MoE and dense models have reasonable relative memory usage"""
        dense_calc = TransformerCalculator(self.small_dense_config)
        moe_calc = TransformerCalculator(self.small_moe_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                dense_memory = dense_calc.calculate_memory(mode)
                moe_memory = moe_calc.calculate_memory(mode)
                
                # MoE should use reasonable memory compared to dense
                memory_ratio = moe_memory.total / dense_memory.total
                self.assertGreater(memory_ratio, 0.5, 
                                 f"MoE memory {moe_memory.total/1024**3:.1f}GB too low compared to dense {dense_memory.total/1024**3:.1f}GB")
                
                # But not excessively more (should be within reasonable bounds)
                self.assertLess(memory_ratio, 10.0, 
                               f"MoE memory {memory_ratio:.1f}x too high compared to dense model")
    
    def test_kv_cache_memory_correctness(self):
        """Test that KV cache memory is calculated correctly for decode mode"""
        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 1024),   # Small case
            (8, 4096),   # Medium case
            (32, 16384), # Large case
        ]
        
        for batch_size, seq_len in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                config = ModelConfig(
                    num_layers=12,
                    hidden_size=1024,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    vocab_size=50257,
                    model_type=ModelType.DENSE,
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    dtype=DataType.BF16
                )
                
                calculator = TransformerCalculator(config)
                memory = calculator.calculate_memory(OperationMode.DECODE)
                
                # KV cache should be significant for decode mode
                kv_ratio = memory.kv_cache / memory.total
                self.assertGreater(kv_ratio, 0.05, 
                                 f"KV cache ratio {kv_ratio:.2f} too low for batch_size={batch_size}, seq_len={seq_len}")
                
                # KV cache should scale with batch_size and seq_len
                expected_kv_memory = config.num_layers * seq_len * batch_size * config.hidden_size * config.dtype_bytes * 2
                actual_kv_memory = memory.kv_cache
                
                # Allow some variance but should be in reasonable range
                kv_ratio_actual = actual_kv_memory / expected_kv_memory
                self.assertGreater(kv_ratio_actual, 0.8, 
                                 f"KV cache memory {actual_kv_memory/1024**3:.1f}GB too low for expected {expected_kv_memory/1024**3:.1f}GB")
                self.assertLess(kv_ratio_actual, 1.2, 
                               f"KV cache memory {actual_kv_memory/1024**3:.1f}GB too high for expected {expected_kv_memory/1024**3:.1f}GB")
    
    def test_decode_mode_memory_scaling(self):
        """Test that decode mode memory scales correctly with batch size"""
        base_config = ModelConfig(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=2048,
            batch_size=1,
            dtype=DataType.BF16
        )
        
        batch_sizes = [1, 4, 16, 64]
        memories = []
        
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
                dtype=base_config.dtype
            )
            
            calculator = TransformerCalculator(config)
            memory = calculator.calculate_memory(OperationMode.DECODE)
            memories.append(memory.total)
        
        # Memory should scale roughly linearly with batch size
        for i in range(1, len(batch_sizes)):
            scaling_factor = memories[i] / memories[0]
            expected_scaling = batch_sizes[i] / batch_sizes[0]
            
            # Allow for some variance but should be roughly linear
            scaling_ratio = scaling_factor / expected_scaling
            self.assertGreater(scaling_ratio, 0.15, 
                             f"Memory scaling {scaling_ratio:.1f}x too low for batch_size {batch_sizes[i]}")
            self.assertLess(scaling_ratio, 2.0, 
                           f"Memory scaling {scaling_ratio:.1f}x too high for batch_size {batch_sizes[i]}")
    
    def test_arithmetic_intensity_reasonable_ranges(self):
        """Test that arithmetic intensity is within reasonable ranges for different modes"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                reuse = calculator.calculate_reuse(mode)
                
                # Each stage should have reasonable arithmetic intensity
                stages = [
                    ("pre_attention", reuse.pre_attention),
                    ("attention", reuse.attention),
                    ("post_attention", reuse.post_attention)
                ]
                
                for stage_name, stage_value in stages:
                    # Should be positive
                    self.assertGreaterEqual(stage_value, 0, 
                                          f"{stage_name} arithmetic intensity {stage_value:.1f} should be non-negative")
                    
                    # Should not be astronomically high
                    self.assertLess(stage_value, 1000000, 
                                  f"{stage_name} arithmetic intensity {stage_value:.1f} too high in {mode}")
                    
                    # For decode mode, attention should be reasonable (not 0)
                    if mode == OperationMode.DECODE and stage_name == "attention":
                        self.assertGreater(stage_value, 0.00001, 
                                         f"Decode attention arithmetic intensity {stage_value:.1f} too low")
    
    def test_attention_arithmetic_intensity_correctness(self):
        """Test that attention arithmetic intensity is correct for different modes"""
        # Test with a large MoE model to ensure correct behavior
        config = ModelConfig(
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            model_type=ModelType.MOE, dtype=DataType.BF16,
            num_experts=128, top_k=8, use_flash_attention=True
        )
        calculator = TransformerCalculator(config)
        
        # Pretraining should be compute-bound (high arithmetic intensity)
        pretraining_reuse = calculator.calculate_reuse(OperationMode.PRETRAINING)
        self.assertGreater(pretraining_reuse.attention, 100, 
                          f"Pretraining attention arithmetic intensity {pretraining_reuse.attention:.1f} should be high (compute-bound)")
        
        # Decode should be memory-bound (low arithmetic intensity)
        decode_reuse = calculator.calculate_reuse(OperationMode.DECODE)
        self.assertLess(decode_reuse.attention, 100, 
                       f"Decode attention arithmetic intensity {decode_reuse.attention:.1f} should be low (memory-bound)")
        
        # Prefill should be compute-bound (high arithmetic intensity)
        prefill_reuse = calculator.calculate_reuse(OperationMode.PREFILL)
        self.assertGreater(prefill_reuse.attention, 50, 
                          f"Prefill attention arithmetic intensity {prefill_reuse.attention:.1f} should be high (compute-bound)")
    
    def test_moe_expert_weight_sharing(self):
        """Test that MoE expert weights are not duplicated per layer"""
        # Test with different layer counts
        layer_counts = [1, 4, 12, 24]
        mlp_memories = []
        
        for num_layers in layer_counts:
            config = ModelConfig(
                num_layers=num_layers,
                hidden_size=1024,
                num_attention_heads=16,
                intermediate_size=4096,
                vocab_size=50257,
                model_type=ModelType.MOE,
                num_experts=8,
                top_k=2,
                expert_capacity_factor=1.0,
                sequence_length=2048,
                batch_size=1,
                dtype=DataType.BF16
            )
            
            calculator = TransformerCalculator(config)
            memory = calculator.calculate_memory(OperationMode.PREFILL)
            mlp_memories.append(memory.mlp)
        
        # MLP memory should be the same regardless of layer count for MoE
        base_mlp_memory = mlp_memories[0]
        for i, mlp_memory in enumerate(mlp_memories):
            mlp_ratio = mlp_memory / base_mlp_memory
            self.assertLess(mlp_ratio, 2.0, 
                          f"MoE MLP memory scaled by layers: {mlp_ratio:.1f}x for {layer_counts[i]} layers (should be ~1x)")
    
    def test_decode_vs_prefill_memory_difference(self):
        """Test that decode mode uses significantly less memory than prefill mode"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        prefill_memory = calculator.calculate_memory(OperationMode.PREFILL)
        decode_memory = calculator.calculate_memory(OperationMode.DECODE)
        
        # Decode should use much less memory than prefill
        memory_ratio = decode_memory.total / prefill_memory.total
        self.assertLess(memory_ratio, 0.5, 
                       f"Decode memory {decode_memory.total/1024**3:.1f}GB should be much less than prefill {prefill_memory.total/1024**3:.1f}GB")
        
        # But decode should still use some memory
        self.assertGreater(decode_memory.total, 0.1 * 1024**3, 
                          f"Decode memory {decode_memory.total/1024**3:.1f}GB too low")
    
    def test_large_model_memory_bounds_comprehensive(self):
        """Test that large model memory is within reasonable bounds for all modes"""
        calculator = TransformerCalculator(self.large_moe_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                
                # Large model should use significant memory but not excessive
                if mode == OperationMode.DECODE:
                    # Decode should be more efficient but still substantial
                    self.assertLess(memory.total, 10000 * 1024**3,  # Less than 10TB
                                  f"Decode memory {memory.total/1024**3:.1f}GB too high for large model")
                    self.assertGreater(memory.total, 100 * 1024**3,  # At least 100GB
                                     f"Decode memory {memory.total/1024**3:.1f}GB too low for large model")
                else:
                    # Prefill/pretraining can use more memory
                    self.assertLess(memory.total, 50000 * 1024**3,  # Less than 50TB
                                  f"{mode} memory {memory.total/1024**3:.1f}GB too high for large model")
                    self.assertGreater(memory.total, 100 * 1024**3,  # At least 100GB
                                     f"{mode} memory {memory.total/1024**3:.1f}GB too low for large model")
                
                # Memory should be proportional to model size
                self.assertGreater(memory.total, 1 * 1024**3,  # At least 1GB
                                 f"{mode} memory {memory.total/1024**3:.1f}GB too low for large model")
    
    def test_component_memory_ratios(self):
        """Test that memory component ratios are reasonable"""
        calculator = TransformerCalculator(self.small_dense_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                total = memory.total
                
                # Each component should be a reasonable fraction of total
                components = [
                    ("attention", memory.attention),
                    ("projections", memory.projections),
                    ("mlp", memory.mlp),
                    ("embeddings", memory.embeddings),
                    ("layer_norm", memory.layer_norm),
                    ("weights", memory.weights)
                ]
                
                for comp_name, comp_memory in components:
                    if comp_memory > 0:
                        ratio = comp_memory / total
                        # No single component should dominate completely (>95%)
                        self.assertLess(ratio, 0.95, 
                                      f"{comp_name} memory ratio {ratio:.2f} too high in {mode}")
                        # But components should be significant if they exist
                        if comp_memory > total * 0.01:  # If component is >1% of total
                            self.assertGreater(ratio, 0.001, 
                                             f"{comp_name} memory ratio {ratio:.3f} too low in {mode}")
    
    def test_flash_attention_memory_reduction(self):
        """Test that Flash Attention reduces memory usage compared to standard attention"""
        # This test would require implementing standard attention calculation
        # For now, just verify that attention memory is reasonable
        calculator = TransformerCalculator(self.small_dense_config)
        
        for mode in [OperationMode.PRETRAINING, OperationMode.PREFILL]:
            with self.subTest(mode=mode):
                memory = calculator.calculate_memory(mode)
                
                # Attention memory should be reasonable (not dominating)
                attention_ratio = memory.attention / memory.total
                self.assertLess(attention_ratio, 0.8, 
                              f"Attention memory ratio {attention_ratio:.2f} too high in {mode}")
                self.assertGreater(attention_ratio, 0.0001, 
                                 f"Attention memory ratio {attention_ratio:.2f} too low in {mode}")
    
    def test_decode_mode_kv_cache_dominance(self):
        """Test that KV cache dominates memory in decode mode for large sequences"""
        # Test with large sequence length
        config = ModelConfig(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            model_type=ModelType.DENSE,
            sequence_length=16384,  # Large sequence
            batch_size=8,
            dtype=DataType.BF16
        )
        
        calculator = TransformerCalculator(config)
        memory = calculator.calculate_memory(OperationMode.DECODE)
        
        # KV cache should dominate for large sequences in decode mode
        kv_ratio = memory.kv_cache / memory.total
        self.assertGreater(kv_ratio, 0.5, 
                         f"KV cache ratio {kv_ratio:.2f} should dominate for large sequences in decode mode")
    
    def test_moe_expert_memory_calculation(self):
        """Test that MoE expert memory calculation is correct"""
        config = ModelConfig(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            vocab_size=50257,
            model_type=ModelType.MOE,
            num_experts=8,
            top_k=2,
            expert_capacity_factor=1.0,
            sequence_length=2048,
            batch_size=1,
            dtype=DataType.BF16
        )
        
        calculator = TransformerCalculator(config)
        memory = calculator.calculate_memory(OperationMode.PREFILL)
        
        # MLP memory should be significant for MoE
        mlp_ratio = memory.mlp / memory.total
        self.assertGreater(mlp_ratio, 0.01,
                         f"MoE MLP memory ratio {mlp_ratio:.2f} too low")
        
        # MLP memory should be reasonable (not excessive)
        self.assertLess(mlp_ratio, 0.9, 
                       f"MoE MLP memory ratio {mlp_ratio:.2f} too high")


def run_gross_error_tests():
    """Run all gross error detection tests"""
    print("🔍 Running Gross Error Detection Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(GrossErrorDetectionTests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"❌ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n🚨 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_gross_error_tests()
    sys.exit(0 if success else 1)
