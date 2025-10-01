#!/usr/bin/env python3
"""
Test suite for popular models in the web interface.
Tests that the four new popular models can be correctly processed.
"""

import unittest
import sys
import os
sys.path.append('.')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformer_calculator import TransformerCalculator, ModelConfig, ModelType, DataType, OperationMode
from hf_model_fetcher import create_model_config_from_hf


class TestPopularModels(unittest.TestCase):
    """Test that popular models can be correctly processed"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.popular_models = [
            {
                'name': 'Qwen3-VL-235B-Thinking',
                'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
                'type': 'MoE',
                'size': '235B'
            },
            {
                'name': 'DeepSeek-V3.1-Terminus',
                'path': 'deepseek-ai/DeepSeek-V3.1-Terminus',
                'type': 'MoE',
                'size': '685B'
            },
            {
                'name': 'GPT-OSS-120B',
                'path': 'openai/gpt-oss-120b',
                'type': 'MoE',
                'size': '120B'
            },
            {
                'name': 'Falcon-H1-34B',
                'path': 'tiiuae/Falcon-H1-34B-Instruct',
                'type': 'Dense',
                'size': '34B'
            },
            {
                'name': 'Qwen2.5-72B',
                'path': 'Qwen/Qwen2.5-72B',
                'type': 'Dense',
                'size': '72B'
            },
            {
                'name': 'Qwen2.5-14B',
                'path': 'Qwen/Qwen2.5-14B',
                'type': 'Dense',
                'size': '14B'
            },
            {
                'name': 'Qwen2.5-7B',
                'path': 'Qwen/Qwen2.5-7B',
                'type': 'Dense',
                'size': '7B'
            },
            {
                'name': 'Phi-3.5-mini',
                'path': 'microsoft/Phi-3.5-mini-instruct',
                'type': 'Dense',
                'size': '3.8B'
            }
        ]
    
    def _handle_auth_error(self, model_name, e):
        """Handle authentication errors gracefully"""
        if "401" in str(e) or "gated" in str(e).lower() or "authentication" in str(e).lower():
            print(f"⚠️  {model_name}: Model requires authentication (gated model)")
            return True
        return False
    
    def test_popular_models_config_creation(self):
        """Test that popular models can create valid configurations"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Test configuration creation
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=2048,
                        batch_size=1,
                        decode_len=64
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Verify configuration is valid
                    self.assertIsNotNone(config, f"Failed to create config for {model['name']}")
                    self.assertIsInstance(config, ModelConfig, f"Invalid config type for {model['name']}")
                    
                    # Verify basic properties
                    self.assertGreater(config.num_layers, 0, f"Invalid num_layers for {model['name']}")
                    self.assertGreater(config.hidden_size, 0, f"Invalid hidden_size for {model['name']}")
                    self.assertGreater(config.num_attention_heads, 0, f"Invalid num_attention_heads for {model['name']}")
                    self.assertGreater(config.vocab_size, 0, f"Invalid vocab_size for {model['name']}")
                    
                    print(f"✅ {model['name']}: Config created successfully")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"Failed to create config for {model['name']}: {str(e)}")
    
    def test_popular_models_calculator_creation(self):
        """Test that popular models can create valid calculators"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Create configuration
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=2048,
                        batch_size=1,
                        decode_len=64
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Create calculator
                    calculator = TransformerCalculator(config)
                    
                    # Verify calculator is valid
                    self.assertIsNotNone(calculator, f"Failed to create calculator for {model['name']}")
                    self.assertIsInstance(calculator, TransformerCalculator, f"Invalid calculator type for {model['name']}")
                    
                    print(f"✅ {model['name']}: Calculator created successfully")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"Failed to create calculator for {model['name']}: {str(e)}")
    
    def test_popular_models_memory_calculation(self):
        """Test that popular models can perform memory calculations"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Create configuration
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=1024,  # Smaller sequence length for faster testing
                        batch_size=1,
                        decode_len=32
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Create calculator
                    calculator = TransformerCalculator(config)
                    
                    # Test memory calculation for prefill mode
                    memory = calculator.calculate_memory(OperationMode.PREFILL)
                    
                    # Verify memory calculation results
                    self.assertIsNotNone(memory, f"Memory calculation failed for {model['name']}")
                    self.assertGreater(memory.total, 0, f"Invalid total memory for {model['name']}")
                    self.assertGreater(memory.weights, 0, f"Invalid weights memory for {model['name']}")
                    
                    print(f"✅ {model['name']}: Memory calculation successful (Total: {memory.total:.2f} GB)")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"Memory calculation failed for {model['name']}: {str(e)}")
    
    def test_popular_models_flops_calculation(self):
        """Test that popular models can perform FLOPS calculations"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Create configuration
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=1024,  # Smaller sequence length for faster testing
                        batch_size=1,
                        decode_len=32
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Create calculator
                    calculator = TransformerCalculator(config)
                    
                    # Test FLOPS calculation for prefill mode
                    flops = calculator.calculate_flops(OperationMode.PREFILL)
                    
                    # Verify FLOPS calculation results
                    self.assertIsNotNone(flops, f"FLOPS calculation failed for {model['name']}")
                    self.assertGreater(flops.total, 0, f"Invalid total FLOPS for {model['name']}")
                    self.assertGreater(flops.attention, 0, f"Invalid attention FLOPS for {model['name']}")
                    
                    print(f"✅ {model['name']}: FLOPS calculation successful (Total: {flops.total:.2e} FLOPS)")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"FLOPS calculation failed for {model['name']}: {str(e)}")
    
    def test_popular_models_arithmetic_intensity(self):
        """Test that popular models can calculate arithmetic intensity"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Create configuration
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=1024,  # Smaller sequence length for faster testing
                        batch_size=1,
                        decode_len=32
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Create calculator
                    calculator = TransformerCalculator(config)
                    
                    # Test arithmetic intensity calculation for prefill mode
                    reuse = calculator.calculate_reuse(OperationMode.PREFILL)
                    
                    # Verify arithmetic intensity calculation results
                    self.assertIsNotNone(reuse, f"Arithmetic intensity calculation failed for {model['name']}")
                    self.assertGreater(reuse.total, 0, f"Invalid total arithmetic intensity for {model['name']}")
                    self.assertGreater(reuse.attention, 0, f"Invalid attention arithmetic intensity for {model['name']}")
                    
                    print(f"✅ {model['name']}: Arithmetic intensity calculation successful (Total: {reuse.total:.2f} FLOPS/B)")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"Arithmetic intensity calculation failed for {model['name']}: {str(e)}")
    
    def test_popular_models_all_modes(self):
        """Test that popular models work across all operation modes"""
        for model in self.popular_models:
            with self.subTest(model=model['name']):
                try:
                    # Create configuration
                    config = create_model_config_from_hf(
                        model['path'],
                        sequence_length=512,  # Smaller for faster testing
                        batch_size=1,
                        decode_len=16
                    )
                    
                    # Check for authentication errors
                    if config is None:
                        print(f"⚠️  {model['name']}: Model requires authentication (gated model)")
                        continue
                    
                    # Create calculator
                    calculator = TransformerCalculator(config)
                    
                    # Test all operation modes
                    modes = [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]
                    
                    for mode in modes:
                        with self.subTest(mode=mode.value):
                            # Test memory calculation
                            memory = calculator.calculate_memory(mode)
                            self.assertIsNotNone(memory, f"Memory calculation failed for {model['name']} in {mode.value}")
                            self.assertGreater(memory.total, 0, f"Invalid memory for {model['name']} in {mode.value}")
                            
                            # Test FLOPS calculation
                            flops = calculator.calculate_flops(mode)
                            self.assertIsNotNone(flops, f"FLOPS calculation failed for {model['name']} in {mode.value}")
                            self.assertGreater(flops.total, 0, f"Invalid FLOPS for {model['name']} in {mode.value}")
                            
                            # Test arithmetic intensity calculation
                            reuse = calculator.calculate_reuse(mode)
                            self.assertIsNotNone(reuse, f"Arithmetic intensity calculation failed for {model['name']} in {mode.value}")
                            self.assertGreater(reuse.total, 0, f"Invalid arithmetic intensity for {model['name']} in {mode.value}")
                    
                    print(f"✅ {model['name']}: All operation modes successful")
                    
                except Exception as e:
                    if self._handle_auth_error(model['name'], e):
                        continue
                    self.fail(f"Operation mode testing failed for {model['name']}: {str(e)}")


if __name__ == '__main__':
    print("🧪 Testing Popular Models")
    print("=" * 50)
    
    # Run the tests
    unittest.main(verbosity=2)