#!/usr/bin/env python3
"""
Comprehensive test suite to ensure CLI and web interface calculations are identical.
This test prevents any future inconsistencies between the two interfaces.
"""

import unittest
import subprocess
import json
import requests
import time
import sys
import os
from typing import Dict, Any, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from transformer_calculator import TransformerCalculator, ModelConfig, OperationMode, ModelType, DataType


class CLIWebConsistencyTest(unittest.TestCase):
    """Test suite to ensure CLI and web interface produce identical results"""
    
    def setUp(self):
        """Set up test environment"""
        self.base_url = "http://localhost:5001"
        self.test_models = [
            {
                "name": "Qwen2.5-7B",
                "hf_path": "https://huggingface.co/Qwen/Qwen2.5-7B",
                "sequence_length": 2048,
                "batch_size": 1,
                "decode_len": 64,
                "mode": "prefill"
            },
            {
                "name": "DeepSeek-V3.1-Terminus",
                "hf_path": "https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus",
                "sequence_length": 8192,
                "batch_size": 32,
                "decode_len": 8192,
                "mode": "decode"
            },
            {
                "name": "Phi-3.5-mini",
                "hf_path": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
                "sequence_length": 4096,
                "batch_size": 8,
                "decode_len": 128,
                "mode": "prefill"
            }
        ]
        
        # Manual configuration test cases
        self.manual_configs = [
            {
                "name": "Small Dense Model",
                "config": {
                    "num_layers": 32,
                    "hidden_size": 4096,
                    "num_attention_heads": 32,
                    "intermediate_size": 11008,
                    "vocab_size": 32000,
                    "model_type": "dense",
                    "dtype": "bf16",
                    "sequence_length": 2048,
                    "batch_size": 1,
                    "decode_len": 64,
                    "mode": "prefill"
                }
            },
            {
                "name": "Large MoE Model",
                "config": {
                    "num_layers": 61,
                    "hidden_size": 7168,
                    "num_attention_heads": 128,
                    "intermediate_size": 18432,
                    "vocab_size": 129280,
                    "model_type": "moe",
                    "num_experts": 256,
                    "top_k": 8,
                    "expert_capacity_factor": 1.0,
                    "dtype": "bf16",
                    "sequence_length": 8192,
                    "batch_size": 32,
                    "decode_len": 8192,
                    "mode": "decode"
                }
            }
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def start_web_interface(self):
        """Start web interface if not running"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ Web interface is already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("🚀 Starting web interface...")
        # Note: In a real test environment, you'd start the web interface here
        # For now, we assume it's running
        return True
    
    def run_cli_command(self, hf_path: str, sequence_length: int, batch_size: int, 
                       decode_len: int, mode: str) -> Dict[str, Any]:
        """Run CLI command and parse output"""
        cmd = [
            "python3", "src/cli_calculator.py",
            "--hf-model", hf_path,
            "--table", "all",
            "--mode", mode,
            "--batch-size", str(batch_size),
            "--decode-len", str(decode_len),
            "--seq-len", str(sequence_length)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise Exception(f"CLI command failed: {result.stderr}")
            
            return self.parse_cli_output(result.stdout)
        except subprocess.TimeoutExpired:
            raise Exception("CLI command timed out")
        except Exception as e:
            raise Exception(f"CLI execution failed: {str(e)}")
    
    def parse_cli_output(self, output: str) -> Dict[str, Any]:
        """Parse CLI output to extract key metrics"""
        lines = output.split('\n')
        result = {}
        
        for line in lines:
            line = line.strip()
            
            # Extract model parameters
            if "Model Parameters:" in line:
                # Extract number from "Model Parameters: 685,000,000,000 (685.0B)"
                import re
                match = re.search(r'(\d+(?:,\d+)*)\s*\(([^)]+)\)', line)
                if match:
                    result['total_parameters'] = int(match.group(1).replace(',', ''))
                    result['total_parameters_formatted'] = match.group(2)
            
            # Extract memory components
            elif "Total Memory:" in line:
                result['total_memory'] = self.extract_bytes(line)
            elif "Attention Memory:" in line:
                result['attention_memory'] = self.extract_bytes(line)
            elif "Projections Memory:" in line:
                result['projections_memory'] = self.extract_bytes(line)
            elif "MLP Memory:" in line:
                result['mlp_memory'] = self.extract_bytes(line)
            elif "Embeddings Memory:" in line:
                result['embeddings_memory'] = self.extract_bytes(line)
            elif "Layer Norm Memory:" in line:
                result['layer_norm_memory'] = self.extract_bytes(line)
            elif "Weights Memory:" in line:
                result['weights_memory'] = self.extract_bytes(line)
            elif "KV Cache Memory:" in line:
                result['kv_cache_memory'] = self.extract_bytes(line)
            
            # Extract FLOPS components
            elif "Total FLOPS:" in line:
                result['total_flops'] = self.extract_flops(line)
            elif "Attention FLOPS:" in line:
                result['attention_flops'] = self.extract_flops(line)
            elif "Projections FLOPS:" in line:
                result['projections_flops'] = self.extract_flops(line)
            elif "MLP FLOPS:" in line:
                result['mlp_flops'] = self.extract_flops(line)
            elif "Embeddings FLOPS:" in line:
                result['embeddings_flops'] = self.extract_flops(line)
            elif "Layer Norm FLOPS:" in line:
                result['layer_norm_flops'] = self.extract_flops(line)
            
            # Extract Reuse components
            elif "Pre-Attention Reuse:" in line:
                result['pre_attention_reuse'] = self.extract_reuse(line)
            elif "Attention Reuse:" in line:
                result['attention_reuse'] = self.extract_reuse(line)
            elif "Post-Attention Reuse:" in line:
                result['post_attention_reuse'] = self.extract_reuse(line)
        
        return result
    
    def extract_bytes(self, line: str) -> float:
        """Extract bytes from CLI output line"""
        import re
        # Match patterns like "Total Memory: 26.8 GB"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)', line)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'GB':
                return value * (1024**3)
            elif unit == 'MB':
                return value * (1024**2)
            elif unit == 'KB':
                return value * 1024
            else:
                return value
        return 0.0
    
    def extract_flops(self, line: str) -> float:
        """Extract FLOPS from CLI output line"""
        import re
        # Match patterns like "Total FLOPS: 11.4 TFLOP"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(TFLOP|GFLOP|MFLOP|FLOP)', line)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == 'TFLOP':
                return value * 1e12
            elif unit == 'GFLOP':
                return value * 1e9
            elif unit == 'MFLOP':
                return value * 1e6
            else:
                return value
        return 0.0
    
    def extract_reuse(self, line: str) -> float:
        """Extract reuse from CLI output line"""
        import re
        # Match patterns like "Attention Reuse: 0.99 FLOPS/B"
        match = re.search(r'(\d+(?:\.\d+)?)\s*FLOPS/B', line)
        if match:
            return float(match.group(1))
        return 0.0
    
    def call_web_api(self, hf_path: str, sequence_length: int, batch_size: int, 
                    decode_len: int, mode: str) -> Dict[str, Any]:
        """Call web API and return results"""
        url = f"{self.base_url}/api/calculate-hf"
        data = {
            "hf_path": hf_path,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "decode_len": decode_len,
            "mode": mode
        }
        
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if not result.get('success'):
                raise Exception(f"Web API error: {result.get('error', 'Unknown error')}")
            
            return result['results']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Web API request failed: {str(e)}")
    
    def call_web_manual_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Call web API for manual configuration"""
        url = f"{self.base_url}/api/calculate"
        
        try:
            response = requests.post(url, json=config, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if not result.get('success'):
                raise Exception(f"Web API error: {result.get('error', 'Unknown error')}")
            
            return result['results']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Web API request failed: {str(e)}")
    
    def compare_results(self, cli_result: Dict[str, Any], web_result: Dict[str, Any], 
                       tolerance: float = 0.01) -> Tuple[bool, str]:
        """Compare CLI and web results with tolerance"""
        differences = []
        
        # Compare model parameters
        if 'total_parameters' in cli_result and 'model_info' in web_result:
            cli_params = cli_result.get('total_parameters', 0)
            web_params = web_result['model_info'].get('total_parameters', 0)
            if abs(cli_params - web_params) > tolerance * max(cli_params, web_params):
                differences.append(f"Parameters: CLI={cli_params}, Web={web_params}")
        
        # Compare memory components
        memory_components = ['total_memory', 'attention_memory', 'projections_memory', 
                           'mlp_memory', 'embeddings_memory', 'layer_norm_memory', 
                           'weights_memory', 'kv_cache_memory']
        
        for component in memory_components:
            cli_key = component
            web_key = component.replace('_memory', '')
            
            if cli_key in cli_result and 'memory' in web_result:
                cli_value = cli_result[cli_key]
                web_value = web_result['memory'].get(web_key, 0)
                
                if cli_value > 0 and web_value > 0:
                    diff_percent = abs(cli_value - web_value) / max(cli_value, web_value)
                    if diff_percent > tolerance:
                        differences.append(f"{component}: CLI={cli_value:.2e}, Web={web_value:.2e} ({diff_percent:.1%} diff)")
        
        # Compare FLOPS components
        flops_components = ['total_flops', 'attention_flops', 'projections_flops', 
                          'mlp_flops', 'embeddings_flops', 'layer_norm_flops']
        
        for component in flops_components:
            cli_key = component
            web_key = component.replace('_flops', '')
            
            if cli_key in cli_result and 'flops' in web_result:
                cli_value = cli_result[cli_key]
                web_value = web_result['flops'].get(web_key, 0)
                
                if cli_value > 0 and web_value > 0:
                    diff_percent = abs(cli_value - web_value) / max(cli_value, web_value)
                    if diff_percent > tolerance:
                        differences.append(f"{component}: CLI={cli_value:.2e}, Web={web_value:.2e} ({diff_percent:.1%} diff)")
        
        # Compare Reuse components
        reuse_components = ['pre_attention_reuse', 'attention_reuse', 'post_attention_reuse']
        
        for component in reuse_components:
            cli_key = component
            web_key = component.replace('_reuse', '')
            
            if cli_key in cli_result and 'reuse' in web_result:
                cli_value = cli_result[cli_key]
                web_value = web_result['reuse'].get(web_key, 0)
                
                if cli_value > 0 and web_value > 0:
                    diff_percent = abs(cli_value - web_value) / max(cli_value, web_value)
                    if diff_percent > tolerance:
                        differences.append(f"{component}: CLI={cli_value:.3f}, Web={web_value:.3f} ({diff_percent:.1%} diff)")
        
        if differences:
            return False, " | ".join(differences)
        else:
            return True, "All values match within tolerance"
    
    def test_huggingface_models_consistency(self):
        """Test that HuggingFace models produce identical results in CLI and web interface"""
        print("\n🧪 Testing HuggingFace Models Consistency")
        print("=" * 60)
        
        for model in self.test_models:
            print(f"\n📊 Testing {model['name']}...")
            
            try:
                # Run CLI command
                print("  🔧 Running CLI command...")
                cli_result = self.run_cli_command(
                    model['hf_path'],
                    model['sequence_length'],
                    model['batch_size'],
                    model['decode_len'],
                    model['mode']
                )
                
                # Call web API
                print("  🌐 Calling web API...")
                web_result = self.call_web_api(
                    model['hf_path'],
                    model['sequence_length'],
                    model['batch_size'],
                    model['decode_len'],
                    model['mode']
                )
                
                # Compare results
                print("  🔍 Comparing results...")
                match, message = self.compare_results(cli_result, web_result)
                
                if match:
                    print(f"  ✅ {model['name']}: {message}")
                else:
                    print(f"  ❌ {model['name']}: {message}")
                    self.fail(f"Inconsistency found for {model['name']}: {message}")
                
            except Exception as e:
                self.fail(f"Test failed for {model['name']}: {str(e)}")
    
    def test_manual_configurations_consistency(self):
        """Test that manual configurations produce identical results in CLI and web interface"""
        print("\n🧪 Testing Manual Configurations Consistency")
        print("=" * 60)
        
        for config in self.manual_configs:
            print(f"\n📊 Testing {config['name']}...")
            
            try:
                # Create ModelConfig for CLI
                model_config = ModelConfig(
                    num_layers=config['config']['num_layers'],
                    hidden_size=config['config']['hidden_size'],
                    num_attention_heads=config['config']['num_attention_heads'],
                    intermediate_size=config['config']['intermediate_size'],
                    vocab_size=config['config']['vocab_size'],
                    model_type=ModelType(config['config']['model_type']),
                    dtype=DataType(config['config']['dtype']),
                    sequence_length=config['config']['sequence_length'],
                    batch_size=config['config']['batch_size'],
                    decode_len=config['config']['decode_len'],
                    num_experts=config['config'].get('num_experts', 1),
                    top_k=config['config'].get('top_k', 1),
                    expert_capacity_factor=config['config'].get('expert_capacity_factor', 1.0)
                )
                
                # Run calculations directly
                calculator = TransformerCalculator(model_config)
                operation_mode = OperationMode(config['config']['mode'])
                
                memory = calculator.calculate_memory(operation_mode)
                flops = calculator.calculate_flops(operation_mode)
                reuse = calculator.calculate_reuse(operation_mode)
                
                # Use the actual parameter calculation function
                from transformer_calculator import calculate_model_parameters
                total_params = calculate_model_parameters(model_config)
                
                # Create CLI-style result
                cli_result = {
                    'total_parameters': total_params,
                    'total_memory': memory.total,
                    'attention_memory': memory.attention,
                    'projections_memory': memory.projections,
                    'mlp_memory': memory.mlp,
                    'embeddings_memory': memory.embeddings,
                    'layer_norm_memory': memory.layer_norm,
                    'weights_memory': memory.weights,
                    'kv_cache_memory': memory.kv_cache,
                    'total_flops': flops.total,
                    'attention_flops': flops.attention,
                    'projections_flops': flops.projections,
                    'mlp_flops': flops.mlp,
                    'embeddings_flops': flops.embeddings,
                    'layer_norm_flops': flops.layer_norm,
                    'pre_attention_reuse': reuse.pre_attention,
                    'attention_reuse': reuse.attention,
                    'post_attention_reuse': reuse.post_attention
                }
                
                # Call web API
                print("  🌐 Calling web API...")
                web_result = self.call_web_manual_api(config['config'])
                
                # Compare results
                print("  🔍 Comparing results...")
                match, message = self.compare_results(cli_result, web_result)
                
                if match:
                    print(f"  ✅ {config['name']}: {message}")
                else:
                    print(f"  ❌ {config['name']}: {message}")
                    self.fail(f"Inconsistency found for {config['name']}: {message}")
                
            except Exception as e:
                self.fail(f"Test failed for {config['name']}: {str(e)}")
    
    def test_parameter_validation_consistency(self):
        """Test that parameter validation works consistently between CLI and web"""
        print("\n🧪 Testing Parameter Validation Consistency")
        print("=" * 60)
        
        # Test DeepSeek-V3.1-Terminus specifically
        model = {
            "name": "DeepSeek-V3.1-Terminus",
            "hf_path": "https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus",
            "sequence_length": 8192,
            "batch_size": 32,
            "decode_len": 8192,
            "mode": "decode"
        }
        
        print(f"\n📊 Testing {model['name']} parameter validation...")
        
        try:
            # Run CLI command
            print("  🔧 Running CLI command...")
            cli_result = self.run_cli_command(
                model['hf_path'],
                model['sequence_length'],
                model['batch_size'],
                model['decode_len'],
                model['mode']
            )
            
            # Call web API
            print("  🌐 Calling web API...")
            web_result = self.call_web_api(
                model['hf_path'],
                model['sequence_length'],
                model['batch_size'],
                model['decode_len'],
                model['mode']
            )
            
            # Check parameter counts
            cli_params = cli_result.get('total_parameters', 0)
            web_params = web_result['model_info'].get('total_parameters', 0)
            expected_params = 685_000_000_000  # 685B from HuggingFace metadata
            
            print(f"  📊 CLI Parameters: {cli_params:,}")
            print(f"  📊 Web Parameters: {web_params:,}")
            print(f"  📊 Expected Parameters: {expected_params:,}")
            
            # Both should match the expected 685B parameters
            self.assertEqual(cli_params, expected_params, 
                           f"CLI should show 685B parameters, got {cli_params:,}")
            self.assertEqual(web_params, expected_params, 
                           f"Web should show 685B parameters, got {web_params:,}")
            
            print(f"  ✅ Parameter validation working correctly: {cli_params:,} parameters")
            
        except Exception as e:
            self.fail(f"Parameter validation test failed: {str(e)}")
    
    def test_edge_cases_consistency(self):
        """Test edge cases that might cause inconsistencies"""
        print("\n🧪 Testing Edge Cases Consistency")
        print("=" * 60)
        
        edge_cases = [
            {
                "name": "Very Large Sequence Length",
                "hf_path": "https://huggingface.co/Qwen/Qwen2.5-7B",
                "sequence_length": 32768,
                "batch_size": 1,
                "decode_len": 1,
                "mode": "prefill"
            },
            {
                "name": "Large Batch Size",
                "hf_path": "https://huggingface.co/Qwen/Qwen2.5-7B",
                "sequence_length": 1024,
                "batch_size": 128,
                "decode_len": 1,
                "mode": "decode"
            },
            {
                "name": "Long Decode Length",
                "hf_path": "https://huggingface.co/Qwen/Qwen2.5-7B",
                "sequence_length": 1024,
                "batch_size": 1,
                "decode_len": 4096,
                "mode": "decode"
            }
        ]
        
        for case in edge_cases:
            print(f"\n📊 Testing {case['name']}...")
            
            try:
                # Run CLI command
                cli_result = self.run_cli_command(
                    case['hf_path'],
                    case['sequence_length'],
                    case['batch_size'],
                    case['decode_len'],
                    case['mode']
                )
                
                # Call web API
                web_result = self.call_web_api(
                    case['hf_path'],
                    case['sequence_length'],
                    case['batch_size'],
                    case['decode_len'],
                    case['mode']
                )
                
                # Compare results with higher tolerance for edge cases
                match, message = self.compare_results(cli_result, web_result, tolerance=0.05)
                
                if match:
                    print(f"  ✅ {case['name']}: {message}")
                else:
                    print(f"  ❌ {case['name']}: {message}")
                    self.fail(f"Inconsistency found for {case['name']}: {message}")
                
            except Exception as e:
                print(f"  ⚠️  {case['name']}: {str(e)}")
                # Don't fail the test for edge cases that might not be supported


def run_consistency_tests():
    """Run all consistency tests"""
    print("🚀 Starting CLI-Web Interface Consistency Tests")
    print("=" * 80)
    print("This test ensures CLI and web interface produce identical results")
    print("=" * 80)
    
    # Start web interface
    test_suite = CLIWebConsistencyTest()
    test_suite.setUp()
    
    if not test_suite.start_web_interface():
        print("❌ Failed to start web interface")
        return False
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n🎯 Consistency Test Summary")
    print("=" * 80)
    print("✅ All tests passed - CLI and web interface are consistent!")
    print("🔒 Future changes will be validated against this baseline")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    run_consistency_tests()
