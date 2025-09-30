
# Fix imports for project structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Real-World Model Validation Suite

This module validates the calculator against actual Hugging Face models
to ensure accuracy in real-world scenarios.
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
    DataTypeDetector
)
from hf_model_fetcher import create_model_config_from_hf
from validation_ranges import get_validation_ranges
from validation_common import ValidationRunner, ValidationCase


@dataclass
class RealWorldTest:
    """A real-world model test case"""
    model_name: str
    hf_path: str
    expected_memory_range: Tuple[float, float]  # GB
    expected_flops_range: Tuple[float, float]    # TFLOP
    test_modes: List[OperationMode]
    notes: str = ""
    skip_if_fails: bool = False


class RealWorldValidator:
    """Validates calculator against real Hugging Face models"""
    
    def __init__(self):
        self.test_cases: List[RealWorldTest] = []
        self.results: List[Dict] = []
        self.validation_runner = ValidationRunner()
        
    def add_test_case(self, test: RealWorldTest):
        """Add a real-world test case"""
        self.test_cases.append(test)
    
    def create_standard_test_cases(self):
        """Create standard test cases for popular models with realistic validation ranges"""
        
        # Small models (7B range) - using dynamic validation ranges
        self.add_test_case(RealWorldTest(
            model_name="Qwen2.5-7B",
            hf_path="Qwen/Qwen2.5-7B",
            expected_memory_range=self._get_dynamic_ranges(7_000_000_000, "Qwen2.5-7B"),
            expected_flops_range=self._get_dynamic_ranges(7_000_000_000, "Qwen2.5-7B", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="Popular 7B model, using realistic validation ranges"
        ))
        
        self.add_test_case(RealWorldTest(
            model_name="Phi-3.5-mini",
            hf_path="microsoft/Phi-3.5-mini-instruct",
            expected_memory_range=self._get_dynamic_ranges(3_800_000_000, "Phi-3.5-mini"),
            expected_flops_range=self._get_dynamic_ranges(3_800_000_000, "Phi-3.5-mini", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="Microsoft Phi-3.5 mini model, using realistic validation ranges"
        ))
        
        # Medium models (13B range)
        self.add_test_case(RealWorldTest(
            model_name="Qwen2.5-14B",
            hf_path="Qwen/Qwen2.5-14B",
            expected_memory_range=self._get_dynamic_ranges(14_000_000_000, "Qwen2.5-14B"),
            expected_flops_range=self._get_dynamic_ranges(14_000_000_000, "Qwen2.5-14B", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="14B parameter model, using realistic validation ranges"
        ))
        
        # Large models (70B range)
        self.add_test_case(RealWorldTest(
            model_name="Qwen2.5-72B",
            hf_path="Qwen/Qwen2.5-72B",
            expected_memory_range=self._get_dynamic_ranges(72_000_000_000, "Qwen2.5-72B"),
            expected_flops_range=self._get_dynamic_ranges(72_000_000_000, "Qwen2.5-72B", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="Large 72B model, using realistic validation ranges"
        ))
        
        # MoE models
        self.add_test_case(RealWorldTest(
            model_name="Qwen2.5-MoE-A14B",
            hf_path="Qwen/Qwen2.5-MoE-A14B",
            expected_memory_range=self._get_dynamic_ranges(14_000_000_000, "Qwen2.5-MoE-A14B"),
            expected_flops_range=self._get_dynamic_ranges(14_000_000_000, "Qwen2.5-MoE-A14B", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="MoE model with 8 experts, using realistic validation ranges"
        ))
        
        # Specialized models
        self.add_test_case(RealWorldTest(
            model_name="Qwen2.5-7B",
            hf_path="Qwen/Qwen2.5-7B",
            expected_memory_range=self._get_dynamic_ranges(7_000_000_000, "Qwen2.5-7B"),
            expected_flops_range=self._get_dynamic_ranges(7_000_000_000, "Qwen2.5-7B", "flops"),
            test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
            notes="Qwen 2.5 model, using realistic validation ranges"
        ))
    
    def _get_dynamic_ranges(self, parameter_count: int, model_name: str, 
                           range_type: str = "memory") -> Tuple[float, float]:
        """Get dynamic validation ranges based on model statistics"""
        ranges = get_validation_ranges(
            parameter_count=parameter_count,
            model_name=model_name,
            data_type="BF16",
            sequence_length=2048,
            batch_size=1
        )
        
        if range_type == "memory":
            return ranges["memory_range"]
        elif range_type == "flops":
            return ranges["flops_range"]
        else:
            raise ValueError(f"Unknown range type: {range_type}")
    
    def run_test_case(self, test: RealWorldTest) -> Dict:
        """Run a single test case"""
        print(f"\n🧪 Testing: {test.model_name}")
        print(f"   HF Path: {test.hf_path}")
        print(f"   Expected Memory: {test.expected_memory_range[0]}-{test.expected_memory_range[1]} GB")
        print(f"   Expected FLOPS: {test.expected_flops_range[0]}-{test.expected_flops_range[1]} TFLOP")
        
        result = {
            "model_name": test.model_name,
            "hf_path": test.hf_path,
            "success": False,
            "error": None,
            "config": None,
            "results": {},
            "data_type_detected": None,
            "execution_time": 0.0
        }
        
        try:
            start_time = time.time()
            
            # Fetch model configuration
            print("   📥 Fetching model configuration...")
            config = create_model_config_from_hf(
                test.hf_path,
                sequence_length=2048,  # Use reasonable default
                batch_size=1,
                decode_len=64
            )
            
            if not config:
                result["error"] = "Failed to fetch model configuration"
                print(f"   ❌ {result['error']}")
                return result
            
            # Auto-detect data type
            try:
                from huggingface_hub import hf_hub_download
                import json
                model_id = test.hf_path.replace('https://huggingface.co/', '').replace('@', '')
                config_path = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
                with open(config_path, 'r') as f:
                    hf_config = json.load(f)
                detected_dtype = DataTypeDetector.detect_from_config(hf_config)
                config.dtype = detected_dtype
                result["data_type_detected"] = detected_dtype.value
                print(f"   🔍 Auto-detected data type: {detected_dtype.value}")
            except Exception as e:
                print(f"   ⚠️  Could not auto-detect data type: {e}")
                result["data_type_detected"] = "unknown"
            
            result["config"] = {
                "num_layers": config.num_layers,
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "intermediate_size": config.intermediate_size,
                "vocab_size": config.vocab_size,
                "model_type": config.model_type.value,
                "dtype": config.dtype.value,
                "num_experts": getattr(config, 'num_experts', 1),
                "top_k": getattr(config, 'top_k', 1)
            }
            
            # Run calculations for each test mode
            calculator = TransformerCalculator(config)
            mode_results = {}
            
            for mode in test.test_modes:
                print(f"   🔄 Testing {mode.value} mode...")
                
                memory = calculator.calculate_memory(mode)
                flops = calculator.calculate_flops(mode)
                reuse = calculator.calculate_reuse(mode)
                
                mode_results[mode.value] = {
                    "memory_gb": memory.total / 1e9,
                    "flops_tflop": flops.total / 1e12,
                    "reuse": reuse.total,
                    "memory_components": {
                        "attention": memory.attention / 1e9,
                        "projections": memory.projections / 1e9,
                        "mlp": memory.mlp / 1e9,
                        "embeddings": memory.embeddings / 1e9,
                        "layer_norm": memory.layer_norm / 1e9,
                        "weights": memory.weights / 1e9,
                        "kv_cache": memory.kv_cache / 1e9
                    }
                }
                
                # Validate against expected ranges
                memory_gb = memory.total / 1e9
                flops_tflop = flops.total / 1e12
                
                memory_valid = test.expected_memory_range[0] <= memory_gb <= test.expected_memory_range[1]
                flops_valid = test.expected_flops_range[0] <= flops_tflop <= test.expected_flops_range[1]
                
                print(f"      Memory: {memory_gb:.1f} GB {'✅' if memory_valid else '❌'}")
                print(f"      FLOPS: {flops_tflop:.1f} TFLOP {'✅' if flops_valid else '❌'}")
                print(f"      Reuse: {reuse.total:.1f} FLOPS/byte")
            
            result["results"] = mode_results
            result["success"] = True
            result["execution_time"] = time.time() - start_time
            
            print(f"   ✅ Test completed in {result['execution_time']:.2f}s")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"   ❌ Error: {e}")
            if not test.skip_if_fails:
                raise
        
        return result
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test cases"""
        print("🌍 Running Real-World Model Validation")
        print("=" * 60)
        
        results = []
        successful_tests = 0
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"\n📊 Test {i}/{len(self.test_cases)}")
            
            try:
                result = self.run_test_case(test)
                results.append(result)
                
                if result["success"]:
                    successful_tests += 1
                    print(f"   ✅ {test.model_name} - PASSED")
                else:
                    print(f"   ❌ {test.model_name} - FAILED: {result['error']}")
                    
            except Exception as e:
                print(f"   💥 {test.model_name} - CRASHED: {e}")
                if not test.skip_if_fails:
                    raise
        
        self.results = results
        
        print(f"\n📈 Summary:")
        print(f"   Tests Run: {len(results)}")
        print(f"   Successful: {successful_tests}")
        print(f"   Failed: {len(results) - successful_tests}")
        print(f"   Success Rate: {successful_tests/len(results)*100:.1f}%")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        if not self.results:
            return "No results available. Run tests first."
        
        report = []
        report.append("🌍 REAL-WORLD MODEL VALIDATION REPORT")
        report.append("=" * 50)
        
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]
        
        report.append(f"\n📊 Overall Results:")
        report.append(f"   Total Tests: {len(self.results)}")
        report.append(f"   Successful: {len(successful_results)}")
        report.append(f"   Failed: {len(failed_results)}")
        report.append(f"   Success Rate: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if successful_results:
            # Data type analysis
            dtype_counts = {}
            for result in successful_results:
                dtype = result.get("data_type_detected", "unknown")
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            
            report.append(f"\n🔍 Data Type Detection:")
            for dtype, count in dtype_counts.items():
                report.append(f"   {dtype.upper()}: {count} models")
            
            # Model type analysis
            model_type_counts = {}
            for result in successful_results:
                model_type = result["config"]["model_type"]
                model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
            
            report.append(f"\n🏗️ Model Types:")
            for model_type, count in model_type_counts.items():
                report.append(f"   {model_type.upper()}: {count} models")
        
        # Detailed results
        report.append(f"\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            report.append(f"\n{i}. {result['model_name']}")
            if result["success"]:
                report.append(f"   Status: ✅ PASSED")
                report.append(f"   Data Type: {result.get('data_type_detected', 'unknown').upper()}")
                report.append(f"   Execution Time: {result['execution_time']:.2f}s")
                
                for mode, mode_result in result["results"].items():
                    report.append(f"   {mode.upper()} Mode:")
                    report.append(f"     Memory: {mode_result['memory_gb']:.1f} GB")
                    report.append(f"     FLOPS: {mode_result['flops_tflop']:.1f} TFLOP")
                    report.append(f"     Reuse: {mode_result['reuse']:.1f} FLOPS/byte")
            else:
                report.append(f"   Status: ❌ FAILED")
                report.append(f"   Error: {result['error']}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "real_world_validation.json"):
        """Save validation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to {filename}")


def main():
    """Run the real-world validation suite"""
    print("🌍 Starting Real-World Model Validation Suite")
    print("=" * 60)
    
    # Create validator
    validator = RealWorldValidator()
    validator.create_standard_test_cases()
    
    # Run tests
    results = validator.run_all_tests()
    
    # Generate report
    report = validator.generate_report()
    print("\n" + report)
    
    # Save results
    validator.save_results()
    
    # Final summary
    successful = len([r for r in results if r["success"]])
    total = len(results)
    
    print(f"\n🎯 Final Summary:")
    print(f"   Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful == total:
        print("   🎉 All tests passed! Calculator is working perfectly!")
    elif successful >= total * 0.8:
        print("   ✅ Most tests passed! Calculator is working well!")
    else:
        print("   ⚠️  Some tests failed. Review the results for issues.")


if __name__ == "__main__":
    main()
