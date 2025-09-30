
# Fix imports for project structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Comprehensive Test Runner

This module runs all validation and benchmarking tests to ensure
the transformer calculator is working correctly and efficiently.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Import our test modules
from benchmark_suite import BenchmarkSuite
from real_world_validation import RealWorldValidator
from performance_profiler import PerformanceProfiler
from transformer_calculator import ModelConfig, ModelType, DataType, OperationMode


class ComprehensiveTestRunner:
    """Runs all tests and generates comprehensive reports"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_benchmark_suite(self) -> Dict[str, Any]:
        """Run the benchmark suite"""
        print("🧪 Running Benchmark Suite...")
        print("=" * 50)
        
        suite = BenchmarkSuite()
        suite.create_standard_validation_cases()
        results = suite.run_all_validations()
        
        # Calculate summary statistics
        accuracy_scores = [r.accuracy_score for r in results]
        execution_times = [r.execution_time for r in results]
        
        summary = {
            "total_tests": len(results),
            "average_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "passed_tests": len([r for r in results if r.accuracy_score >= 0.8]),
            "results": results
        }
        
        print(f"✅ Benchmark Suite Complete: {summary['passed_tests']}/{summary['total_tests']} tests passed")
        return summary
    
    def run_real_world_validation(self) -> Dict[str, Any]:
        """Run real-world model validation"""
        print("\n🌍 Running Real-World Validation...")
        print("=" * 50)
        
        validator = RealWorldValidator()
        validator.create_standard_test_cases()
        results = validator.run_all_tests()
        
        # Calculate summary statistics
        successful_tests = [r for r in results if r["success"]]
        
        summary = {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(results) * 100 if results else 0,
            "average_execution_time": sum(r.get("execution_time", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0,
            "results": results
        }
        
        print(f"✅ Real-World Validation Complete: {summary['successful_tests']}/{summary['total_tests']} tests passed")
        return summary
    
    def run_performance_profiling(self) -> Dict[str, Any]:
        """Run performance profiling"""
        print("\n🔍 Running Performance Profiling...")
        print("=" * 50)
        
        profiler = PerformanceProfiler()
        
        # Test with a standard configuration
        test_config = ModelConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            model_type=ModelType.DENSE,
            dtype=DataType.BF16
        )
        
        # Profile calculator
        calculator_results = profiler.profile_calculator(test_config)
        
        # Benchmark scaling
        scaling_results = profiler.benchmark_scaling(test_config)
        
        # Benchmark data types
        dtype_results = profiler.benchmark_data_types(test_config)
        
        summary = {
            "calculator_performance": calculator_results,
            "scaling_performance": scaling_results,
            "data_type_performance": dtype_results,
            "metrics": [metric.to_dict() for metric in profiler.metrics]
        }
        
        print("✅ Performance Profiling Complete")
        return summary
    
    def run_data_type_validation(self) -> Dict[str, Any]:
        """Run comprehensive data type validation"""
        print("\n🎯 Running Data Type Validation...")
        print("=" * 50)
        
        # Test all data types with a standard model
        base_config = ModelConfig(
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=16,
            intermediate_size=5504,
            vocab_size=32000,
            model_type=ModelType.DENSE
        )
        
        data_types = [DataType.FP32, DataType.FP16, DataType.BF16, DataType.INT8, DataType.INT4]
        results = {}
        
        for dtype in data_types:
            print(f"Testing {dtype.value.upper()}...")
            
            config = ModelConfig(
                num_layers=base_config.num_layers,
                hidden_size=base_config.hidden_size,
                num_attention_heads=base_config.num_attention_heads,
                intermediate_size=base_config.intermediate_size,
                vocab_size=base_config.vocab_size,
                model_type=base_config.model_type,
                dtype=dtype
            )
            
            from transformer_calculator import TransformerCalculator
            calculator = TransformerCalculator(config)
            
            memory = calculator.calculate_memory(OperationMode.PREFILL)
            flops = calculator.calculate_flops(OperationMode.PREFILL)
            reuse = calculator.calculate_reuse(OperationMode.PREFILL)
            
            results[dtype.value] = {
                "dtype_bytes": dtype.bytes,
                "memory_reduction_factor": dtype.memory_reduction_factor,
                "memory_gb": memory.total / 1e9,
                "flops_tflop": flops.total / 1e12,
                "reuse": reuse.total,
                "memory_components": {
                    "attention": memory.attention / 1e9,
                    "projections": memory.projections / 1e9,
                    "mlp": memory.mlp / 1e9,
                    "embeddings": memory.embeddings / 1e9,
                    "layer_norm": memory.layer_norm / 1e9,
                    "weights": memory.weights / 1e9
                }
            }
            
            print(f"  Memory: {memory.total/1e9:.1f} GB")
            print(f"  FLOPS: {flops.total/1e12:.1f} TFLOP")
            print(f"  Reuse: {reuse.total:.1f} FLOPS/byte")
        
        print("✅ Data Type Validation Complete")
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        all_results = {
            "start_time": self.start_time,
            "benchmark_suite": None,
            "real_world_validation": None,
            "performance_profiling": None,
            "data_type_validation": None
        }
        
        try:
            # Run benchmark suite
            all_results["benchmark_suite"] = self.run_benchmark_suite()
        except Exception as e:
            print(f"❌ Benchmark Suite failed: {e}")
            all_results["benchmark_suite"] = {"error": str(e)}
        
        try:
            # Run real-world validation
            all_results["real_world_validation"] = self.run_real_world_validation()
        except Exception as e:
            print(f"❌ Real-World Validation failed: {e}")
            all_results["real_world_validation"] = {"error": str(e)}
        
        try:
            # Run performance profiling
            all_results["performance_profiling"] = self.run_performance_profiling()
        except Exception as e:
            print(f"❌ Performance Profiling failed: {e}")
            all_results["performance_profiling"] = {"error": str(e)}
        
        try:
            # Run data type validation
            all_results["data_type_validation"] = self.run_data_type_validation()
        except Exception as e:
            print(f"❌ Data Type Validation failed: {e}")
            all_results["data_type_validation"] = {"error": str(e)}
        
        all_results["end_time"] = time.time()
        all_results["total_duration"] = all_results["end_time"] - all_results["start_time"]
        
        return all_results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("📊 COMPREHENSIVE TEST SUITE REPORT")
        report.append("=" * 60)
        
        # Overall summary
        total_duration = results.get("total_duration", 0)
        report.append(f"\n⏱️  Test Duration: {total_duration:.1f} seconds")
        
        # Benchmark Suite Results
        if results.get("benchmark_suite") and "error" not in results["benchmark_suite"]:
            bs = results["benchmark_suite"]
            report.append(f"\n🧪 Benchmark Suite:")
            report.append(f"   Tests: {bs['total_tests']}")
            report.append(f"   Passed: {bs['passed_tests']}")
            report.append(f"   Average Accuracy: {bs['average_accuracy']:.2f}")
            report.append(f"   Average Time: {bs['average_execution_time']:.3f}s")
        else:
            report.append(f"\n🧪 Benchmark Suite: ❌ FAILED")
        
        # Real-World Validation Results
        if results.get("real_world_validation") and "error" not in results["real_world_validation"]:
            rw = results["real_world_validation"]
            report.append(f"\n🌍 Real-World Validation:")
            report.append(f"   Tests: {rw['total_tests']}")
            report.append(f"   Successful: {rw['successful_tests']}")
            report.append(f"   Success Rate: {rw['success_rate']:.1f}%")
            report.append(f"   Average Time: {rw['average_execution_time']:.3f}s")
        else:
            report.append(f"\n🌍 Real-World Validation: ❌ FAILED")
        
        # Performance Profiling Results
        if results.get("performance_profiling") and "error" not in results["performance_profiling"]:
            pp = results["performance_profiling"]
            report.append(f"\n🔍 Performance Profiling:")
            report.append(f"   Status: ✅ COMPLETED")
            if "metrics" in pp and pp["metrics"]:
                avg_time = sum(m["execution_time"] for m in pp["metrics"]) / len(pp["metrics"])
                avg_memory = sum(m["memory_peak"] for m in pp["metrics"]) / len(pp["metrics"])
                report.append(f"   Average Execution Time: {avg_time:.3f}s")
                report.append(f"   Average Peak Memory: {avg_memory:.1f} MB")
        else:
            report.append(f"\n🔍 Performance Profiling: ❌ FAILED")
        
        # Data Type Validation Results
        if results.get("data_type_validation") and "error" not in results["data_type_validation"]:
            dt = results["data_type_validation"]
            report.append(f"\n🎯 Data Type Validation:")
            report.append(f"   Data Types Tested: {len(dt)}")
            for dtype, data in dt.items():
                report.append(f"   {dtype.upper()}: {data['memory_gb']:.1f} GB, {data['memory_reduction_factor']:.1f}x reduction")
        else:
            report.append(f"\n🎯 Data Type Validation: ❌ FAILED")
        
        # Overall Assessment
        report.append(f"\n🎯 Overall Assessment:")
        
        passed_tests = 0
        total_tests = 0
        
        if results.get("benchmark_suite") and "error" not in results["benchmark_suite"]:
            passed_tests += results["benchmark_suite"]["passed_tests"]
            total_tests += results["benchmark_suite"]["total_tests"]
        
        if results.get("real_world_validation") and "error" not in results["real_world_validation"]:
            passed_tests += results["real_world_validation"]["successful_tests"]
            total_tests += results["real_world_validation"]["total_tests"]
        
        if total_tests > 0:
            success_rate = passed_tests / total_tests * 100
            report.append(f"   Overall Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                report.append(f"   Status: 🎉 EXCELLENT - All systems working perfectly!")
            elif success_rate >= 80:
                report.append(f"   Status: ✅ GOOD - Most systems working well!")
            elif success_rate >= 70:
                report.append(f"   Status: ⚠️  FAIR - Some issues detected!")
            else:
                report.append(f"   Status: ❌ POOR - Significant issues detected!")
        else:
            report.append(f"   Status: ❌ NO TESTS COMPLETED")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "comprehensive_test_results.json"):
        """Save comprehensive test results to JSON file"""
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key == "benchmark_suite" and value and "results" in value:
                # Convert BenchmarkResult objects to dicts
                json_results[key] = {
                    "total_tests": value["total_tests"],
                    "average_accuracy": value["average_accuracy"],
                    "average_execution_time": value["average_execution_time"],
                    "passed_tests": value["passed_tests"]
                }
            elif key == "real_world_validation" and value and "results" in value:
                json_results[key] = {
                    "total_tests": value["total_tests"],
                    "successful_tests": value["successful_tests"],
                    "success_rate": value["success_rate"],
                    "average_execution_time": value["average_execution_time"]
                }
            elif hasattr(value, 'to_dict'):
                # Handle objects with to_dict method (like PerformanceMetrics)
                json_results[key] = value.to_dict()
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Comprehensive results saved to {filename}")


def main():
    """Run the comprehensive test suite"""
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Generate comprehensive report
    report = runner.generate_comprehensive_report(results)
    print("\n" + report)
    
    # Save results
    runner.save_results(results)
    
    print(f"\n🎯 Test Suite Complete!")
    print(f"   Total Duration: {results.get('total_duration', 0):.1f} seconds")
    print(f"   Results saved to comprehensive_test_results.json")


if __name__ == "__main__":
    main()
