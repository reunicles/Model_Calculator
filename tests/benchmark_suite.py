"""
Performance Benchmarking and Validation Suite

This module provides comprehensive testing and validation for the transformer calculator,
ensuring accuracy across different models, data types, and configurations.
"""

import time
import json
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
    MemoryComponents, FLOPSComponents, ReuseComponents
)
from validation_common import ValidationRunner, ValidationCase, create_standard_validation_cases
from validation_ranges import get_validation_ranges


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    model_name: str
    config: ModelConfig
    mode: OperationMode
    memory: MemoryComponents
    flops: FLOPSComponents
    reuse: ReuseComponents
    execution_time: float
    memory_peak: float
    accuracy_score: float = 0.0
    notes: str = ""


@dataclass
class ValidationCase:
    """A validation test case with expected results"""
    name: str
    model_config: Dict[str, Any]
    expected_memory_range: Tuple[float, float]  # (min, max) in bytes
    expected_flops_range: Tuple[float, float]    # (min, max) in FLOPS
    expected_reuse_range: Tuple[float, float]   # (min, max) reuse
    tolerance: float = 0.1  # 10% tolerance by default
    notes: str = ""


class BenchmarkSuite:
    """Comprehensive benchmarking and validation suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.validation_cases: List[ValidationCase] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    def add_validation_case(self, case: ValidationCase):
        """Add a validation test case"""
        self.validation_cases.append(case)
    
    def create_standard_validation_cases(self):
        """Create standard validation cases for common model configurations"""
        
        # Small Dense Model (7B parameters)
        self.add_validation_case(ValidationCase(
            name="Small Dense Model (7B)",
            model_config={
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "model_type": ModelType.DENSE,
                "dtype": DataType.BF16
            },
            expected_memory_range=(5e9, 50e9),  # 5-50 GB (more realistic range)
            expected_flops_range=(1e12, 50e12),  # 1-50 TFLOP (more realistic)
            expected_reuse_range=(100, 10000),   # 100-10000 FLOPS/byte (more realistic)
            tolerance=0.5,  # 50% tolerance
            notes="Standard 7B parameter model"
        ))
        
        # Large Dense Model (70B parameters)
        self.add_validation_case(ValidationCase(
            name="Large Dense Model (70B)",
            model_config={
                "num_layers": 80,
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "intermediate_size": 28672,
                "vocab_size": 32000,
                "model_type": ModelType.DENSE,
                "dtype": DataType.BF16
            },
            expected_memory_range=(50e9, 500e9),  # 50-500 GB (more realistic range)
            expected_flops_range=(10e12, 500e12),  # 10-500 TFLOP (more realistic)
            expected_reuse_range=(100, 10000),     # 100-10000 FLOPS/byte (more realistic)
            tolerance=0.5,  # 50% tolerance
            notes="Large 70B parameter model"
        ))
        
        # MoE Model (8 experts)
        self.add_validation_case(ValidationCase(
            name="MoE Model (8 experts)",
            model_config={
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "model_type": ModelType.MOE,
                "num_experts": 8,
                "top_k": 2,
                "dtype": DataType.BF16
            },
            expected_memory_range=(10e9, 200e9),   # 10-200 GB (MoE can be higher)
            expected_flops_range=(1e12, 100e12),   # 1-100 TFLOP (more realistic)
            expected_reuse_range=(50, 5000),     # 50-5000 FLOPS/byte (more realistic)
            tolerance=0.5,  # 50% tolerance for MoE
            notes="MoE model with 8 experts"
        ))
        
        # Quantized Model (INT8)
        self.add_validation_case(ValidationCase(
            name="Quantized Model (INT8)",
            model_config={
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "model_type": ModelType.DENSE,
                "dtype": DataType.INT8
            },
            expected_memory_range=(1e9, 20e9),    # 1-20 GB (4x reduction)
            expected_flops_range=(1e12, 50e12),   # 1-50 TFLOP (more realistic)
            expected_reuse_range=(1000, 20000),  # 1000-20000 FLOPS/byte (more realistic)
            tolerance=0.5,  # 50% tolerance
            notes="INT8 quantized model"
        ))
        
        # Extreme Quantization (INT4)
        self.add_validation_case(ValidationCase(
            name="Extreme Quantization (INT4)",
            model_config={
                "num_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "model_type": ModelType.DENSE,
                "dtype": DataType.INT4
            },
            expected_memory_range=(0.5e9, 10e9),   # 0.5-10 GB (8x reduction)
            expected_flops_range=(0.5e12, 20e12),   # 0.5-20 TFLOP
            expected_reuse_range=(1000, 50000),    # 1000-50000 FLOPS/byte
            tolerance=0.5,  # More tolerance for INT4
            notes="INT4 quantized model"
        ))
    
    def run_validation_case(self, case: ValidationCase) -> BenchmarkResult:
        """Run a single validation case and return results"""
        start_time = time.time()
        
        # Create model config
        config = ModelConfig(**case.model_config)
        calculator = TransformerCalculator(config)
        
        # Run calculations for all modes
        mode = OperationMode.PREFILL  # Use prefill as standard test
        memory = calculator.calculate_memory(mode)
        flops = calculator.calculate_flops(mode)
        reuse = calculator.calculate_reuse(mode)
        
        execution_time = time.time() - start_time
        
        # Calculate accuracy score
        memory_score = self._calculate_accuracy_score(
            memory.total, case.expected_memory_range, case.tolerance
        )
        flops_score = self._calculate_accuracy_score(
            flops.total, case.expected_flops_range, case.tolerance
        )
        reuse_score = self._calculate_accuracy_score(
            reuse.total, case.expected_reuse_range, case.tolerance
        )
        
        accuracy_score = (memory_score + flops_score + reuse_score) / 3
        
        return BenchmarkResult(
            model_name=case.name,
            config=config,
            mode=mode,
            memory=memory,
            flops=flops,
            reuse=reuse,
            execution_time=execution_time,
            memory_peak=0.0,  # Would need memory profiling for real measurement
            accuracy_score=accuracy_score,
            notes=case.notes
        )
    
    def _calculate_accuracy_score(self, actual: float, expected_range: Tuple[float, float], tolerance: float) -> float:
        """Calculate accuracy score (0-1) based on expected range and tolerance"""
        min_expected, max_expected = expected_range
        
        # Check if within expected range
        if min_expected <= actual <= max_expected:
            return 1.0
        
        # Calculate how far off we are
        if actual < min_expected:
            error_ratio = (min_expected - actual) / min_expected
        else:
            error_ratio = (actual - max_expected) / max_expected
        
        # Apply tolerance
        if error_ratio <= tolerance:
            return 1.0 - (error_ratio / tolerance) * 0.5
        else:
            return max(0.0, 1.0 - error_ratio)
    
    def run_all_validations(self) -> List[BenchmarkResult]:
        """Run all validation cases"""
        print("🧪 Running Validation Suite...")
        print("=" * 60)
        
        results = []
        for i, case in enumerate(self.validation_cases, 1):
            print(f"\n📊 Test {i}/{len(self.validation_cases)}: {case.name}")
            print(f"   Expected Memory: {case.expected_memory_range[0]/1e9:.1f}-{case.expected_memory_range[1]/1e9:.1f} GB")
            print(f"   Expected FLOPS: {case.expected_flops_range[0]/1e12:.1f}-{case.expected_flops_range[1]/1e12:.1f} TFLOP")
            
            try:
                result = self.run_validation_case(case)
                results.append(result)
                
                # Print results
                print(f"   ✅ Actual Memory: {result.memory.total/1e9:.1f} GB")
                print(f"   ✅ Actual FLOPS: {result.flops.total/1e12:.1f} TFLOP")
                print(f"   ✅ Actual Reuse: {result.reuse.total:.1f} FLOPS/byte")
                print(f"   ✅ Accuracy Score: {result.accuracy_score:.2f}")
                print(f"   ⏱️  Execution Time: {result.execution_time:.3f}s")
                
                if result.accuracy_score < 0.8:
                    print(f"   ⚠️  Warning: Low accuracy score!")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        self.results = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No results available. Run validations first."
        
        report = []
        report.append("📊 PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        
        # Overall statistics
        accuracy_scores = [r.accuracy_score for r in self.results]
        execution_times = [r.execution_time for r in self.results]
        
        report.append(f"\n📈 Overall Performance:")
        report.append(f"   Tests Run: {len(self.results)}")
        report.append(f"   Average Accuracy: {statistics.mean(accuracy_scores):.2f}")
        report.append(f"   Best Accuracy: {max(accuracy_scores):.2f}")
        report.append(f"   Worst Accuracy: {min(accuracy_scores):.2f}")
        report.append(f"   Average Execution Time: {statistics.mean(execution_times):.3f}s")
        
        # Data type performance
        dtype_performance = {}
        for result in self.results:
            dtype = result.config.dtype.value
            if dtype not in dtype_performance:
                dtype_performance[dtype] = []
            dtype_performance[dtype].append(result.accuracy_score)
        
        report.append(f"\n🎯 Data Type Performance:")
        for dtype, scores in dtype_performance.items():
            report.append(f"   {dtype.upper()}: {statistics.mean(scores):.2f} avg accuracy")
        
        # Model type performance
        model_performance = {}
        for result in self.results:
            model_type = result.config.model_type.value
            if model_type not in model_performance:
                model_performance[model_type] = []
            model_performance[model_type].append(result.accuracy_score)
        
        report.append(f"\n🏗️ Model Type Performance:")
        for model_type, scores in model_performance.items():
            report.append(f"   {model_type.upper()}: {statistics.mean(scores):.2f} avg accuracy")
        
        # Detailed results
        report.append(f"\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            report.append(f"\n{i}. {result.model_name}")
            report.append(f"   Memory: {result.memory.total/1e9:.1f} GB")
            report.append(f"   FLOPS: {result.flops.total/1e12:.1f} TFLOP")
            report.append(f"   Reuse: {result.reuse.total:.1f} FLOPS/byte")
            report.append(f"   Accuracy: {result.accuracy_score:.2f}")
            report.append(f"   Time: {result.execution_time:.3f}s")
            if result.notes:
                report.append(f"   Notes: {result.notes}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        results_data = []
        for result in self.results:
            results_data.append({
                "model_name": result.model_name,
                "config": {
                    "num_layers": result.config.num_layers,
                    "hidden_size": result.config.hidden_size,
                    "num_attention_heads": result.config.num_attention_heads,
                    "intermediate_size": result.config.intermediate_size,
                    "vocab_size": result.config.vocab_size,
                    "model_type": result.config.model_type.value,
                    "dtype": result.config.dtype.value,
                    "num_experts": getattr(result.config, 'num_experts', 1),
                    "top_k": getattr(result.config, 'top_k', 1)
                },
                "results": {
                    "memory_total": result.memory.total,
                    "flops_total": result.flops.total,
                    "reuse_total": result.reuse.total,
                    "execution_time": result.execution_time,
                    "accuracy_score": result.accuracy_score
                },
                "notes": result.notes
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")


def main():
    """Run the benchmark suite"""
    print("🚀 Starting Transformer Calculator Benchmark Suite")
    print("=" * 60)
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    suite.create_standard_validation_cases()
    
    # Run validations
    results = suite.run_all_validations()
    
    # Generate report
    report = suite.generate_performance_report()
    print("\n" + report)
    
    # Save results
    suite.save_results()
    
    # Summary
    if results:
        avg_accuracy = statistics.mean([r.accuracy_score for r in results])
        print(f"\n🎯 Summary:")
        print(f"   Tests Passed: {len([r for r in results if r.accuracy_score >= 0.8])}/{len(results)}")
        print(f"   Average Accuracy: {avg_accuracy:.2f}")
        
        if avg_accuracy >= 0.9:
            print("   🎉 Excellent performance!")
        elif avg_accuracy >= 0.8:
            print("   ✅ Good performance!")
        else:
            print("   ⚠️  Performance needs improvement!")


if __name__ == "__main__":
    main()
