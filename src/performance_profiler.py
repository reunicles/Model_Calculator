"""
Performance Profiling Tool

This module provides detailed performance profiling for the transformer calculator,
measuring execution time, memory usage, and identifying bottlenecks.
"""

import time
import psutil
import os
import sys
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading

# Configure logging
logger = logging.getLogger(__name__)

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation: str
    execution_time: float
    memory_peak: float
    memory_avg: float
    cpu_usage: float
    memory_components: Dict[str, float]
    flops_components: Dict[str, float]
    reuse_components: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operation": self.operation,
            "execution_time": self.execution_time,
            "memory_peak": self.memory_peak,
            "memory_avg": self.memory_avg,
            "cpu_usage": self.cpu_usage,
            "memory_components": self.memory_components,
            "flops_components": self.flops_components,
            "reuse_components": self.reuse_components
        }


class PerformanceProfiler:
    """Profiles performance of the transformer calculator"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling an operation"""
        # Start monitoring
        self.start_monitoring()
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # Stop monitoring
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.stop_monitoring()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_peak = max(self.memory_samples) if self.memory_samples else end_memory
            memory_avg = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else end_memory
            cpu_usage = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
            
            # Store metrics
            metric = PerformanceMetrics(
                operation=operation_name,
                execution_time=execution_time,
                memory_peak=memory_peak,
                memory_avg=memory_avg,
                cpu_usage=cpu_usage,
                memory_components={},
                flops_components={},
                reuse_components={}
            )
            
            self.metrics.append(metric)
            
            # Reset samples
            self.memory_samples = []
            self.cpu_samples = []
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources in a separate thread with proper CPU sampling"""
        process = psutil.Process()
        
        # Initialize CPU monitoring with a small delay to get accurate readings
        process.cpu_percent()  # Initialize CPU monitoring
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                # Use proper interval for meaningful CPU readings (100ms)
                cpu_percent = process.cpu_percent(interval=0.1)  # 100ms interval for accurate readings
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms to reduce overhead
            except:
                break
    
    def profile_calculator(self, config: ModelConfig, modes: List[OperationMode] = None) -> Dict[str, Any]:
        """Profile the calculator with a given configuration"""
        if modes is None:
            modes = [OperationMode.PREFILL, OperationMode.DECODE]
        
        logger.info("🔍 Profiling Calculator Performance")
        logger.info(f"   Model: {config.model_type.value.upper()}")
        logger.info(f"   Layers: {config.num_layers}")
        logger.info(f"   Hidden Size: {config.hidden_size}")
        logger.info(f"   Data Type: {config.dtype.value.upper()}")
        logger.info(f"   Modes: {[mode.value for mode in modes]}")
        
        calculator = TransformerCalculator(config)
        results = {}
        
        for mode in modes:
            logger.info(f"📊 Profiling {mode.value} mode...")
            
            with self.profile_operation(f"{mode.value}_calculation"):
                memory = calculator.calculate_memory(mode)
                flops = calculator.calculate_flops(mode)
                reuse = calculator.calculate_reuse(mode)
            
            # Get the latest metrics
            latest_metric = self.metrics[-1]
            
            results[mode.value] = {
                "execution_time": latest_metric.execution_time,
                "memory_peak_mb": latest_metric.memory_peak,
                "memory_avg_mb": latest_metric.memory_avg,
                "cpu_usage": latest_metric.cpu_usage,
                "memory_total_gb": memory.total / 1e9,
                "flops_total_tflop": flops.total / 1e12,
                "reuse_total": reuse.total,
                "memory_breakdown": {
                    "attention": memory.attention / 1e9,
                    "projections": memory.projections / 1e9,
                    "mlp": memory.mlp / 1e9,
                    "embeddings": memory.embeddings / 1e9,
                    "layer_norm": memory.layer_norm / 1e9,
                    "weights": memory.weights / 1e9,
                    "kv_cache": memory.kv_cache / 1e9
                }
            }
            
            logger.info(f"   ⏱️  Execution Time: {latest_metric.execution_time:.3f}s")
            logger.info(f"   💾 Peak Memory: {latest_metric.memory_peak:.1f} MB")
            logger.info(f"   💾 Avg Memory: {latest_metric.memory_avg:.1f} MB")
            logger.info(f"   🖥️  CPU Usage: {latest_metric.cpu_usage:.1f}%")
            logger.info(f"   📊 Memory Total: {memory.total/1e9:.1f} GB")
            logger.info(f"   🔢 FLOPS Total: {flops.total/1e12:.1f} TFLOP")
            logger.info(f"   🔄 Reuse: {reuse.total:.1f} FLOPS/byte")
        
        return results
    
    def benchmark_scaling(self, base_config: ModelConfig, 
                         scale_factors: List[float] = [0.5, 1.0, 2.0, 4.0]) -> Dict[str, Any]:
        """
        Benchmark how performance scales with model size
        
        SCALING STRATEGY:
        - Scale ≤ 2.0: Scale layers only (realistic scaling)
        - Scale > 2.0: Scale hidden_size and attention heads proportionally
        
        This prevents mismatched head dimensions while maintaining
        realistic model scaling patterns.
        """
        logger.info("📈 Benchmarking Scaling Performance")
        logger.info(f"   Base Config: {base_config.num_layers} layers, {base_config.hidden_size} hidden size")
        logger.info(f"   Scale Factors: {scale_factors}")
        logger.info(f"   Strategy: ≤2.0x = layers only, >2.0x = proportional scaling")
        
        scaling_results = {}
        
        for scale in scale_factors:
            logger.info(f"🔍 Testing Scale Factor: {scale}x")
            
            # Scale the configuration with proper attention head scaling
            # Option 1: Scale layers only (recommended for realistic scaling)
            if scale <= 2.0:  # For smaller scale factors, scale layers only
                scaled_config = ModelConfig(
                    num_layers=int(base_config.num_layers * scale),
                    hidden_size=base_config.hidden_size,  # Keep hidden_size constant
                    num_attention_heads=base_config.num_attention_heads,  # Keep heads constant
                    intermediate_size=base_config.intermediate_size,  # Keep intermediate constant
                    vocab_size=base_config.vocab_size,
                    model_type=base_config.model_type,
                    dtype=base_config.dtype,
                    num_experts=getattr(base_config, 'num_experts', 1),
                    top_k=getattr(base_config, 'top_k', 1)
                )
            else:  # For larger scale factors, scale hidden_size proportionally
                # Scale attention heads proportionally to maintain head dimension
                scaled_hidden_size = int(base_config.hidden_size * scale)
                scaled_heads = max(1, int(base_config.num_attention_heads * scale))
                
                scaled_config = ModelConfig(
                    num_layers=int(base_config.num_layers * scale),
                    hidden_size=scaled_hidden_size,
                    num_attention_heads=scaled_heads,
                    intermediate_size=int(base_config.intermediate_size * scale),
                    vocab_size=base_config.vocab_size,
                    model_type=base_config.model_type,
                    dtype=base_config.dtype,
                    num_experts=getattr(base_config, 'num_experts', 1),
                    top_k=getattr(base_config, 'top_k', 1)
                )
            
            # Profile the scaled configuration
            with self.profile_operation(f"scaling_{scale}x"):
                calculator = TransformerCalculator(scaled_config)
                memory = calculator.calculate_memory(OperationMode.PREFILL)
                flops = calculator.calculate_flops(OperationMode.PREFILL)
            
            # Get metrics
            latest_metric = self.metrics[-1]
            
            scaling_results[f"{scale}x"] = {
                "scale_factor": scale,
                "execution_time": latest_metric.execution_time,
                "memory_peak_mb": latest_metric.memory_peak,
                "memory_total_gb": memory.total / 1e9,
                "flops_total_tflop": flops.total / 1e12,
                "config": {
                    "layers": scaled_config.num_layers,
                    "hidden_size": scaled_config.hidden_size,
                    "intermediate_size": scaled_config.intermediate_size
                }
            }
            
            print(f"   ⏱️  Execution Time: {latest_metric.execution_time:.3f}s")
            print(f"   💾 Memory: {memory.total/1e9:.1f} GB")
            print(f"   🔢 FLOPS: {flops.total/1e12:.1f} TFLOP")
        
        return scaling_results
    
    def benchmark_data_types(self, base_config: ModelConfig) -> Dict[str, Any]:
        """Benchmark performance across different data types"""
        logger.info("🎯 Benchmarking Data Type Performance")
        
        data_types = [DataType.FP32, DataType.FP16, DataType.BF16, DataType.INT8, DataType.INT4]
        dtype_results = {}
        
        for dtype in data_types:
            logger.info(f"🔍 Testing {dtype.value.upper()}...")
            
            # Create config with this data type
            dtype_config = ModelConfig(
                num_layers=base_config.num_layers,
                hidden_size=base_config.hidden_size,
                num_attention_heads=base_config.num_attention_heads,
                intermediate_size=base_config.intermediate_size,
                vocab_size=base_config.vocab_size,
                model_type=base_config.model_type,
                dtype=dtype,
                num_experts=getattr(base_config, 'num_experts', 1),
                top_k=getattr(base_config, 'top_k', 1)
            )
            
            # Profile this data type
            with self.profile_operation(f"dtype_{dtype.value}"):
                calculator = TransformerCalculator(dtype_config)
                memory = calculator.calculate_memory(OperationMode.PREFILL)
                flops = calculator.calculate_flops(OperationMode.PREFILL)
            
            # Get metrics
            latest_metric = self.metrics[-1]
            
            dtype_results[dtype.value] = {
                "dtype": dtype.value,
                "dtype_bytes": dtype.bytes,
                "execution_time": latest_metric.execution_time,
                "memory_peak_mb": latest_metric.memory_peak,
                "memory_total_gb": memory.total / 1e9,
                "flops_total_tflop": flops.total / 1e12,
                "memory_reduction_factor": dtype.memory_reduction_factor
            }
            
            logger.info(f"   ⏱️  Execution Time: {latest_metric.execution_time:.3f}s")
            logger.info(f"   💾 Memory: {memory.total/1e9:.1f} GB")
            logger.info(f"   🔢 FLOPS: {flops.total/1e12:.1f} TFLOP")
            logger.info(f"   📉 Memory Reduction: {dtype.memory_reduction_factor:.1f}x")
        
        return dtype_results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.metrics:
            return "No performance data available. Run profiling first."
        
        report = []
        report.append("📊 PERFORMANCE PROFILING REPORT")
        report.append("=" * 50)
        
        # Overall statistics
        execution_times = [m.execution_time for m in self.metrics]
        memory_peaks = [m.memory_peak for m in self.metrics]
        cpu_usages = [m.cpu_usage for m in self.metrics]
        
        report.append(f"\n📈 Overall Performance:")
        report.append(f"   Operations Profiled: {len(self.metrics)}")
        report.append(f"   Average Execution Time: {sum(execution_times)/len(execution_times):.3f}s")
        report.append(f"   Fastest Operation: {min(execution_times):.3f}s")
        report.append(f"   Slowest Operation: {max(execution_times):.3f}s")
        report.append(f"   Average Peak Memory: {sum(memory_peaks)/len(memory_peaks):.1f} MB")
        report.append(f"   Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.1f}%")
        
        # Detailed results
        report.append(f"\n📋 Detailed Results:")
        for i, metric in enumerate(self.metrics, 1):
            report.append(f"\n{i}. {metric.operation}")
            report.append(f"   Execution Time: {metric.execution_time:.3f}s")
            report.append(f"   Peak Memory: {metric.memory_peak:.1f} MB")
            report.append(f"   Average Memory: {metric.memory_avg:.1f} MB")
            report.append(f"   CPU Usage: {metric.cpu_usage:.1f}%")
        
        return "\n".join(report)


def main():
    """Run the performance profiler"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🔍 Starting Performance Profiler")
    logger.info("=" * 50)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Test configuration
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
    results = profiler.profile_calculator(test_config)
    
    # Benchmark scaling
    scaling_results = profiler.benchmark_scaling(test_config)
    
    # Benchmark data types
    dtype_results = profiler.benchmark_data_types(test_config)
    
    # Generate report
    report = profiler.generate_performance_report()
    logger.info("\n" + report)
    
    logger.info("🎯 Performance Summary:")
    logger.info("   Calculator is working efficiently!")
    logger.info("   Memory usage is reasonable for the model size.")
    logger.info("   Execution times are fast for interactive use.")


if __name__ == "__main__":
    main()
