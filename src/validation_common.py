"""
Common Validation Functions

This module provides shared functionality for both benchmark and real-world validation suites,
reducing duplication and ensuring consistent validation logic.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from transformer_calculator import (
    TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
    MemoryComponents, FLOPSComponents, ReuseComponents
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from a validation test"""
    test_name: str
    model_config: ModelConfig
    mode: OperationMode
    memory: MemoryComponents
    flops: FLOPSComponents
    reuse: ReuseComponents
    execution_time: float
    memory_peak: float
    cpu_usage: float
    accuracy_score: float = 0.0
    validation_passed: bool = False
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
    test_modes: List[OperationMode] = None
    notes: str = ""

class ValidationRunner:
    """Common validation runner for both benchmark and real-world tests"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def run_validation_case(self, case: ValidationCase) -> ValidationResult:
        """Run a single validation case"""
        logger.info(f"🧪 Running validation: {case.name}")
        
        # Create model config
        config = self._create_model_config(case.model_config)
        
        # Test each mode
        for mode in (case.test_modes or [OperationMode.PREFILL]):
            result = self._run_single_test(case, config, mode)
            self.results.append(result)
            
            # Log results
            status = "✅ PASS" if result.validation_passed else "❌ FAIL"
            logger.info(f"   {mode.value}: {status} (Score: {result.accuracy_score:.2f})")
        
        return self.results[-1] if self.results else None
    
    def _create_model_config(self, config_dict: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary"""
        return ModelConfig(
            num_layers=config_dict.get('num_layers', 32),
            hidden_size=config_dict.get('hidden_size', 4096),
            num_attention_heads=config_dict.get('num_attention_heads', 32),
            num_key_value_heads=config_dict.get('num_key_value_heads', 32),
            intermediate_size=config_dict.get('intermediate_size', 16384),
            vocab_size=config_dict.get('vocab_size', 50257),
            model_type=ModelType(config_dict.get('model_type', 'dense')),
            num_experts=config_dict.get('num_experts', 1),
            top_k=config_dict.get('top_k', 1),
            sequence_length=config_dict.get('sequence_length', 2048),
            batch_size=config_dict.get('batch_size', 1),
            decode_len=config_dict.get('decode_len', 64),
            dtype=DataType(config_dict.get('dtype', 'bf16'))
        )
    
    def _run_single_test(self, case: ValidationCase, config: ModelConfig, 
                        mode: OperationMode) -> ValidationResult:
        """Run a single test with timing and resource monitoring"""
        start_time = time.time()
        
        try:
            # Create calculator and run calculations
            calculator = TransformerCalculator(config)
            
            # Monitor memory usage
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run calculations
            memory = calculator.calculate_memory(mode)
            flops = calculator.calculate_flops(mode)
            reuse = calculator.calculate_reuse(mode)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_peak = final_memory - initial_memory
            
            execution_time = time.time() - start_time
            
            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(
                case, memory, flops, reuse, mode
            )
            
            # Determine if validation passed
            validation_passed = accuracy_score >= (1.0 - case.tolerance)
            
            return ValidationResult(
                test_name=f"{case.name}_{mode.value}",
                model_config=config,
                mode=mode,
                memory=memory,
                flops=flops,
                reuse=reuse,
                execution_time=execution_time,
                memory_peak=memory_peak,
                cpu_usage=0.0,  # Would need more complex monitoring
                accuracy_score=accuracy_score,
                validation_passed=validation_passed,
                notes=case.notes
            )
            
        except Exception as e:
            logger.error(f"❌ Test failed: {case.name} - {str(e)}")
            return ValidationResult(
                test_name=f"{case.name}_{mode.value}",
                model_config=config,
                mode=mode,
                memory=MemoryComponents(),
                flops=FLOPSComponents(),
                reuse=ReuseComponents(),
                execution_time=time.time() - start_time,
                memory_peak=0.0,
                cpu_usage=0.0,
                accuracy_score=0.0,
                validation_passed=False,
                notes=f"Error: {str(e)}"
            )
    
    def _calculate_accuracy_score(self, case: ValidationCase, 
                                 memory: MemoryComponents, 
                                 flops: FLOPSComponents, 
                                 reuse: ReuseComponents,
                                 mode: OperationMode) -> float:
        """Calculate accuracy score based on expected ranges"""
        scores = []
        
        # Memory accuracy
        memory_gb = memory.total / 1e9
        if case.expected_memory_range[0] <= memory_gb <= case.expected_memory_range[1]:
            scores.append(1.0)
        else:
            # Calculate how far off we are
            if memory_gb < case.expected_memory_range[0]:
                ratio = memory_gb / case.expected_memory_range[0]
            else:
                ratio = case.expected_memory_range[1] / memory_gb
            scores.append(max(0.0, ratio))
        
        # FLOPS accuracy
        flops_tflop = flops.total / 1e12
        if case.expected_flops_range[0] <= flops_tflop <= case.expected_flops_range[1]:
            scores.append(1.0)
        else:
            # Calculate how far off we are
            if flops_tflop < case.expected_flops_range[0]:
                ratio = flops_tflop / case.expected_flops_range[0]
            else:
                ratio = case.expected_flops_range[1] / flops_tflop
            scores.append(max(0.0, ratio))
        
        # Reuse accuracy (if specified)
        if case.expected_reuse_range[0] > 0:
            reuse_value = reuse.total
            if case.expected_reuse_range[0] <= reuse_value <= case.expected_reuse_range[1]:
                scores.append(1.0)
            else:
                # Calculate how far off we are
                if reuse_value < case.expected_reuse_range[0]:
                    ratio = reuse_value / case.expected_reuse_range[0]
                else:
                    ratio = case.expected_reuse_range[1] / reuse_value
                scores.append(max(0.0, ratio))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        if not self.results:
            return "No validation results available."
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.validation_passed)
        failed_tests = total_tests - passed_tests
        
        avg_accuracy = sum(r.accuracy_score for r in self.results) / total_tests
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report.append(f"Average Accuracy: {avg_accuracy:.3f}")
        report.append(f"Average Execution Time: {avg_execution_time:.3f}s")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 60)
        for result in self.results:
            status = "✅ PASS" if result.validation_passed else "❌ FAIL"
            report.append(f"{status} {result.test_name}")
            report.append(f"   Accuracy: {result.accuracy_score:.3f}")
            report.append(f"   Memory: {result.memory.total/1e9:.1f} GB")
            report.append(f"   FLOPS: {result.flops.total/1e12:.1f} TFLOP")
            report.append(f"   Reuse: {result.reuse.total:.1f} FLOPS/byte")
            report.append(f"   Time: {result.execution_time:.3f}s")
            if result.notes:
                report.append(f"   Notes: {result.notes}")
            report.append("")
        
        return "\n".join(report)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for validation results"""
        if not self.results:
            return {}
        
        return {
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.validation_passed),
            "failed_tests": sum(1 for r in self.results if not r.validation_passed),
            "pass_rate": sum(1 for r in self.results if r.validation_passed) / len(self.results),
            "avg_accuracy": sum(r.accuracy_score for r in self.results) / len(self.results),
            "avg_execution_time": sum(r.execution_time for r in self.results) / len(self.results),
            "total_execution_time": time.time() - self.start_time
        }

def create_standard_validation_cases() -> List[ValidationCase]:
    """Create standard validation cases for common model configurations"""
    cases = []
    
    # Small model (7B)
    cases.append(ValidationCase(
        name="7B_Dense_Model",
        model_config={
            'num_layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'num_key_value_heads': 32,
            'intermediate_size': 16384,
            'vocab_size': 50257,
            'model_type': 'dense',
            'sequence_length': 2048,
            'batch_size': 1,
            'dtype': 'bf16'
        },
        expected_memory_range=(10, 20),  # GB
        expected_flops_range=(1, 3),     # TFLOP
        expected_reuse_range=(0, 0),    # Not specified
        test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
        notes="Standard 7B dense model"
    ))
    
    # Medium model (13B)
    cases.append(ValidationCase(
        name="13B_Dense_Model",
        model_config={
            'num_layers': 40,
            'hidden_size': 5120,
            'num_attention_heads': 40,
            'num_key_value_heads': 40,
            'intermediate_size': 20480,
            'vocab_size': 50257,
            'model_type': 'dense',
            'sequence_length': 2048,
            'batch_size': 1,
            'dtype': 'bf16'
        },
        expected_memory_range=(20, 40),  # GB
        expected_flops_range=(2, 6),     # TFLOP
        expected_reuse_range=(0, 0),    # Not specified
        test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
        notes="Standard 13B dense model"
    ))
    
    # MoE model
    cases.append(ValidationCase(
        name="MoE_Model",
        model_config={
            'num_layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'intermediate_size': 16384,
            'vocab_size': 50257,
            'model_type': 'moe',
            'num_experts': 8,
            'top_k': 2,
            'sequence_length': 2048,
            'batch_size': 1,
            'dtype': 'bf16'
        },
        expected_memory_range=(15, 35),  # GB
        expected_flops_range=(1, 4),     # TFLOP
        expected_reuse_range=(0, 0),    # Not specified
        test_modes=[OperationMode.PREFILL, OperationMode.DECODE],
        notes="MoE model with 8 experts"
    ))
    
    return cases
