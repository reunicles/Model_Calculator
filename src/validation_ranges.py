"""
Validation Ranges Based on Actual Model Statistics

This module provides realistic validation ranges based on empirical data
from actual transformer models, ensuring meaningful validation.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelSize(Enum):
    """Model size categories based on parameter count"""
    SMALL = "small"      # 1B-7B parameters
    MEDIUM = "medium"    # 7B-13B parameters  
    LARGE = "large"      # 13B-70B parameters
    XLARGE = "xlarge"    # 70B+ parameters

@dataclass
class ModelStatistics:
    """Real-world model statistics for validation"""
    model_name: str
    parameter_count: int
    hidden_size: int
    num_layers: int
    num_heads: int
    memory_usage_gb: float
    flops_tflop: float
    data_type: str
    notes: str = ""

# Empirical data from real models
REAL_MODEL_STATS = [
    # Small models (1B-7B)
    ModelStatistics("Llama-2-7B", 7_000_000_000, 4096, 32, 32, 14.0, 2.1, "BF16", "Standard 7B model"),
    ModelStatistics("Mistral-7B", 7_200_000_000, 4096, 32, 32, 14.4, 2.2, "BF16", "Mistral 7B"),
    ModelStatistics("Qwen2.5-7B", 7_000_000_000, 4096, 32, 32, 14.0, 2.1, "BF16", "Qwen 2.5 7B"),
    
    # Medium models (7B-13B)
    ModelStatistics("Llama-2-13B", 13_000_000_000, 5120, 40, 40, 26.0, 4.2, "BF16", "13B parameter model"),
    ModelStatistics("Mistral-7B-Instruct", 7_200_000_000, 4096, 32, 32, 14.4, 2.2, "BF16", "Instruct-tuned 7B"),
    
    # Large models (13B-70B)
    ModelStatistics("Llama-2-70B", 70_000_000_000, 8192, 80, 64, 140.0, 14.0, "BF16", "Large 70B model"),
    ModelStatistics("Mixtral-8x7B", 45_000_000_000, 4096, 32, 32, 90.0, 4.5, "BF16", "MoE with 8 experts"),
    
    # XLarge models (70B+)
    ModelStatistics("Llama-2-70B-Chat", 70_000_000_000, 8192, 80, 64, 140.0, 14.0, "BF16", "Chat-tuned 70B"),
]

class ValidationRangeCalculator:
    """Calculate realistic validation ranges based on model statistics"""
    
    def __init__(self):
        self.stats_by_size = self._categorize_by_size()
    
    def _categorize_by_size(self) -> Dict[ModelSize, List[ModelStatistics]]:
        """Categorize models by size"""
        stats_by_size = {size: [] for size in ModelSize}
        
        for stat in REAL_MODEL_STATS:
            if stat.parameter_count <= 7_000_000_000:
                stats_by_size[ModelSize.SMALL].append(stat)
            elif stat.parameter_count <= 13_000_000_000:
                stats_by_size[ModelSize.MEDIUM].append(stat)
            elif stat.parameter_count <= 70_000_000_000:
                stats_by_size[ModelSize.LARGE].append(stat)
            else:
                stats_by_size[ModelSize.XLARGE].append(stat)
        
        return stats_by_size
    
    def get_validation_ranges(self, parameter_count: int, 
                            data_type: str = "BF16",
                            sequence_length: int = 2048,
                            batch_size: int = 1) -> Dict[str, Tuple[float, float]]:
        """
        Get realistic validation ranges based on model size and configuration
        
        Args:
            parameter_count: Number of model parameters
            data_type: Data type (affects memory usage)
            sequence_length: Sequence length (affects memory and FLOPS)
            batch_size: Batch size (affects memory and FLOPS)
        
        Returns:
            Dictionary with validation ranges for memory (GB) and FLOPS (TFLOP)
        """
        # Determine model size category
        if parameter_count <= 7_000_000_000:
            size_category = ModelSize.SMALL
        elif parameter_count <= 13_000_000_000:
            size_category = ModelSize.MEDIUM
        elif parameter_count <= 70_000_000_000:
            size_category = ModelSize.LARGE
        else:
            size_category = ModelSize.XLARGE
        
        # Get statistics for this size category
        stats = self.stats_by_size[size_category]
        if not stats:
            # Fallback to parameter-based estimation
            return self._estimate_ranges_from_parameters(parameter_count, data_type, sequence_length, batch_size)
        
        # Calculate ranges based on real model statistics
        memory_values = [stat.memory_usage_gb for stat in stats]
        flops_values = [stat.flops_tflop for stat in stats]
        
        # Apply scaling factors for sequence length and batch size
        seq_scale = sequence_length / 2048  # Normalize to 2048
        batch_scale = batch_size / 1  # Normalize to batch size 1
        
        # Memory scales roughly linearly with sequence length and batch size
        memory_scale = seq_scale * batch_scale
        # FLOPS scale quadratically with sequence length and linearly with batch size
        flops_scale = (seq_scale ** 2) * batch_scale
        
        # Apply data type scaling
        dtype_scale = self._get_dtype_memory_scale(data_type)
        
        # Calculate ranges with tolerance
        memory_min = min(memory_values) * memory_scale * dtype_scale * 0.8  # 20% tolerance
        memory_max = max(memory_values) * memory_scale * dtype_scale * 1.2  # 20% tolerance
        
        flops_min = min(flops_values) * flops_scale * 0.8  # 20% tolerance
        flops_max = max(flops_values) * flops_scale * 1.2  # 20% tolerance
        
        return {
            "memory_range": (memory_min, memory_max),
            "flops_range": (flops_min, flops_max),
            "size_category": size_category.value,
            "based_on_models": len(stats)
        }
    
    def _estimate_ranges_from_parameters(self, parameter_count: int, 
                                       data_type: str, 
                                       sequence_length: int, 
                                       batch_size: int) -> Dict[str, Tuple[float, float]]:
        """Fallback estimation based on parameter count"""
        # Rough estimation formulas based on empirical data
        bytes_per_param = self._get_dtype_memory_scale(data_type) * 2  # 2 bytes for BF16 baseline
        
        # Memory estimation (parameters + activations)
        param_memory_gb = (parameter_count * bytes_per_param) / 1e9
        activation_memory_gb = (parameter_count * sequence_length * batch_size * bytes_per_param) / 1e12
        
        total_memory_gb = param_memory_gb + activation_memory_gb
        
        # FLOPS estimation (rough approximation)
        flops_per_param = 2  # Rough estimate
        total_flops_tflop = (parameter_count * sequence_length * batch_size * flops_per_param) / 1e12
        
        return {
            "memory_range": (total_memory_gb * 0.7, total_memory_gb * 1.3),
            "flops_range": (total_flops_tflop * 0.7, total_flops_tflop * 1.3),
            "size_category": "estimated",
            "based_on_models": 0
        }
    
    def _get_dtype_memory_scale(self, data_type: str) -> float:
        """Get memory scaling factor for data type"""
        dtype_scales = {
            "FP32": 2.0,
            "FP16": 1.0,
            "BF16": 1.0,
            "INT8": 0.5,
            "INT4": 0.25,
            "FP8": 0.5
        }
        return dtype_scales.get(data_type.upper(), 1.0)
    
    def get_model_specific_ranges(self, model_name: str, 
                                parameter_count: int,
                                **kwargs) -> Dict[str, Tuple[float, float]]:
        """Get validation ranges for a specific model if available"""
        # Look for exact model match
        for stat in REAL_MODEL_STATS:
            if stat.model_name.lower() in model_name.lower():
                return self.get_validation_ranges(stat.parameter_count, **kwargs)
        
        # Fallback to parameter-based estimation
        return self.get_validation_ranges(parameter_count, **kwargs)

# Global instance for easy access
validation_calculator = ValidationRangeCalculator()

def get_validation_ranges(parameter_count: int, 
                         data_type: str = "BF16",
                         sequence_length: int = 2048,
                         batch_size: int = 1,
                         model_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """
    Get realistic validation ranges for a model
    
    Args:
        parameter_count: Number of model parameters
        data_type: Data type (FP32, FP16, BF16, INT8, INT4, FP8)
        sequence_length: Sequence length
        batch_size: Batch size
        model_name: Optional model name for specific matching
    
    Returns:
        Dictionary with validation ranges and metadata
    """
    if model_name:
        return validation_calculator.get_model_specific_ranges(
            model_name, parameter_count, 
            data_type=data_type, 
            sequence_length=sequence_length, 
            batch_size=batch_size
        )
    else:
        return validation_calculator.get_validation_ranges(
            parameter_count, 
            data_type=data_type, 
            sequence_length=sequence_length, 
            batch_size=batch_size
        )
