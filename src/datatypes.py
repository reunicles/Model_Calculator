"""
Data Type Definitions and Detection Utilities

This module provides data type definitions, detection utilities, and conversion functions
for the transformer calculator.
"""

import re
from typing import Dict, Any, Optional
from enum import Enum


class DataType(Enum):
    """Supported data types for model calculations"""
    FP32 = "fp32"
    FP16 = "fp16" 
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    
    @property
    def bytes(self) -> float:
        """Get the number of bytes per parameter for this data type"""
        byte_sizes = {
            DataType.FP32: 4.0,
            DataType.FP16: 2.0,
            DataType.BF16: 2.0,
            DataType.INT8: 1.0,
            DataType.INT4: 0.5,
            DataType.FP8: 1.0
        }
        return byte_sizes[self]
    
    @property
    def memory_reduction_factor(self) -> float:
        """Get memory reduction factor compared to FP32"""
        return 4.0 / self.bytes
    
    @classmethod
    def from_string(cls, dtype_str: str) -> 'DataType':
        """Create DataType from string representation"""
        dtype_str = dtype_str.lower().strip()
        dtype_map = {
            'fp32': cls.FP32,
            'float32': cls.FP32,
            'fp16': cls.FP16,
            'float16': cls.FP16,
            'bf16': cls.BF16,
            'bfloat16': cls.BF16,
            'int8': cls.INT8,
            'int4': cls.INT4,
            'fp8': cls.FP8,
            'float8': cls.FP8
        }
        
        if dtype_str in dtype_map:
            return dtype_map[dtype_str]
        else:
            raise ValueError(f"Unsupported data type: {dtype_str}")


class DataTypeDetector:
    """Detects data type from Hugging Face config and model names"""
    
    @staticmethod
    def detect_from_config(config: dict) -> DataType:
        """Detect data type from Hugging Face config.json"""
        
        # Check for explicit torch_dtype
        if 'torch_dtype' in config:
            return DataType.from_string(config['torch_dtype'])
        
        # Check for quantization config
        if 'quantization_config' in config:
            quant_config = config['quantization_config']
            if 'bits' in quant_config:
                bits = quant_config['bits']
                if bits == 4:
                    return DataType.INT4
                elif bits == 8:
                    return DataType.INT8
        
        # Check for model-specific hints
        model_name = config.get('model_type', '').lower()
        if 'qwen' in model_name or 'deepseek' in model_name:
            return DataType.BF16  # These models typically use BF16
        
        # Default to BF16 for modern models
        return DataType.BF16
    
    @staticmethod
    def detect_from_model_name(model_name: str) -> DataType:
        """Detect data type from model name patterns"""
        model_name = model_name.lower()
        
        # Check for explicit data type indicators
        if 'fp32' in model_name or 'float32' in model_name:
            return DataType.FP32
        elif 'fp16' in model_name or 'float16' in model_name:
            return DataType.FP16
        elif 'bf16' in model_name or 'bfloat16' in model_name:
            return DataType.BF16
        elif 'int8' in model_name:
            return DataType.INT8
        elif 'int4' in model_name:
            return DataType.INT4
        elif 'fp8' in model_name or 'float8' in model_name:
            return DataType.FP8
        
        # Check for quantization indicators
        if 'quantized' in model_name or 'quant' in model_name:
            if '4bit' in model_name or '4-bit' in model_name:
                return DataType.INT4
            elif '8bit' in model_name or '8-bit' in model_name:
                return DataType.INT8
        
        # Default to BF16 for modern models
        return DataType.BF16
