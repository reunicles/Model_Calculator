"""
Hugging Face Model Metadata Parser

This module parses Hugging Face model metadata to extract official parameter counts
and compare them with our calculated values.
"""

import requests
import re
from typing import Optional, Dict, Any
import json


def parse_hf_model_metadata(repo_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse Hugging Face model metadata to extract parameter count and other info.
    
    Args:
        repo_id: Hugging Face model ID (e.g., 'deepseek-ai/DeepSeek-V3.1-Terminus')
        
    Returns:
        Dictionary with metadata or None if parsing fails
    """
    try:
        # Construct the model page URL
        url = f"https://huggingface.co/{repo_id}"
        
        # Fetch the model page
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Look for the model size in the page content
        content = response.text
        
        # Pattern to match "Model size: X.XB params" or "X.XB params"
        model_size_patterns = [
            r'Model size[:\s]*(\d+\.?\d*)\s*B\s*params',
            r'(\d+\.?\d*)\s*B\s*params',
            r'"model_size":\s*"(\d+\.?\d*)\s*B\s*params"',
            r'model_size[:\s]*(\d+\.?\d*)\s*B\s*params'
        ]
        
        model_size = None
        for pattern in model_size_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                model_size = float(match.group(1))
                break
        
        # Look for tensor type information
        tensor_type_patterns = [
            r'Tensor type[:\s]*([^<>\n]+)',
            r'"tensor_type":\s*"([^"]+)"'
        ]
        
        tensor_type = None
        for pattern in tensor_type_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                tensor_type = match.group(1).strip()
                break
        
        # Look for model type (MoE, Dense, etc.)
        model_type_patterns = [
            r'Model type[:\s]*([^<>\n]+)',
            r'"model_type":\s*"([^"]+)"'
        ]
        
        model_type = None
        for pattern in model_type_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                model_type = match.group(1).strip()
                break
        
        if model_size is None:
            return None
            
        return {
            'model_size_billions': model_size,
            'model_size_parameters': model_size * 1e9,
            'tensor_type': tensor_type,
            'model_type': model_type,
            'repo_id': repo_id
        }
        
    except Exception as e:
        print(f"Error parsing metadata for {repo_id}: {e}")
        return None


def compare_parameters(calculated_params: float, official_params: float, 
                      tolerance_percent: float = 20.0) -> Dict[str, Any]:
    """
    Compare calculated parameters with official metadata.
    
    Args:
        calculated_params: Our calculated parameter count
        official_params: Official parameter count from HF metadata
        tolerance_percent: Allowed discrepancy percentage
        
    Returns:
        Dictionary with comparison results
    """
    if official_params == 0:
        return {
            'match': False,
            'error': 'Official parameter count is zero',
            'calculated': calculated_params,
            'official': official_params,
            'difference': float('inf'),
            'difference_percent': float('inf')
        }
    
    difference = abs(calculated_params - official_params)
    difference_percent = (difference / official_params) * 100
    
    match = difference_percent <= tolerance_percent
    
    return {
        'match': match,
        'calculated': calculated_params,
        'official': official_params,
        'difference': difference,
        'difference_percent': difference_percent,
        'tolerance_percent': tolerance_percent
    }


def validate_model_parameters(repo_id: str, calculated_params: float, 
                            tolerance_percent: float = 20.0) -> Dict[str, Any]:
    """
    Validate calculated parameters against Hugging Face metadata.
    
    Args:
        repo_id: Hugging Face model ID
        calculated_params: Our calculated parameter count
        tolerance_percent: Allowed discrepancy percentage
        
    Returns:
        Dictionary with validation results
    """
    metadata = parse_hf_model_metadata(repo_id)
    
    if metadata is None:
        return {
            'valid': False,
            'error': 'Could not parse Hugging Face metadata',
            'calculated': calculated_params,
            'official': None
        }
    
    comparison = compare_parameters(
        calculated_params, 
        metadata['model_size_parameters'], 
        tolerance_percent
    )
    
    return {
        'valid': comparison['match'],
        'calculated': comparison['calculated'],
        'official': comparison['official'],
        'difference_percent': comparison['difference_percent'],
        'tolerance_percent': comparison['tolerance_percent'],
        'metadata': metadata,
        'error': None if comparison['match'] else f"Parameter count discrepancy: {comparison['difference_percent']:.1f}% (calculated: {comparison['calculated']:,.0f}, official: {comparison['official']:,.0f})"
    }


if __name__ == "__main__":
    # Test the parser
    test_models = [
        "deepseek-ai/DeepSeek-V3.1-Terminus",
        "baidu/ERNIE-4.5-21B-A3B-Thinking",
        "Qwen/Qwen2.5-72B"
    ]
    
    for model in test_models:
        print(f"\nTesting {model}:")
        metadata = parse_hf_model_metadata(model)
        if metadata:
            print(f"  Model size: {metadata['model_size_billions']}B parameters")
            print(f"  Tensor type: {metadata['tensor_type']}")
            print(f"  Model type: {metadata['model_type']}")
        else:
            print("  Could not parse metadata")

