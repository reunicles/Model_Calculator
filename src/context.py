"""
Context Classes for Transformer Calculations

This module provides dataclasses for MemoryContext and CalculationContext
used throughout the transformer calculator.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UnifiedContext:
    """Unified context for all transformer calculations (memory, FLOPS, reuse)"""
    # Core parameters
    seq_len: int
    batch_size: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    dtype_bytes: float
    num_layers: int
    vocab_size: int
    model_type: str
    mode: str  # Operation mode: 'pretraining', 'prefill', 'decode'
    
    # MoE parameters
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    expert_capacity_factor: Optional[float] = None
    moe_intermediate_size: Optional[int] = None
    
    # Backward compatibility aliases
    @property
    def MemoryContext(self):
        """Backward compatibility for MemoryContext"""
        return self
    
    @property 
    def CalculationContext(self):
        """Backward compatibility for CalculationContext"""
        return self


# Backward compatibility aliases
MemoryContext = UnifiedContext
CalculationContext = UnifiedContext
