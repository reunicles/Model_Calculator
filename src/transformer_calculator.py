"""
Transformer Model Memory and FLOPS Calculator

This module provides comprehensive calculations for HBM memory usage and GPU compute FLOPS
for transformer models, supporting both Dense and MoE architectures.

Key Features:
- Memory calculations for HBM (High Bandwidth Memory) usage
- FLOPS calculations for GPU compute requirements
- Support for Dense and MoE (Mixture of Experts) architectures
- Flash Attention and Flash Decode optimizations
- Arithmetic Intensity (Reuse) calculations
- Comprehensive validation and benchmarking

Mathematical Foundations:
- Attention Memory: O(seq_len²) for standard attention, O(seq_len) for Flash Attention
- FLOPS: Quadratic in sequence length for attention, linear for other components
- MoE: Expert routing and capacity management calculations
- Memory Units: All memory calculations in bytes
- FLOPS Units: All FLOPS calculations in floating-point operations

Architecture Support:
- Dense Transformers: Standard transformer with all parameters active
- MoE Transformers: Mixture of Experts with selective activation
- Flash Attention: Memory-efficient attention for training and inference
- Flash Decode: Optimized attention for auto-regressive generation

Author: AI Assistant
Version: 2.0
Last Updated: 2025
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Import the new modules
try:
    from .datatypes import DataType, DataTypeDetector
    from .context import MemoryContext, CalculationContext
    from .table_formatter import TableFormatter
except ImportError:
    # Fallback for when run as script
    from datatypes import DataType, DataTypeDetector
    from context import MemoryContext, CalculationContext
    from table_formatter import TableFormatter


def calculate_model_parameters(config: 'ModelConfig') -> int:
    """Calculate total number of model parameters (centralized function for dense vs MoE)"""
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    
    # Embedding parameters
    embedding_params = vocab_size * hidden_size
    
    # Per-layer parameters
    # Attention: Q, K, V projections + output projection
    # For GQA/MQA, K and V projections are scaled by num_kv_heads/num_heads
    kv_scale = num_kv_heads / num_heads
    attention_params = hidden_size * hidden_size + 2 * hidden_size * hidden_size * kv_scale + hidden_size * hidden_size
    
    # MLP parameters
    if config.model_type == ModelType.MOE:
        # For MoE, use moe_intermediate_size if available, otherwise fall back to intermediate_size
        moe_size = config.moe_intermediate_size if config.moe_intermediate_size is not None else intermediate_size
        
        # Regular routed experts
        routed_experts = config.num_experts
        routed_mlp_params = routed_experts * 2 * hidden_size * moe_size
        
        # Shared experts (if any)
        shared_experts = getattr(config, 'num_shared_experts', 0)
        shared_mlp_params = shared_experts * 2 * hidden_size * moe_size
        
        mlp_params = routed_mlp_params + shared_mlp_params
    else:
        mlp_params = 2 * hidden_size * intermediate_size
    
    # Layer norm parameters (2 per layer: pre-attention, pre-MLP)
    layer_norm_params = 2 * hidden_size
    
    per_layer_params = attention_params + mlp_params + layer_norm_params
    total_params = embedding_params + num_layers * per_layer_params
    
    return total_params


def validate_parameters_with_hf_metadata(config: 'ModelConfig', repo_id: str, 
                                       tolerance_percent: float = 20.0) -> Dict[str, Any]:
    """
    Validate calculated parameters against Hugging Face metadata.
    
    Args:
        config: Model configuration
        repo_id: Hugging Face model ID
        tolerance_percent: Allowed discrepancy percentage
        
    Returns:
        Dictionary with validation results
    """
    try:
        from .hf_metadata_parser import validate_model_parameters
    except ImportError:
        # Fallback for when run as script
        from hf_metadata_parser import validate_model_parameters
    
    calculated_params = calculate_model_parameters(config)
    return validate_model_parameters(repo_id, calculated_params, tolerance_percent)


class ModelType(Enum):
    DENSE = "dense"
    MOE = "moe"


class OperationMode(Enum):
    PRETRAINING = "pretraining"
    PREFILL = "prefill"
    DECODE = "decode"










class ComponentCalculator:
    """Unified calculator for all component calculations"""
    
    def __init__(self, config: 'ModelConfig'):
        self.config = config
    
    def calculate_attention_components(self, context: CalculationContext) -> Tuple[float, float]:
        """Calculate attention memory and FLOPS with realistic Flash Attention modeling"""
        # Memory calculation
        if self.config.use_flash_attention and context.seq_len >= self.config.flash_attention_block_size:
            # Flash Attention: Realistic memory traffic modeling
            attention_memory = self._calculate_flash_attention_memory(context)
        else:
            # Standard attention: O(seq_len^2) memory
            attention_scores = context.seq_len * context.seq_len * context.batch_size * context.num_heads * context.dtype_bytes
            attention_output = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
            attention_memory = attention_scores + attention_output
        
        # FLOPS calculation
        hidden_per_head = context.hidden_size // context.num_heads
        
        # Mode-dependent attention FLOPS calculation
        if context.mode == 'decode':
            # Decode: Memory-bound operation with O(seq_len) FLOPS
            # Math_Used.md: L × [5Bd² + 2BdS] for decode
            # Attention part: 2BdS (Q @ K^T for 1 token against S cached keys)
            full_seq_len = self.config.sequence_length  # Use full sequence length, not context.seq_len
            attention_flops = 2 * context.batch_size * context.hidden_size * full_seq_len
        else:
            # Pretraining/Prefill: Compute-bound operation with O(seq_len^2) FLOPS
            # Math_Used.md: L × [5BSd² + 2BS²d] for prefill
            # Attention part: 2BS²d (Q @ K^T for all positions)
            if self.config.use_flash_attention and context.seq_len >= self.config.flash_attention_block_size:
                # Flash Attention: Still O(seq_len^2) FLOPS but with memory optimization
                # Math_Used.md formula: 2BS²d
                attention_flops = 2 * context.seq_len * context.seq_len * context.batch_size * context.hidden_size
            else:
                # Standard attention: O(seq_len^2) FLOPS
                # Math_Used.md formula: 2BS²d
                attention_flops = 2 * context.seq_len * context.seq_len * context.batch_size * context.hidden_size
        
        return attention_memory, attention_flops
    
    def _calculate_flash_attention_memory(self, context: CalculationContext) -> float:
        """
        Calculate realistic Flash Attention memory traffic
        
        Flash Attention uses tiling and recomputation to reduce memory from O(seq_len²) to O(seq_len).
        This method models the actual memory traffic patterns including:
        - Q, K, V tile loading patterns
        - Attention score recomputation overhead
        - Block-wise memory access patterns
        - Sequence length bucketing for different behaviors
        """
        block_size = self.config.flash_attention_block_size
        seq_len = context.seq_len
        batch_size = context.batch_size
        num_heads = context.num_heads
        hidden_size = context.hidden_size
        dtype_bytes = context.dtype_bytes
        
        # Sequence length bucketing for different Flash Attention behaviors
        if seq_len <= 1024:
            # Short sequences: Minimal tiling overhead
            memory_factor = 1.1  # 10% overhead for tiling
        elif seq_len <= 4096:
            # Medium sequences: Moderate tiling overhead
            memory_factor = 1.2  # 20% overhead for tiling
        elif seq_len <= 16384:
            # Long sequences: Moderate tiling overhead
            memory_factor = 1.5  # 50% overhead for tiling
        elif seq_len <= 65536:
            # Very long sequences: Higher tiling overhead
            memory_factor = 2.0  # 100% overhead for tiling
        else:
            # Extremely long sequences: High tiling overhead
            memory_factor = 2.5  # 150% overhead for tiling
        
        # Flash Attention memory: Block-wise processing
        # Flash Attention reduces memory from O(seq_len^2) to O(seq_len) by processing in blocks
        # But we still need to process all sequence positions, just not all at once
        # Memory scales with seq_len but with much smaller constant factors
        # For very long sequences, we need more memory for intermediate computations
        # Flash Attention still needs significant memory for intermediate computations
        flash_attention_memory = seq_len * batch_size * num_heads * dtype_bytes * 2.0 * memory_factor
        
        # Output projection memory (always needed)
        output_memory = seq_len * batch_size * hidden_size * dtype_bytes
        
        # Total Flash Attention memory (Q, K, V are handled by projections component)
        total_memory = flash_attention_memory + output_memory
        
        return total_memory
    
    def _calculate_flash_decode_memory(self, context: CalculationContext) -> float:
        """
        Calculate realistic Flash Decode memory traffic
        
        Flash Decode uses KV cache optimization to reduce memory from O(seq_len) to O(1).
        This method models the actual memory traffic patterns including:
        - KV cache block-wise loading
        - Attention computation with cached keys/values
        - Sequence length bucketing for different behaviors
        """
        block_size = self.config.flash_decode_block_size if hasattr(self.config, 'flash_decode_block_size') else 1024
        seq_len = context.seq_len
        batch_size = context.batch_size
        num_heads = context.num_heads
        hidden_size = context.hidden_size
        dtype_bytes = context.dtype_bytes
        
        # Sequence length bucketing for different Flash Decode behaviors
        if seq_len <= 512:
            # Short sequences: Minimal cache overhead
            cache_factor = 1.1  # 10% overhead for cache management
        elif seq_len <= 2048:
            # Medium sequences: Moderate cache overhead
            cache_factor = 1.3  # 30% overhead for cache management
        elif seq_len <= 8192:
            # Long sequences: Significant cache overhead
            cache_factor = 1.5  # 50% overhead for cache management
        else:
            # Very long sequences: High cache overhead
            cache_factor = 2.0  # 100% overhead for cache management
        
        # Base memory: Q projection (new token)
        q_memory = seq_len * batch_size * hidden_size * dtype_bytes
        
        # Flash Decode memory: Block-wise KV cache access
        # Each block processes: block_size × batch_size × num_heads
        flash_decode_memory = block_size * batch_size * num_heads * dtype_bytes * cache_factor
        
        # Output projection memory (always needed)
        output_memory = seq_len * batch_size * hidden_size * dtype_bytes
        
        # Total Flash Decode memory
        total_memory = q_memory + flash_decode_memory + output_memory
        
        return total_memory
    
    def calculate_projection_components(self, context: CalculationContext) -> Tuple[float, float]:
        """Calculate projection memory and FLOPS"""
        # Memory calculation - PROJECTION WEIGHTS, not activations
        kv_scale = context.num_kv_heads / context.num_heads
        q_proj_weights = context.hidden_size * context.hidden_size * context.dtype_bytes
        k_proj_weights = context.hidden_size * context.hidden_size * kv_scale * context.dtype_bytes
        v_proj_weights = context.hidden_size * context.hidden_size * kv_scale * context.dtype_bytes
        out_proj_weights = context.hidden_size * context.hidden_size * context.dtype_bytes
        projection_memory = q_proj_weights + k_proj_weights + v_proj_weights + out_proj_weights
        
        # FLOPS calculation
        # Math_Used.md: 3BSd² for Q, K, V projections (QKV Projections row)
        # Q projection: BSd²
        q_flops = context.seq_len * context.batch_size * context.hidden_size * context.hidden_size
        # K projection: BSd² * kv_scale (for GQA/MQA)
        k_flops = context.seq_len * context.batch_size * context.hidden_size * context.hidden_size * kv_scale
        # V projection: BSd² * kv_scale (for GQA/MQA)
        v_flops = context.seq_len * context.batch_size * context.hidden_size * context.hidden_size * kv_scale
        # Output projection: BSd²
        out_flops = context.seq_len * context.batch_size * context.hidden_size * context.hidden_size
        projection_flops = q_flops + k_flops + v_flops + out_flops
        
        return projection_memory, projection_flops
    
    def calculate_mlp_components(self, context: CalculationContext) -> Tuple[float, float]:
        """Calculate MLP memory and FLOPS"""
        if context.model_type == ModelType.MOE:
            # MoE calculation
            moe_calc = MoECalculator(self.config)
            # Create MemoryContext from CalculationContext
            from context import MemoryContext
            memory_context = MemoryContext(
                seq_len=context.seq_len,
                batch_size=context.batch_size,
                hidden_size=context.hidden_size,
                num_heads=context.num_heads,
                num_kv_heads=context.num_kv_heads,
                intermediate_size=context.intermediate_size,
                dtype_bytes=context.dtype_bytes,
                num_layers=context.num_layers,
                vocab_size=context.vocab_size,
                model_type=context.model_type.value,
                mode=context.mode,
                num_experts=context.num_experts,
                top_k=context.top_k,
                expert_capacity_factor=context.expert_capacity_factor,
                moe_intermediate_size=getattr(context, 'moe_intermediate_size', None)
            )
            mlp_memory = moe_calc.calculate_hbm_memory(memory_context)
            mlp_flops = moe_calc.calculate_flops(memory_context)
        else:
            # Dense MLP calculation
            # Math_Used.md: 2BSdd_ff for MLP (MLP Up + MLP Down rows)
            up_proj = context.seq_len * context.batch_size * context.intermediate_size * context.dtype_bytes
            down_proj = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
            mlp_memory = up_proj + down_proj
            
            # Math_Used.md: 2BSdd_ff FLOPS for MLP
            # Up projection: BSdd_ff
            up_flops = context.seq_len * context.batch_size * context.hidden_size * context.intermediate_size
            # Down projection: BSdd_ff
            down_flops = context.seq_len * context.batch_size * context.intermediate_size * context.hidden_size
            mlp_flops = up_flops + down_flops
        
        return mlp_memory, mlp_flops
    
    def calculate_embedding_components(self, context: CalculationContext) -> Tuple[float, float]:
        """Calculate embedding memory and FLOPS"""
        # Memory: token embeddings + positional embeddings
        token_embeddings = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
        positional_embeddings = context.seq_len * context.hidden_size * context.dtype_bytes
        embedding_memory = token_embeddings + positional_embeddings
        
        # FLOPS: embedding lookup and projection
        embedding_flops = context.seq_len * context.batch_size * context.hidden_size * context.hidden_size
        
        return embedding_memory, embedding_flops
    
    def calculate_layer_norm_components(self, context: CalculationContext) -> Tuple[float, float]:
        """Calculate layer norm memory and FLOPS"""
        # Memory: LAYER NORM WEIGHTS (gamma and beta), not activations
        layer_norm_memory = 2 * context.hidden_size * context.dtype_bytes  # Per layer: gamma + beta
        
        # FLOPS: mean, variance, normalization (with multiply-add factor)
        layer_norm_flops = 2 * context.seq_len * context.batch_size * context.hidden_size * 2
        
        return layer_norm_memory, layer_norm_flops


class MemoryCalculator:
    """Calculator for memory calculations with context"""
    
    def __init__(self, config: 'ModelConfig'):
        self.config = config
    
    def create_context(self, seq_len: int, batch_size: int, hidden_size: int, 
                      intermediate_size: int, dtype_bytes: int) -> MemoryContext:
        """Create MemoryContext from parameters"""
        return MemoryContext(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=intermediate_size,
            dtype_bytes=dtype_bytes
        )
    
    def calculate_attention_memory(self, context: MemoryContext) -> float:
        """Calculate attention memory"""
        if self.config.use_flash_attention:
            # Flash Attention: O(seq_len) memory
            return context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
        else:
            # Standard attention: O(seq_len^2) memory
            attention_scores = context.seq_len * context.seq_len * context.batch_size * context.num_heads * context.dtype_bytes
            attention_output = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
            return attention_scores + attention_output
    
    def calculate_projection_memory(self, context: MemoryContext) -> float:
        """Calculate projection memory"""
        # Q, K, V projections with GQA scaling
        kv_scale = context.num_kv_heads / context.num_heads
        q_proj = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
        k_proj = context.seq_len * context.batch_size * context.hidden_size * kv_scale * context.dtype_bytes
        v_proj = context.seq_len * context.batch_size * context.hidden_size * kv_scale * context.dtype_bytes
        out_proj = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
        return q_proj + k_proj + v_proj + out_proj
    
    def calculate_mlp_memory(self, context: MemoryContext) -> float:
        """Calculate MLP memory"""
        if self.config.model_type == ModelType.MOE:
            moe_calc = MoECalculator(self.config)
            return moe_calc.calculate_hbm_memory(context)
        else:
            # Dense MLP: up_proj + down_proj
            up_proj = context.seq_len * context.batch_size * context.intermediate_size * context.dtype_bytes
            down_proj = context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
            return up_proj + down_proj


class ValidationRules:
    """Consolidated validation rules for model configuration"""
    
    @staticmethod
    def validate_moe_config(config: 'ModelConfig') -> List[str]:
        """
        Validate MoE (Mixture of Experts) specific configuration
        
        This function validates that MoE models have the required configuration parameters
        and that they are within reasonable bounds for proper operation.
        
        MoE Requirements:
        - num_experts > 1: Must have multiple experts for routing
        - top_k >= 1: Must select at least one expert per token
        - top_k <= num_experts: Cannot select more experts than available
        - expert_capacity_factor > 0: Must have positive capacity factor
        - moe_intermediate_size > 0: Must have valid intermediate size
        
        Args:
            config: ModelConfig instance to validate
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        
        Raises:
            ValueError: If MoE configuration is invalid
        """
        errors = []
        
        # Only validate MoE models
        if config.model_type != ModelType.MOE:
            return errors
            
        # Validate num_experts
        if config.num_experts <= 1:
            errors.append("MoE models must have num_experts > 1")
        elif config.num_experts > 1000:
            errors.append(f"num_experts ({config.num_experts}) seems unreasonably high")
            
        # Validate top_k
        if config.top_k < 1:
            errors.append("top_k must be >= 1")
        elif config.top_k > config.num_experts:
            errors.append(f"top_k ({config.top_k}) cannot be greater than num_experts ({config.num_experts})")
        elif config.top_k > 8:
            errors.append(f"top_k ({config.top_k}) seems unreasonably high for most MoE models")
            
        # Validate expert_capacity_factor
        if config.expert_capacity_factor <= 0:
            errors.append("expert_capacity_factor must be > 0")
        elif config.expert_capacity_factor > 10:
            errors.append(f"expert_capacity_factor ({config.expert_capacity_factor}) seems unreasonably high")
            
        # Validate moe_intermediate_size
        if config.moe_intermediate_size is not None:
            if config.moe_intermediate_size <= 0:
                errors.append("moe_intermediate_size must be > 0")
            elif config.moe_intermediate_size > config.intermediate_size * 10:
                errors.append(f"moe_intermediate_size ({config.moe_intermediate_size}) seems unreasonably large compared to intermediate_size ({config.intermediate_size})")
                
        return errors
    
    @staticmethod
    def validate_attention_config(config: 'ModelConfig') -> List[str]:
        """Validate attention configuration"""
        errors = []
        if config.num_key_value_heads > config.num_attention_heads:
            errors.append("num_key_value_heads cannot exceed num_attention_heads")
        return errors
    
    @staticmethod
    def validate_sequence_lengths(config: 'ModelConfig', model_max_seq_len: int) -> List[str]:
        """Validate sequence length parameters"""
        errors = []
        if config.sequence_length > model_max_seq_len:
            errors.append(f"sequence_length ({config.sequence_length}) cannot exceed model's max_position_embeddings ({model_max_seq_len})")
        if config.hp_seq_len > model_max_seq_len:
            errors.append(f"hp_seq_len ({config.hp_seq_len}) cannot exceed model's max_position_embeddings ({model_max_seq_len})")
        return errors
    
    @staticmethod
    def validate_all(config: 'ModelConfig', model_max_seq_len: int = None) -> List[str]:
        """Validate all configuration rules"""
        errors = []
        errors.extend(ValidationRules.validate_moe_config(config))
        errors.extend(ValidationRules.validate_attention_config(config))
        if model_max_seq_len is not None:
            errors.extend(ValidationRules.validate_sequence_lengths(config, model_max_seq_len))
        return errors


    


class MoECalculator:
    """Calculator for MoE-specific memory and FLOPS calculations with advanced optimizations"""
    
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self.advanced_optimizations = False
        self.routing_metrics = {}
    
    def enable_advanced_optimizations(self):
        """
        ⚠️  EXPERIMENTAL: Enable advanced MoE routing and optimization features
        
        This requires the optional moe_advanced.py module and provides
        demonstrative routing strategies. Not recommended for production use.
        """
        self.advanced_optimizations = True
        try:
            from moe_advanced import AdvancedMoECalculator
            self.advanced_calc = AdvancedMoECalculator(self.config)
            logger.warning("⚠️  EXPERIMENTAL: Advanced MoE optimizations enabled (demonstrative only)")
        except ImportError:
            logger.warning("Advanced MoE optimizations not available. Install moe_advanced.py module.")
            logger.info("Note: Core MoE calculations will use simplified methods.")
            self.advanced_optimizations = False
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing metrics if advanced optimizations are enabled"""
        if self.advanced_optimizations and hasattr(self, 'advanced_calc'):
            return self.advanced_calc.get_routing_metrics()
        return {}
    
    def calculate_hbm_memory(self, context: MemoryContext) -> float:
        """
        Calculate HBM memory for MoE models (entire model stored in HBM)
        
        Uses simplified calculations by default. Advanced optimizations
        can be enabled with enable_advanced_optimizations() for
        demonstrative routing strategies.
        """
        if self.advanced_optimizations and hasattr(self, 'advanced_calc'):
            logger.warning("Using experimental advanced MoE calculations (demonstrative)")
            # Use advanced MoE calculations
            calc_context = CalculationContext(
                seq_len=context.seq_len,
                batch_size=context.batch_size,
                hidden_size=context.hidden_size,
                num_heads=context.num_heads,
                num_kv_heads=context.num_kv_heads,
                intermediate_size=context.intermediate_size,
                dtype_bytes=context.dtype_bytes,
                num_layers=self.config.num_layers,
                vocab_size=self.config.vocab_size,
                model_type=self.config.model_type,
                num_experts=self.config.num_experts,
                top_k=self.config.top_k,
                expert_capacity_factor=self.config.expert_capacity_factor
            )
            
            memory_components = self.advanced_calc.calculate_advanced_moe_memory(calc_context)
            return sum(memory_components.values())
        else:
            # Standard MoE calculation
            num_experts = self.config.num_experts
            top_k = self.config.top_k
            
            # For decode mode, we only process 1 token at a time, not the full sequence
            # This is the key fix: use seq_len=1 for decode mode calculations
            effective_seq_len = 1  # Decode processes 1 token at a time
            expert_capacity = int(effective_seq_len * context.batch_size * self.config.expert_capacity_factor)
            
            # Expert selection and routing (small overhead) - only for 1 token
            routing_memory = effective_seq_len * context.batch_size * num_experts * context.dtype_bytes
            
            # HBM storage: ALL experts stored in HBM (not just activated ones)
            # Each expert has: hidden_size -> intermediate_size -> hidden_size
            expert_weights_memory = num_experts * (context.hidden_size * context.intermediate_size + 
                                                 context.intermediate_size * context.hidden_size) * context.dtype_bytes
            
            # Expert computation memory (only activated experts moved to GPU) - only for 1 token
            activated_expert_memory = top_k * expert_capacity * context.intermediate_size * context.dtype_bytes
            
            # Output combination - only for 1 token
            output_memory = effective_seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
            
            return routing_memory + expert_weights_memory + activated_expert_memory + output_memory
    
    def calculate_gpu_memory(self, context: MemoryContext) -> float:
        """Calculate GPU memory for MoE models (only activated experts)"""
        num_experts = self.config.num_experts
        top_k = self.config.top_k
        
        # For decode mode, we only process 1 token at a time, not the full sequence
        effective_seq_len = 1  # Decode processes 1 token at a time
        expert_capacity = int(effective_seq_len * context.batch_size * self.config.expert_capacity_factor)
        
        # Expert selection and routing (small overhead) - only for 1 token
        routing_memory = effective_seq_len * context.batch_size * num_experts * context.dtype_bytes
        
        # GPU memory: Only activated experts moved to GPU
        activated_expert_weights = top_k * (context.hidden_size * context.intermediate_size + 
                                          context.intermediate_size * context.hidden_size) * context.dtype_bytes
        
        # Expert computation memory (activated experts) - only for 1 token
        activated_expert_memory = top_k * expert_capacity * context.intermediate_size * context.dtype_bytes
        
        # Output combination - only for 1 token
        output_memory = effective_seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
        
        return routing_memory + activated_expert_weights + activated_expert_memory + output_memory
    
    def get_gpu_memory_components(self, memory_components: 'MemoryComponents', context: MemoryContext) -> 'MemoryComponents':
        """Get GPU memory components for MoE models (only activated experts)"""
        # For MoE models, use GPU memory (only activated experts) for arithmetic intensity
        mlp_gpu_memory = self.calculate_gpu_memory(context)
        
        # For MoE models, attention is SHARED across all experts
        # Attention mechanism is the same regardless of which experts are activated
        # Therefore, attention memory is NOT reduced by expert activation ratio
        attention_gpu_memory = memory_components.attention
        
        return MemoryComponents(
            attention=attention_gpu_memory,  # Use GPU memory for attention
            projections=memory_components.projections,
            mlp=mlp_gpu_memory,  # Use GPU memory for MLP
            embeddings=memory_components.embeddings,
            layer_norm=memory_components.layer_norm,
            activations=memory_components.activations,
            weights=memory_components.weights,  # Keep full weights for HBM storage
            kv_cache=memory_components.kv_cache,
            total=memory_components.total
        )
    
    def calculate_flops(self, context: MemoryContext) -> float:
        """Calculate MoE MLP FLOPS"""
        num_experts = self.config.num_experts
        top_k = self.config.top_k
        
        # For decode mode, we only process 1 token at a time, not the full sequence
        effective_seq_len = 1  # Decode processes 1 token at a time
        expert_capacity = int(effective_seq_len * context.batch_size * self.config.expert_capacity_factor)
        
        # Expert selection and routing FLOPS - only for 1 token
        routing_flops = effective_seq_len * context.batch_size * num_experts
        
        # Expert computation FLOPS (only activated experts) - only for 1 token
        # Each expert: hidden_size -> intermediate_size -> hidden_size
        expert_flops = top_k * expert_capacity * (context.hidden_size * context.intermediate_size + 
                                                context.intermediate_size * context.hidden_size)
        
        # Output combination FLOPS - only for 1 token
        output_flops = effective_seq_len * context.batch_size * context.hidden_size
        
        return routing_flops + expert_flops + output_flops


class ConfigBuilder:
    """Builder class for creating ModelConfig instances with overrides"""
    
    @staticmethod
    def from_base_config(base_config: 'ModelConfig', **overrides) -> 'ModelConfig':
        """Create new config with overrides from base config"""
        return ModelConfig(
            num_layers=overrides.get('num_layers', base_config.num_layers),
            hidden_size=overrides.get('hidden_size', base_config.hidden_size),
            num_attention_heads=overrides.get('num_attention_heads', base_config.num_attention_heads),
            num_key_value_heads=overrides.get('num_key_value_heads', base_config.num_key_value_heads),
            intermediate_size=overrides.get('intermediate_size', base_config.intermediate_size),
            vocab_size=overrides.get('vocab_size', base_config.vocab_size),
            model_type=overrides.get('model_type', base_config.model_type),
            num_experts=overrides.get('num_experts', base_config.num_experts),
            expert_capacity_factor=overrides.get('expert_capacity_factor', base_config.expert_capacity_factor),
            top_k=overrides.get('top_k', base_config.top_k),
            moe_intermediate_size=overrides.get('moe_intermediate_size', base_config.moe_intermediate_size),
            sequence_length=overrides.get('sequence_length', base_config.sequence_length),
            hp_seq_len=overrides.get('hp_seq_len', base_config.hp_seq_len),
            batch_size=overrides.get('batch_size', base_config.batch_size),
            decode_len=overrides.get('decode_len', base_config.decode_len),
            dtype=overrides.get('dtype', base_config.dtype),
            use_flash_attention=overrides.get('use_flash_attention', base_config.use_flash_attention),
            use_flash_decode=overrides.get('use_flash_decode', base_config.use_flash_decode),
            flash_attention_block_size=overrides.get('flash_attention_block_size', base_config.flash_attention_block_size),
            flash_decode_block_size=overrides.get('flash_decode_block_size', base_config.flash_decode_block_size)
        )
    
    @staticmethod
    def for_pretraining(base_config: 'ModelConfig') -> 'ModelConfig':
        """Create config for pretraining (uses hp_seq_len)"""
        return ConfigBuilder.from_base_config(
            base_config, 
            sequence_length=base_config.hp_seq_len
        )
    
    @staticmethod
    def for_decode(base_config: 'ModelConfig', total_context_length: int) -> 'ModelConfig':
        """Create config for decode mode with total context length"""
        return ConfigBuilder.from_base_config(
            base_config,
            sequence_length=total_context_length
        )


@dataclass
class ModelConfig:
    """Configuration for a transformer model"""
    # Model architecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # For GQA/MQA
    intermediate_size: Optional[int] = None
    vocab_size: int = 50257
    
    # MoE specific parameters
    model_type: ModelType = ModelType.DENSE
    num_experts: int = 1
    expert_capacity_factor: float = 1.0
    top_k: int = 1
    moe_intermediate_size: Optional[int] = None  # For MoE expert size
    num_shared_experts: int = 0  # Shared experts (used by all tokens)
    
    # Sequence parameters
    sequence_length: int = 2048  # For prefill and decode modes
    hp_seq_len: int = 8192       # Hyperparameter sequence length for pretraining
    batch_size: int = 1
    decode_len: int = 64  # Number of tokens to generate in decode phase
    
    # Precision
    dtype: DataType = DataType.BF16  # Data type for calculations
    
    # Memory optimization parameters
    use_flash_attention: bool = True  # Use Flash Attention for pretraining/prefill
    use_flash_decode: bool = True     # Use Flash Decode for inference
    flash_attention_block_size: int = 4096  # Block size for Flash Attention
    flash_decode_block_size: int = 1024     # Block size for Flash Decode
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
    
    @property
    def dtype_bytes(self) -> float:
        """Get dtype bytes from DataType.bytes (single source of truth)"""
        return self.dtype.bytes
    
    @classmethod
    def from_hf_config(cls, hf_config: dict, model_name: str = "", 
                      sequence_length: int = 2048, batch_size: int = 1, 
                      decode_len: int = 64) -> 'ModelConfig':
        """Create ModelConfig from Hugging Face config with comprehensive parsing"""
        
        # Auto-detect data type
        detected_dtype = DataTypeDetector.detect_from_config(hf_config)
        if not detected_dtype:
            detected_dtype = DataTypeDetector.detect_from_model_name(model_name)
        
        # Handle multimodal models (like Qwen3-VL) that have text_config
        config_to_use = hf_config
        if 'text_config' in hf_config:
            logger.info("Detected multimodal model with text_config")
            config_to_use = hf_config['text_config']
        
        # Extract parameters with proper error handling
        missing_params = []
        warnings = []
        
        # Required parameters with fallbacks
        hidden_size = config_to_use.get('hidden_size', config_to_use.get('d_model'))
        if not hidden_size:
            missing_params.append('hidden_size')
            hidden_size = 4096  # Common default
            warnings.append(f"Missing hidden_size, using default: {hidden_size}")
        
        num_layers = config_to_use.get('num_hidden_layers', config_to_use.get('n_layer'))
        if not num_layers:
            missing_params.append('num_layers')
            num_layers = 32  # Common default
            warnings.append(f"Missing num_layers, using default: {num_layers}")
        
        num_heads = config_to_use.get('num_attention_heads', config_to_use.get('n_head'))
        if not num_heads:
            missing_params.append('num_attention_heads')
            num_heads = 32  # Common default
            warnings.append(f"Missing num_attention_heads, using default: {num_heads}")
        
        num_kv_heads = config_to_use.get('num_key_value_heads', num_heads)
        vocab_size = config_to_use.get('vocab_size')
        if not vocab_size:
            missing_params.append('vocab_size')
            vocab_size = 50257  # Common default
            warnings.append(f"Missing vocab_size, using default: {vocab_size}")
        
        intermediate_size = config_to_use.get('intermediate_size', config_to_use.get('ffn_dim'))
        if not intermediate_size:
            intermediate_size = 4 * hidden_size  # Standard scaling
            warnings.append(f"Missing intermediate_size, using 4x hidden_size: {intermediate_size}")
        
        # Check for MoE
        is_moe = cls._is_moe_model(config_to_use)
        if is_moe:
            num_experts, top_k = cls._extract_moe_params(config_to_use)
            moe_intermediate_size = config_to_use.get('moe_intermediate_size', intermediate_size)
            num_shared_experts = config_to_use.get('n_shared_experts', 0)
            model_type = ModelType.MOE
        else:
            num_experts, top_k = 1, 1
            moe_intermediate_size = None
            num_shared_experts = 0
            model_type = ModelType.DENSE
        
        # Print warnings if any
        if warnings:
            logger.warning(f"⚠️  WARNING: Missing parameters for {model_name}")
            for warning in warnings:
                logger.warning(f"  {warning}")
            logger.info("Consider providing model parameters manually for more accurate calculations.")
        
        # Use model name heuristics for missing critical parameters
        if missing_params and model_name:
            cls._apply_model_name_heuristics(model_name, missing_params, 
                                           {'hidden_size': hidden_size, 'num_layers': num_layers, 
                                            'num_heads': num_heads, 'vocab_size': vocab_size})
        
        return cls(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            model_type=model_type,
            num_experts=num_experts,
            expert_capacity_factor=config_to_use.get('expert_capacity_factor', 1.0),
            top_k=top_k,
            moe_intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len,
            dtype=detected_dtype
        )
    
    @staticmethod
    def _is_moe_model(config: dict) -> bool:
        """Check if model is MoE based on config"""
        return (config.get('num_experts', 1) > 1 or 
                config.get('n_routed_experts', 1) > 1 or
                config.get('n_shared_experts', 0) > 0 or
                config.get('expert_capacity_factor') is not None or
                config.get('num_experts_per_tok', 1) > 1 or
                'moe' in config.get('model_type', '').lower())
    
    @staticmethod
    def _extract_moe_params(config: dict) -> tuple[int, int]:
        """Extract MoE parameters from config"""
        # Handle different MoE parameter names
        num_experts = (config.get('num_experts') or 
                      config.get('n_routed_experts') or 
                      config.get('n_experts') or 8)
        top_k = (config.get('top_k') or 
                config.get('num_experts_per_tok') or 
                config.get('n_experts_per_tok') or 2)
        return num_experts, top_k
    
    @staticmethod
    def _apply_model_name_heuristics(model_name: str, missing_params: list, current_params: dict):
        """Apply model name heuristics for missing parameters"""
        model_name = model_name.lower()
        
        if '235b' in model_name or '235' in model_name:
            if 'hidden_size' in missing_params:
                current_params['hidden_size'] = 5120
                print(f"  Inferred from model name: hidden_size=5120")
            if 'num_layers' in missing_params:
                current_params['num_layers'] = 64
                print(f"  Inferred from model name: num_layers=64")
            if 'num_heads' in missing_params:
                current_params['num_heads'] = 40
                print(f"  Inferred from model name: num_heads=40")
        elif '120b' in model_name or '120' in model_name:
            if 'hidden_size' in missing_params:
                current_params['hidden_size'] = 4096
                print(f"  Inferred from model name: hidden_size=4096")
            if 'num_layers' in missing_params:
                current_params['num_layers'] = 32
                print(f"  Inferred from model name: num_layers=32")
            if 'num_heads' in missing_params:
                current_params['num_heads'] = 32
                print(f"  Inferred from model name: num_heads=32")
        elif '30b' in model_name or '30' in model_name:
            if 'hidden_size' in missing_params:
                current_params['hidden_size'] = 4096
                print(f"  Inferred from model name: hidden_size=4096")
            if 'num_layers' in missing_params:
                current_params['num_layers'] = 32
                print(f"  Inferred from model name: num_layers=32")
            if 'num_heads' in missing_params:
                current_params['num_heads'] = 32
                print(f"  Inferred from model name: num_heads=32")


@dataclass
class MemoryComponents:
    """Memory usage breakdown by component"""
    attention: float = 0.0
    projections: float = 0.0
    mlp: float = 0.0
    embeddings: float = 0.0
    layer_norm: float = 0.0
    activations: float = 0.0
    weights: float = 0.0
    kv_cache: float = 0.0
    gradients: float = 0.0
    optimizer_states: float = 0.0
    total: float = 0.0


@dataclass
class FLOPSComponents:
    """FLOPS breakdown by component"""
    attention: float = 0.0
    projections: float = 0.0
    mlp: float = 0.0
    embeddings: float = 0.0
    layer_norm: float = 0.0
    total: float = 0.0


@dataclass
class ReuseComponents:
    """Arithmetic intensity (reuse) breakdown by stage"""
    pre_attention: float = 0.0  # FLOPS/byte for pre-attention stage
    attention: float = 0.0      # FLOPS/byte for attention stage
    post_attention: float = 0.0 # FLOPS/byte for post-attention stage
    total: float = 0.0


class MemoryComponentsBuilder:
    """Builder pattern for MemoryComponents"""
    
    def __init__(self):
        self.components = {
            'attention': 0.0,
            'projections': 0.0,
            'mlp': 0.0,
            'embeddings': 0.0,
            'layer_norm': 0.0,
            'activations': 0.0,
            'weights': 0.0,
            'kv_cache': 0.0,
            'gradients': 0.0,
            'optimizer_states': 0.0
        }
    
    def add_attention(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add attention memory"""
        self.components['attention'] += memory
        return self
    
    def add_projections(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add projection memory"""
        self.components['projections'] += memory
        return self
    
    def add_mlp(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add MLP memory"""
        self.components['mlp'] += memory
        return self
    
    def add_embeddings(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add embedding memory"""
        self.components['embeddings'] += memory
        return self
    
    def add_layer_norm(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add layer norm memory"""
        self.components['layer_norm'] += memory
        return self
    
    def add_activations(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add activations memory"""
        self.components['activations'] += memory
        return self
    
    def add_weights(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add weights memory"""
        self.components['weights'] += memory
        return self
    
    def add_kv_cache(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add KV cache memory"""
        self.components['kv_cache'] += memory
        return self
    
    def add_gradients(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add gradients memory"""
        self.components['gradients'] += memory
        return self
    
    def add_optimizer_states(self, memory: float) -> 'MemoryComponentsBuilder':
        """Add optimizer states memory"""
        self.components['optimizer_states'] += memory
        return self
    
    def build(self) -> MemoryComponents:
        """Build MemoryComponents with calculated total"""
        total = sum(self.components.values())
        return MemoryComponents(
            attention=self.components['attention'],
            projections=self.components['projections'],
            mlp=self.components['mlp'],
            embeddings=self.components['embeddings'],
            layer_norm=self.components['layer_norm'],
            activations=self.components['activations'],
            weights=self.components['weights'],
            kv_cache=self.components['kv_cache'],
            gradients=self.components['gradients'],
            optimizer_states=self.components['optimizer_states'],
            total=total
        )


class FLOPSComponentsBuilder:
    """Builder pattern for FLOPSComponents"""
    
    def __init__(self):
        self.components = {
            'attention': 0.0,
            'projections': 0.0,
            'mlp': 0.0,
            'embeddings': 0.0,
            'layer_norm': 0.0
        }
    
    def add_attention(self, flops: float) -> 'FLOPSComponentsBuilder':
        """Add attention FLOPS"""
        self.components['attention'] += flops
        return self
    
    def add_projections(self, flops: float) -> 'FLOPSComponentsBuilder':
        """Add projection FLOPS"""
        self.components['projections'] += flops
        return self
    
    def add_mlp(self, flops: float) -> 'FLOPSComponentsBuilder':
        """Add MLP FLOPS"""
        self.components['mlp'] += flops
        return self
    
    def add_embeddings(self, flops: float) -> 'FLOPSComponentsBuilder':
        """Add embedding FLOPS"""
        self.components['embeddings'] += flops
        return self
    
    def add_layer_norm(self, flops: float) -> 'FLOPSComponentsBuilder':
        """Add layer norm FLOPS"""
        self.components['layer_norm'] += flops
        return self
    
    def build(self) -> FLOPSComponents:
        """Build FLOPSComponents with calculated total"""
        total = sum(self.components.values())
        return FLOPSComponents(
            attention=self.components['attention'],
            projections=self.components['projections'],
            mlp=self.components['mlp'],
            embeddings=self.components['embeddings'],
            layer_norm=self.components['layer_norm'],
            total=total
        )


class CalculationStrategy:
    """Base strategy for mode-specific calculations"""
    
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self.component_calc = ComponentCalculator(config)
    
    def calculate_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate memory for this mode"""
        raise NotImplementedError
    
    def calculate_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate FLOPS for this mode"""
        raise NotImplementedError
    
    def calculate_reuse(self, context: CalculationContext) -> ReuseComponents:
        """Calculate reuse for this mode"""
        raise NotImplementedError


class PretrainingStrategy(CalculationStrategy):
    """Strategy for pretraining calculations"""
    
    def calculate_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate pretraining memory (forward + backward pass)"""
        # Forward pass memory
        forward_memory = self._calculate_forward_memory(context)
        
        # Backward pass memory (typically 1.5x forward pass)
        backward_memory = MemoryComponentsBuilder() \
            .add_attention(forward_memory.attention * 1.5) \
            .add_projections(forward_memory.projections * 1.5) \
            .add_mlp(forward_memory.mlp * 1.5) \
            .add_embeddings(forward_memory.embeddings * 1.5) \
            .add_layer_norm(forward_memory.layer_norm * 1.5) \
            .add_activations(forward_memory.activations * 1.5) \
            .add_gradients(self._calculate_gradients_memory(context)) \
            .add_optimizer_states(self._calculate_optimizer_memory(context)) \
            .build()
        
        # Total memory
        return MemoryComponentsBuilder() \
            .add_attention(forward_memory.attention + backward_memory.attention) \
            .add_projections(forward_memory.projections + backward_memory.projections) \
            .add_mlp(forward_memory.mlp + backward_memory.mlp) \
            .add_embeddings(forward_memory.embeddings + backward_memory.embeddings) \
            .add_layer_norm(forward_memory.layer_norm + backward_memory.layer_norm) \
            .add_activations(forward_memory.activations + backward_memory.activations) \
            .add_weights(forward_memory.weights) \
            .add_kv_cache(forward_memory.kv_cache) \
            .add_gradients(backward_memory.gradients) \
            .add_optimizer_states(backward_memory.optimizer_states) \
            .build()
    
    def calculate_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate pretraining FLOPS (forward + backward pass)"""
        forward_flops = self._calculate_forward_flops(context)
        
        # Backward pass typically requires similar FLOPS as forward pass
        backward_flops = FLOPSComponentsBuilder() \
            .add_attention(forward_flops.attention) \
            .add_projections(forward_flops.projections) \
            .add_mlp(forward_flops.mlp) \
            .add_embeddings(forward_flops.embeddings) \
            .add_layer_norm(forward_flops.layer_norm) \
            .build()
        
        # Total FLOPS
        return FLOPSComponentsBuilder() \
            .add_attention(forward_flops.attention + backward_flops.attention) \
            .add_projections(forward_flops.projections + backward_flops.projections) \
            .add_mlp(forward_flops.mlp + backward_flops.mlp) \
            .add_embeddings(forward_flops.embeddings + backward_flops.embeddings) \
            .add_layer_norm(forward_flops.layer_norm + backward_flops.layer_norm) \
            .build()
    
    def calculate_reuse(self, context: CalculationContext) -> ReuseComponents:
        """Calculate pretraining reuse"""
        memory_components = self._calculate_forward_memory(context)
        flops_components = self._calculate_forward_flops(context)
        
        # Calculate reuse for each stage
        pre_attention_reuse = self._calculate_pre_attention_reuse(memory_components, flops_components)
        attention_reuse = self._calculate_attention_reuse(memory_components, flops_components)
        post_attention_reuse = self._calculate_post_attention_reuse(memory_components, flops_components)
        
        total_reuse = pre_attention_reuse + attention_reuse + post_attention_reuse
        
        return ReuseComponents(
            pre_attention=pre_attention_reuse,
            attention=attention_reuse,
            post_attention=post_attention_reuse,
            total=total_reuse
        )
    
    def _calculate_forward_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate forward pass memory"""
        # Calculate component memory
        attention_memory, _ = self.component_calc.calculate_attention_components(context)
        projection_memory, _ = self.component_calc.calculate_projection_components(context)
        mlp_memory, _ = self.component_calc.calculate_mlp_components(context)
        embedding_memory, _ = self.component_calc.calculate_embedding_components(context)
        layer_norm_memory, _ = self.component_calc.calculate_layer_norm_components(context)
        
        # Scale by number of layers
        attention_memory *= context.num_layers
        projection_memory *= context.num_layers
        # For MoE models, expert weights are shared across layers, so don't multiply MLP memory
        if context.model_type != ModelType.MOE:
            mlp_memory *= context.num_layers
        layer_norm_memory *= context.num_layers
        
        # Activations memory
        activations_memory = self._calculate_activations_memory(context)
        
        # Weights memory
        weights_memory = self._calculate_weights_memory(context)
        
        return MemoryComponentsBuilder() \
            .add_attention(attention_memory) \
            .add_projections(projection_memory) \
            .add_mlp(mlp_memory) \
            .add_embeddings(embedding_memory) \
            .add_layer_norm(layer_norm_memory) \
            .add_activations(activations_memory) \
            .add_weights(weights_memory) \
            .build()
    
    def _calculate_forward_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate forward pass FLOPS"""
        # Calculate component FLOPS
        _, attention_flops = self.component_calc.calculate_attention_components(context)
        _, projection_flops = self.component_calc.calculate_projection_components(context)
        _, mlp_flops = self.component_calc.calculate_mlp_components(context)
        _, embedding_flops = self.component_calc.calculate_embedding_components(context)
        _, layer_norm_flops = self.component_calc.calculate_layer_norm_components(context)
        
        # Scale by number of layers
        attention_flops *= context.num_layers
        projection_flops *= context.num_layers
        mlp_flops *= context.num_layers
        layer_norm_flops *= context.num_layers
        
        return FLOPSComponentsBuilder() \
            .add_attention(attention_flops) \
            .add_projections(projection_flops) \
            .add_mlp(mlp_flops) \
            .add_embeddings(embedding_flops) \
            .add_layer_norm(layer_norm_flops) \
            .build()
    
    def _calculate_gradients_memory(self, context: CalculationContext) -> float:
        """Calculate gradients memory"""
        return self._calculate_weights_memory(context)
    
    def _calculate_optimizer_memory(self, context: CalculationContext) -> float:
        """Calculate optimizer states memory"""
        return self._calculate_weights_memory(context) * 2  # Adam optimizer uses 2x weights memory
    
    def _calculate_weights_memory(self, context: CalculationContext) -> float:
        """Calculate model weights memory"""
        # Use the correct total parameters calculation
        try:
            from .transformer_calculator import calculate_model_parameters
        except ImportError:
            # Fallback for when run as script
            from transformer_calculator import calculate_model_parameters
        total_params = calculate_model_parameters(self.config)
        return total_params * context.dtype_bytes
    
    def _calculate_activations_memory(self, context: CalculationContext) -> float:
        """Calculate activations memory"""
        return context.seq_len * context.batch_size * context.hidden_size * context.dtype_bytes
    
    def _calculate_pre_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate pre-attention reuse"""
        memory_bytes = memory.embeddings + memory.projections
        if memory_bytes > 0:
            return flops.embeddings / memory_bytes
        return 0.0
    
    def _calculate_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate attention reuse"""
        # For MoE models, use GPU memory components (bytes moved) for arithmetic intensity
        if self.config.model_type == ModelType.MOE:
            # Get the MoE calculator to access GPU memory components
            from transformer_calculator import MoECalculator
            moe_calc = MoECalculator(self.config)
            gpu_memory_components = moe_calc.get_gpu_memory_components(memory, self._create_memory_context())
            # For MoE, use GPU memory (bytes moved) for arithmetic intensity
            memory_bytes = gpu_memory_components.attention
        else:
            # For dense models, use standard memory calculation
            memory_bytes = memory.attention
        
        if memory_bytes > 0:
            return flops.attention / memory_bytes
        return 0.0
    
    def _calculate_post_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate post-attention reuse"""
        memory_bytes = memory.mlp + memory.layer_norm
        flops_total = flops.mlp + flops.layer_norm
        if memory_bytes > 0:
            return flops_total / memory_bytes
        return 0.0
    
    def _create_memory_context(self):
        """Create memory context for MoE calculations"""
        from context import MemoryContext
        return MemoryContext(seq_len=self.config.sequence_length,
            batch_size=self.config.batch_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=self.config.dtype_bytes,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            model_type=self.config.model_type,
            mode="prefill",  # Default mode for memory context
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            expert_capacity_factor=self.config.expert_capacity_factor
        )


class PrefillStrategy(CalculationStrategy):
    """Strategy for prefill calculations"""
    
    def calculate_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate prefill memory"""
        return self._calculate_forward_memory(context)
    
    def calculate_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate prefill FLOPS"""
        return self._calculate_forward_flops(context)
    
    def calculate_reuse(self, context: CalculationContext) -> ReuseComponents:
        """Calculate prefill reuse"""
        memory_components = self._calculate_forward_memory(context)
        flops_components = self._calculate_forward_flops(context)
        
        pre_attention_reuse = self._calculate_pre_attention_reuse(memory_components, flops_components)
        attention_reuse = self._calculate_attention_reuse(memory_components, flops_components)
        post_attention_reuse = self._calculate_post_attention_reuse(memory_components, flops_components)
        
        total_reuse = pre_attention_reuse + attention_reuse + post_attention_reuse
        
        return ReuseComponents(
            pre_attention=pre_attention_reuse,
            attention=attention_reuse,
            post_attention=post_attention_reuse,
            total=total_reuse
        )
    
    def _create_memory_context(self):
        """Create memory context for MoE calculations"""
        from context import MemoryContext
        return MemoryContext(seq_len=self.config.sequence_length,
            batch_size=self.config.batch_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=self.config.dtype_bytes,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            model_type=self.config.model_type,
            mode="prefill",  # Default mode for memory context
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            expert_capacity_factor=self.config.expert_capacity_factor
        )
    
    def _calculate_forward_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate forward pass memory"""
        # Similar to pretraining but without gradients/optimizer
        attention_memory, _ = self.component_calc.calculate_attention_components(context)
        projection_memory, _ = self.component_calc.calculate_projection_components(context)
        mlp_memory, _ = self.component_calc.calculate_mlp_components(context)
        embedding_memory, _ = self.component_calc.calculate_embedding_components(context)
        layer_norm_memory, _ = self.component_calc.calculate_layer_norm_components(context)
        
        # Scale by number of layers
        attention_memory *= context.num_layers
        projection_memory *= context.num_layers
        # For MoE models, expert weights are shared across layers, so don't multiply MLP memory
        if context.model_type != ModelType.MOE:
            mlp_memory *= context.num_layers
        layer_norm_memory *= context.num_layers
        
        # Weights memory
        weights_memory = self._calculate_weights_memory(context)
        
        # KV cache memory for prefill (stores key-value pairs for all tokens)
        kv_cache_memory = self._calculate_kv_cache_memory(context)
        
        return MemoryComponentsBuilder() \
            .add_attention(attention_memory) \
            .add_projections(projection_memory) \
            .add_mlp(mlp_memory) \
            .add_embeddings(embedding_memory) \
            .add_layer_norm(layer_norm_memory) \
            .add_weights(weights_memory) \
            .add_kv_cache(kv_cache_memory) \
            .build()
    
    def _calculate_kv_cache_memory(self, context: CalculationContext) -> float:
        """Calculate KV cache memory for prefill mode"""
        # Math_Used.md: B × S × H × (d_k + d_v) × bytes
        # KV cache stores key and value tensors per layer: K and V
        head_dim = context.hidden_size // context.num_heads
        # B × S × H × (d_k + d_v) × bytes per layer
        kv_cache_per_layer = context.batch_size * context.seq_len * context.num_kv_heads * (head_dim + head_dim) * context.dtype_bytes
        total = context.num_layers * kv_cache_per_layer
        return total
    
    def _calculate_forward_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate forward pass FLOPS"""
        _, attention_flops = self.component_calc.calculate_attention_components(context)
        _, projection_flops = self.component_calc.calculate_projection_components(context)
        _, mlp_flops = self.component_calc.calculate_mlp_components(context)
        _, embedding_flops = self.component_calc.calculate_embedding_components(context)
        _, layer_norm_flops = self.component_calc.calculate_layer_norm_components(context)
        
        # Scale by number of layers
        attention_flops *= context.num_layers
        projection_flops *= context.num_layers
        mlp_flops *= context.num_layers
        layer_norm_flops *= context.num_layers
        
        return FLOPSComponentsBuilder() \
            .add_attention(attention_flops) \
            .add_projections(projection_flops) \
            .add_mlp(mlp_flops) \
            .add_embeddings(embedding_flops) \
            .add_layer_norm(layer_norm_flops) \
            .build()
    
    def _calculate_weights_memory(self, context: CalculationContext) -> float:
        """Calculate model weights memory"""
        # Use the correct total parameters calculation
        try:
            from .transformer_calculator import calculate_model_parameters
        except ImportError:
            # Fallback for when run as script
            from transformer_calculator import calculate_model_parameters
        total_params = calculate_model_parameters(self.config)
        return total_params * context.dtype_bytes
    
    def _calculate_pre_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate pre-attention reuse"""
        memory_bytes = memory.embeddings + memory.projections
        if memory_bytes > 0:
            return flops.embeddings / memory_bytes
        return 0.0
    
    def _calculate_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate attention reuse"""
        # For MoE models, use GPU memory components (bytes moved) for arithmetic intensity
        if self.config.model_type == ModelType.MOE:
            # Get the MoE calculator to access GPU memory components
            from transformer_calculator import MoECalculator
            moe_calc = MoECalculator(self.config)
            gpu_memory_components = moe_calc.get_gpu_memory_components(memory, self._create_memory_context())
            # For MoE, use GPU memory (bytes moved) for arithmetic intensity
            memory_bytes = gpu_memory_components.attention
        else:
            # For dense models, use standard memory calculation
            memory_bytes = memory.attention
        
        if memory_bytes > 0:
            return flops.attention / memory_bytes
        return 0.0
    
    def _calculate_post_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate post-attention reuse"""
        memory_bytes = memory.mlp + memory.layer_norm
        flops_total = flops.mlp + flops.layer_norm
        if memory_bytes > 0:
            return flops_total / memory_bytes
        return 0.0


class DecodeStrategy(CalculationStrategy):
    """Strategy for decode calculations"""
    
    def calculate_memory(self, context: CalculationContext) -> MemoryComponents:
        """Calculate decode memory with Flash Decode"""
        # For decode, we need to account for the peak memory usage during auto-regressive generation
        # Use Flash Decode for memory efficiency
        attention_memory = self._calculate_flash_decode_attention_memory(context)
        projection_memory, _ = self.component_calc.calculate_projection_components(context)
        mlp_memory, _ = self.component_calc.calculate_mlp_components(context)
        embedding_memory, _ = self.component_calc.calculate_embedding_components(context)
        layer_norm_memory, _ = self.component_calc.calculate_layer_norm_components(context)
        
        # Scale by number of layers
        attention_memory *= context.num_layers
        projection_memory *= context.num_layers
        # For MoE models, expert weights are shared across layers, so don't multiply MLP memory
        if context.model_type != ModelType.MOE:
            mlp_memory *= context.num_layers
        layer_norm_memory *= context.num_layers
        
        # Weights memory
        weights_memory = self._calculate_weights_memory(context)
        
        # KV cache memory for decode
        kv_cache_memory = self._calculate_kv_cache_memory(context)
        
        return MemoryComponentsBuilder() \
            .add_attention(attention_memory) \
            .add_projections(projection_memory) \
            .add_mlp(mlp_memory) \
            .add_embeddings(embedding_memory) \
            .add_layer_norm(layer_norm_memory) \
            .add_weights(weights_memory) \
            .add_kv_cache(kv_cache_memory) \
            .build()
    
    def calculate_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate decode FLOPS"""
        # Report FLOPS for the last token only (effective context length)
        effective_len = self.config.sequence_length + max(self.config.decode_len - 1, 0)
        last_step_config = ConfigBuilder.for_decode(self.config, effective_len)
        tc = TransformerCalculator(last_step_config)
        step_flops = tc._calculate_decode_step_flops(last_step_config)
        return FLOPSComponentsBuilder() \
            .add_attention(step_flops.attention) \
            .add_projections(step_flops.projections) \
            .add_mlp(step_flops.mlp) \
            .add_embeddings(step_flops.embeddings) \
            .add_layer_norm(step_flops.layer_norm) \
            .build()
    
    def calculate_reuse(self, context: CalculationContext) -> ReuseComponents:
        """Calculate decode reuse"""
        memory_components = self.calculate_memory(context)
        flops_components = self.calculate_flops(context)
        
        pre_attention_reuse = self._calculate_pre_attention_reuse(memory_components, flops_components)
        attention_reuse = self._calculate_decode_attention_reuse(memory_components, flops_components)
        post_attention_reuse = self._calculate_post_attention_reuse(memory_components, flops_components)
        
        total_reuse = pre_attention_reuse + attention_reuse + post_attention_reuse
        
        return ReuseComponents(
            pre_attention=pre_attention_reuse,
            attention=attention_reuse,
            post_attention=post_attention_reuse,
            total=total_reuse
        )
    
    def _calculate_flash_decode_attention_memory(self, context: CalculationContext) -> float:
        """Calculate attention memory for Flash Decode"""
        if self.config.use_flash_decode:
            # Flash Decode: O(1) memory per token generation
            block_size = self.config.flash_decode_block_size
            flash_decode_scores = block_size * context.batch_size * context.num_heads * context.dtype_bytes
            flash_decode_output = context.batch_size * context.hidden_size * context.dtype_bytes
            return flash_decode_scores + flash_decode_output
        else:
            # Standard attention
            attention_memory, _ = self.component_calc.calculate_attention_components(context)
            return attention_memory
    
    def _calculate_decode_step_flops(self, context: CalculationContext) -> FLOPSComponents:
        """Calculate FLOPS for a single decode step"""
        _, attention_flops = self.component_calc.calculate_attention_components(context)
        _, projection_flops = self.component_calc.calculate_projection_components(context)
        _, mlp_flops = self.component_calc.calculate_mlp_components(context)
        _, embedding_flops = self.component_calc.calculate_embedding_components(context)
        _, layer_norm_flops = self.component_calc.calculate_layer_norm_components(context)
        
        return FLOPSComponentsBuilder() \
            .add_attention(attention_flops) \
            .add_projections(projection_flops) \
            .add_mlp(mlp_flops) \
            .add_embeddings(embedding_flops) \
            .add_layer_norm(layer_norm_flops) \
            .build()
    
    def _calculate_weights_memory(self, context: CalculationContext) -> float:
        """Calculate model weights memory"""
        # Use the correct total parameters calculation
        try:
            from .transformer_calculator import calculate_model_parameters
        except ImportError:
            # Fallback for when run as script
            from transformer_calculator import calculate_model_parameters
        total_params = calculate_model_parameters(self.config)
        return total_params * context.dtype_bytes
    
    def _calculate_kv_cache_memory(self, context: CalculationContext) -> float:
        """Calculate KV cache memory"""
        # Math_Used.md: B × S × H × (d_k + d_v) × bytes
        # KV cache stores key and value matrices for each layer across all past tokens
        # Use effective context length at the last token of decode
        effective_len = self.config.sequence_length + max(self.config.decode_len - 1, 0)
        head_dim = context.hidden_size // context.num_heads
        num_kv_heads = context.num_kv_heads
        # B × S × H × (d_k + d_v) × bytes per layer
        kv_cache_per_layer = context.batch_size * effective_len * num_kv_heads * (head_dim + head_dim) * context.dtype_bytes
        total_kv_cache = context.num_layers * kv_cache_per_layer
        return total_kv_cache
    
    def _calculate_decode_attention_reuse(self, memory_components: MemoryComponents, flops_components: FLOPSComponents) -> float:
        """Calculate arithmetic intensity for attention stage in decode mode"""
        # For decode mode, use KV cache memory instead of full attention memory
        # and matrix-vector FLOPS instead of matrix-matrix FLOPS
        
        # Decode attention memory: KV cache + small attention computation
        kv_cache_memory = memory_components.kv_cache
        
        # Small attention computation memory (matrix-vector operations)
        # For decode mode, we process 1 token at a time, so seq_len = 1
        seq_len = 1  # Decode processes 1 token at a time
        batch_size = self.config.batch_size
        num_heads = self.config.num_attention_heads
        dtype_bytes = self.config.dtype_bytes
        
        # Matrix-vector attention computation memory (only for 1 token)
        small_attention_memory = batch_size * num_heads * seq_len * dtype_bytes
        
        # Total decode attention memory
        decode_attention_memory = kv_cache_memory + small_attention_memory
        
        # Decode attention FLOPS: matrix-vector operations (O(seq_len))
        head_dim = self.config.hidden_size // num_heads
        decode_attention_flops = batch_size * num_heads * seq_len * head_dim
        
        if decode_attention_memory == 0:
            return 0.0
        
        return decode_attention_flops / decode_attention_memory
    
    def _calculate_pre_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate pre-attention reuse"""
        memory_bytes = memory.embeddings + memory.projections
        if memory_bytes > 0:
            return flops.embeddings / memory_bytes
        return 0.0
    
    def _calculate_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate attention reuse"""
        # For MoE models, use GPU memory components (bytes moved) for arithmetic intensity
        if self.config.model_type == ModelType.MOE:
            # Get the MoE calculator to access GPU memory components
            from transformer_calculator import MoECalculator
            moe_calc = MoECalculator(self.config)
            gpu_memory_components = moe_calc.get_gpu_memory_components(memory, self._create_memory_context())
            # For MoE, use GPU memory (bytes moved) for arithmetic intensity
            memory_bytes = gpu_memory_components.attention
        else:
            # For dense models, use standard memory calculation
            memory_bytes = memory.attention
        
        if memory_bytes > 0:
            return flops.attention / memory_bytes
        return 0.0
    
    def _calculate_post_attention_reuse(self, memory: MemoryComponents, flops: FLOPSComponents) -> float:
        """Calculate post-attention reuse"""
        memory_bytes = memory.mlp + memory.layer_norm
        flops_total = flops.mlp + flops.layer_norm
        if memory_bytes > 0:
            return flops_total / memory_bytes
        return 0.0


class CalculationEngine:
    """Orchestrates all calculations using strategy pattern"""
    
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self._validate_config()
        self._strategies = {
            OperationMode.PRETRAINING: PretrainingStrategy(config),
            OperationMode.PREFILL: PrefillStrategy(config),
            OperationMode.DECODE: DecodeStrategy(config)
        }
    
    def _validate_config(self):
        """Validate model configuration"""
        errors = ValidationRules.validate_all(self.config)
        if errors:
            raise ValueError("; ".join(errors))
    
    def _create_context(self, mode: OperationMode) -> 'CalculationContext':
        """Create calculation context from config and mode"""
        from context import CalculationContext
        
        if mode == OperationMode.PRETRAINING:
            # Use hp_seq_len for pretraining
            seq_len = self.config.hp_seq_len
        elif mode == OperationMode.DECODE:
            # For decode mode, process 1 token at a time
            seq_len = 1
        else:
            # Use sequence_length for prefill
            seq_len = self.config.sequence_length
            
        return CalculationContext(
            seq_len=seq_len,
            batch_size=self.config.batch_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=self.config.dtype_bytes,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            model_type=self.config.model_type,
            mode=mode.value,  # Add operation mode
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            expert_capacity_factor=self.config.expert_capacity_factor
        )
    
    def calculate_memory(self, mode: OperationMode) -> MemoryComponents:
        """Calculate memory for given operation mode"""
        context = self._create_context(mode)
        strategy = self._strategies[mode]
        return strategy.calculate_memory(context)
    
    def calculate_flops(self, mode: OperationMode) -> FLOPSComponents:
        """Calculate FLOPS for given operation mode"""
        context = self._create_context(mode)
        strategy = self._strategies[mode]
        return strategy.calculate_flops(context)
    
    def calculate_reuse(self, mode: OperationMode) -> ReuseComponents:
        """Calculate reuse for given operation mode"""
        context = self._create_context(mode)
        strategy = self._strategies[mode]
        return strategy.calculate_reuse(context)


class TransformerCalculator:
    """
    Main calculator class for transformer memory and FLOPS calculations
    
    This is the primary interface for calculating memory usage and FLOPS requirements
    for transformer models. It supports both Dense and MoE architectures with
    comprehensive optimizations including Flash Attention and Flash Decode.
    
    Key Features:
    - Memory calculations for HBM and GPU memory usage
    - FLOPS calculations for compute requirements
    - Arithmetic Intensity (Reuse) calculations
    - Support for Dense and MoE architectures
    - Flash Attention and Flash Decode optimizations
    
    Mathematical Foundations:
    - Memory: Calculated in bytes, includes activations, weights, gradients
    - FLOPS: Floating-point operations, includes multiply-add operations
    - Reuse: FLOPS per byte fetched from memory (arithmetic intensity)
    
    Args:
        config: ModelConfig instance containing model parameters
    
    Example:
        >>> config = ModelConfig(
        ...     num_layers=32, hidden_size=4096, num_attention_heads=32,
        ...     model_type=ModelType.DENSE, dtype=DataType.BF16
        ... )
        >>> calculator = TransformerCalculator(config)
        >>> memory = calculator.calculate_memory(OperationMode.PREFILL)
        >>> flops = calculator.calculate_flops(OperationMode.PREFILL)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the transformer calculator
        
        Args:
            config: ModelConfig instance containing model parameters
        """
        self.config = config
        self.engine = CalculationEngine(config)
    
    def calculate_memory(self, mode: OperationMode) -> MemoryComponents:
        """Calculate memory usage for given operation mode"""
        return self.engine.calculate_memory(mode)
    
    def calculate_flops(self, mode: OperationMode) -> FLOPSComponents:
        """Calculate FLOPS for given operation mode"""
        return self.engine.calculate_flops(mode)
    
    def calculate_reuse(self, mode: OperationMode) -> ReuseComponents:
        """Calculate arithmetic intensity (reuse) for given operation mode"""
        return self.engine.calculate_reuse(mode)
    
    def _calculate_pretraining_memory(self) -> MemoryComponents:
        """Calculate memory for pre-training (forward + backward pass) using hp_seq_len"""
        # For pretraining, use hp_seq_len (hyperparameter sequence length)
        pretraining_config = self._create_pretraining_config()
        pretraining_calc = TransformerCalculator(pretraining_config)
        # Forward pass memory (no KV cache for pretraining)
        forward_memory = pretraining_calc._calculate_forward_memory(include_kv_cache=False)
        
        # Backward pass memory (typically 1.5x forward pass)
        backward_memory = MemoryComponents(
            attention=forward_memory.attention * 1.5,
            projections=forward_memory.projections * 1.5,
            mlp=forward_memory.mlp * 1.5,
            embeddings=forward_memory.embeddings * 1.5,
            layer_norm=forward_memory.layer_norm * 1.5,
            activations=forward_memory.activations * 1.5,
            gradients=self._calculate_gradients_memory(),
            optimizer_states=self._calculate_optimizer_memory()
        )
        
        # Total memory
        total_memory = MemoryComponents(
            attention=forward_memory.attention + backward_memory.attention,
            projections=forward_memory.projections + backward_memory.projections,
            mlp=forward_memory.mlp + backward_memory.mlp,
            embeddings=forward_memory.embeddings + backward_memory.embeddings,
            layer_norm=forward_memory.layer_norm + backward_memory.layer_norm,
            activations=forward_memory.activations + backward_memory.activations,
            weights=forward_memory.weights,
            kv_cache=forward_memory.kv_cache,
            gradients=backward_memory.gradients,
            optimizer_states=backward_memory.optimizer_states,
            total=0.0
        )
        
        total_memory.total = sum([
            total_memory.attention, total_memory.projections, total_memory.mlp,
            total_memory.embeddings, total_memory.layer_norm, total_memory.activations,
            total_memory.weights, total_memory.kv_cache, total_memory.gradients, total_memory.optimizer_states
        ])
        
        return total_memory
    
    def _calculate_prefill_memory(self) -> MemoryComponents:
        """Calculate memory for prefill phase"""
        return self._calculate_forward_memory(include_kv_cache=False)
    
    def _calculate_decode_memory(self) -> MemoryComponents:
        """Calculate memory for decode phase (auto-regressive generation) with Flash Decode"""
        # For decode, we need to account for the peak memory usage during auto-regressive generation
        # This occurs at the last token: sequence_length + decode_len - 1
        total_context_length = self.config.sequence_length + max(self.config.decode_len - 1, 0)
        
        # Create config for the peak context length
        decode_config = ConfigBuilder.for_decode(self.config, total_context_length)
        
        decode_calc = TransformerCalculator(decode_config)
        return decode_calc._calculate_forward_memory(include_kv_cache=True)
    
    def _calculate_forward_memory(self, include_kv_cache: bool = False) -> MemoryComponents:
        """Calculate forward pass memory"""
        seq_len = self.config.sequence_length
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        intermediate_size = self.config.intermediate_size
        dtype_bytes = self.config.dtype_bytes
        
        # Attention memory - use Flash Decode for decode mode, Flash Attention for others
        if include_kv_cache and self.config.use_flash_decode:
            # Decode mode: use Flash Decode for maximum memory efficiency
            attention_memory = self._calculate_flash_decode_attention_memory(seq_len, batch_size, hidden_size, num_heads, num_kv_heads, dtype_bytes)
        else:
            # Pretraining/Prefill mode: use Flash Attention
            attention_memory = self._calculate_attention_memory(seq_len, batch_size, hidden_size, num_heads, num_kv_heads, dtype_bytes)
        
        # Projection memory
        projection_memory = self._calculate_projection_memory(seq_len, batch_size, hidden_size, num_heads, num_kv_heads, dtype_bytes)
        
        # MLP memory
        mlp_memory = self._calculate_mlp_memory(seq_len, batch_size, hidden_size, intermediate_size, dtype_bytes)
        
        # Embedding memory
        embedding_memory = self._calculate_embedding_memory(batch_size, hidden_size, dtype_bytes)
        
        # Layer norm memory (scaled by number of layers)
        layer_norm_memory = self._calculate_layer_norm_memory(seq_len, batch_size, hidden_size, dtype_bytes) * self.config.num_layers
        
        # Activations memory
        activations_memory = self._calculate_activations_memory(seq_len, batch_size, hidden_size, intermediate_size, dtype_bytes)
        
        # Weights memory
        weights_memory = self._calculate_weights_memory()
        
        # KV cache memory (only for decode mode)
        kv_cache_memory = self._calculate_kv_cache_memory() if include_kv_cache else 0.0
        
        total = attention_memory + projection_memory + mlp_memory + embedding_memory + layer_norm_memory + activations_memory + weights_memory + kv_cache_memory
        
        return MemoryComponents(
            attention=attention_memory,
            projections=projection_memory,
            mlp=mlp_memory,
            embeddings=embedding_memory,
            layer_norm=layer_norm_memory,
            activations=activations_memory,
            weights=weights_memory,
            kv_cache=kv_cache_memory,
            total=total
        )
    
    def _calculate_attention_memory(self, seq_len: int, batch_size: int, hidden_size: int, 
                                  num_heads: int, num_kv_heads: int, dtype_bytes: int) -> float:
        """
        Calculate attention mechanism memory with Flash Attention optimization
        
        This function calculates the memory required for the attention mechanism, including
        Flash Attention optimizations that reduce memory complexity from O(seq_len²) to O(seq_len).
        
        Mathematical Foundation:
        - Standard Attention: O(seq_len²) memory for attention matrix
        - Flash Attention: O(seq_len) memory using tiling and recomputation
        - Memory includes: Q, K, V projections, attention scores, output projections
        
        Formula:
        - Q, K, V Memory: 3 * seq_len * batch_size * hidden_size * dtype_bytes
        - Attention Scores: seq_len * batch_size * num_heads * seq_len * dtype_bytes (Flash Attention: O(seq_len))
        - Output Projection: seq_len * batch_size * hidden_size * dtype_bytes
        
        Args:
            seq_len: Sequence length (number of tokens)
            batch_size: Batch size (number of sequences processed in parallel)
            hidden_size: Model hidden dimension (d_model)
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA/MQA)
            dtype_bytes: Bytes per parameter (e.g., 2 for BF16, 4 for FP32)
        
        Returns:
            float: Total attention memory in bytes
        
        References:
        - Flash Attention: https://arxiv.org/abs/2205.14135
        - Memory-Efficient Attention: https://arxiv.org/abs/2112.05682
        """
        # Use MemoryCalculator for consistent memory calculations
        context = MemoryContext(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=dtype_bytes
        )
        memory_calc = MemoryCalculator(self.config)
        return memory_calc.calculate_attention_memory(context)
    
    def _calculate_flash_decode_attention_memory(self, seq_len: int, batch_size: int, hidden_size: int,
                                               num_heads: int, num_kv_heads: int, dtype_bytes: int) -> float:
        """Calculate attention memory for Flash Decode (inference mode)"""
        # Flash Decode: Even more memory efficient than Flash Attention
        # Uses matrix-vector operations and KV cache efficiently
        
        if self.config.use_flash_decode:
            # Flash Decode: O(1) memory per token generation
            # Only stores small attention computation buffers
            block_size = self.config.flash_decode_block_size
            
            # Flash Decode scores: very small buffers for matrix-vector operations
            flash_decode_scores = block_size * batch_size * num_heads * dtype_bytes
            
            # Attention output (unchanged)
            attention_output = seq_len * batch_size * hidden_size * dtype_bytes
            
            return flash_decode_scores + attention_output
        else:
            # Fall back to Flash Attention for decode mode
            return self._calculate_attention_memory(seq_len, batch_size, hidden_size, 
                                                   num_heads, num_kv_heads, dtype_bytes)
    
    def _calculate_projection_memory(self, seq_len: int, batch_size: int, hidden_size: int,
                                   num_heads: int, num_kv_heads: int, dtype_bytes: int) -> float:
        """Calculate projection layer memory"""
        # Use MemoryCalculator for consistent memory calculations
        context = MemoryContext(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=dtype_bytes
        )
        memory_calc = MemoryCalculator(self.config)
        return memory_calc.calculate_projection_memory(context)
    
    def _calculate_mlp_memory(self, seq_len: int, batch_size: int, hidden_size: int,
                            intermediate_size: int, dtype_bytes: int) -> float:
        """Calculate MLP memory"""
        # Use MemoryCalculator for consistent memory calculations
        context = MemoryContext(
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=intermediate_size,
            dtype_bytes=dtype_bytes
        )
        memory_calc = MemoryCalculator(self.config)
        return memory_calc.calculate_mlp_memory(context)
    
    
    def _get_moe_gpu_memory_components(self, memory_components: MemoryComponents) -> MemoryComponents:
        """Get GPU memory components for MoE models (only activated experts)"""
        context = MemoryContext(seq_len=self.config.sequence_length,
            batch_size=self.config.batch_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            intermediate_size=self.config.intermediate_size,
            dtype_bytes=self.config.dtype_bytes,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            model_type=self.config.model_type,
            mode="prefill",  # Default mode for memory context
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            expert_capacity_factor=self.config.expert_capacity_factor,
            moe_intermediate_size=getattr(self.config, 'moe_intermediate_size', None)
        )
        moe_calc = MoECalculator(self.config)
        return moe_calc.get_gpu_memory_components(memory_components, context)
    
    def _calculate_embedding_memory(self, batch_size: int, hidden_size: int, dtype_bytes: int) -> float:
        """Calculate embedding memory"""
        # Input embeddings
        input_embeddings = self.config.sequence_length * batch_size * hidden_size * dtype_bytes
        
        # Positional embeddings
        positional_embeddings = self.config.sequence_length * hidden_size * dtype_bytes
        
        # Note: Output embeddings are model weights, not per-sequence memory traffic
        # They are counted in weights memory, not here
        
        return input_embeddings + positional_embeddings
    
    def _calculate_layer_norm_memory(self, seq_len: int, batch_size: int, hidden_size: int, dtype_bytes: int) -> float:
        """Calculate layer normalization memory"""
        # Pre-attention and pre-MLP layer norms
        # Each layer norm has 2 parameters (scale and bias) per hidden dimension
        return 2 * seq_len * batch_size * hidden_size * dtype_bytes
    
    def _calculate_activations_memory(self, seq_len: int, batch_size: int, hidden_size: int,
                                    intermediate_size: int, dtype_bytes: int) -> float:
        """Calculate intermediate activations memory"""
        # Hidden states
        hidden_states = seq_len * batch_size * hidden_size * dtype_bytes
        
        # Intermediate activations
        intermediate_activations = seq_len * batch_size * intermediate_size * dtype_bytes
        
        return hidden_states + intermediate_activations
    
    def _calculate_weights_memory(self) -> float:
        """Calculate model weights memory"""
        # Model weights for all parameters
        total_params = self._calculate_total_parameters()
        return total_params * self.config.dtype_bytes
    
    def _calculate_kv_cache_memory(self) -> float:
        """Calculate KV cache memory for inference"""
        # KV cache stores key and value matrices for each layer
        # Each head has its own K and V matrices
        seq_len = self.config.sequence_length
        batch_size = self.config.batch_size
        num_layers = self.config.num_layers
        num_kv_heads = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        dtype_bytes = self.config.dtype_bytes
        
        # KV cache per layer: 2 (K and V) * num_kv_heads * seq_len * head_dim
        head_dim = hidden_size // self.config.num_attention_heads
        kv_cache_per_layer = 2 * num_kv_heads * seq_len * head_dim * dtype_bytes
        
        # Total KV cache across all layers and batch
        total_kv_cache = num_layers * kv_cache_per_layer * batch_size
        
        return total_kv_cache
    
    def _calculate_gradients_memory(self) -> float:
        """Calculate gradients memory"""
        # Gradients for all parameters (typically stored at higher precision, e.g., FP32)
        total_params = self._calculate_total_parameters()
        # Use FP32 (4 bytes) for gradients regardless of model precision
        gradient_dtype_bytes = 4
        return total_params * gradient_dtype_bytes
    
    def _calculate_optimizer_memory(self) -> float:
        """Calculate optimizer states memory (Adam optimizer)"""
        # Adam stores momentum and variance for each parameter (typically FP32)
        total_params = self._calculate_total_parameters()
        # Use FP32 (4 bytes) for optimizer states regardless of model precision
        optimizer_dtype_bytes = 4
        return 2 * total_params * optimizer_dtype_bytes
    
    def _calculate_total_parameters(self) -> int:
        """Calculate total number of parameters using the centralized function"""
        return calculate_model_parameters(self.config)
    
    def _calculate_pretraining_flops(self) -> FLOPSComponents:
        """Calculate FLOPS for pre-training using hp_seq_len"""
        # For pretraining, use hp_seq_len (hyperparameter sequence length)
        pretraining_config = self._create_pretraining_config()
        pretraining_calc = TransformerCalculator(pretraining_config)
        forward_flops = pretraining_calc._calculate_forward_flops()
        
        # Backward pass typically requires similar FLOPS as forward pass
        backward_flops = FLOPSComponents(
            attention=forward_flops.attention,
            projections=forward_flops.projections,
            mlp=forward_flops.mlp,
            embeddings=forward_flops.embeddings,
            layer_norm=forward_flops.layer_norm,
            total=0.0
        )
        
        # Total FLOPS
        total_flops = FLOPSComponents(
            attention=forward_flops.attention + backward_flops.attention,
            projections=forward_flops.projections + backward_flops.projections,
            mlp=forward_flops.mlp + backward_flops.mlp,
            embeddings=forward_flops.embeddings + backward_flops.embeddings,
            layer_norm=forward_flops.layer_norm + backward_flops.layer_norm,
            total=0.0
        )
        
        total_flops.total = sum([
            total_flops.attention, total_flops.projections, total_flops.mlp,
            total_flops.embeddings, total_flops.layer_norm
        ])
        
        return total_flops
    
    def _calculate_pretraining_reuse(self) -> ReuseComponents:
        """Calculate arithmetic intensity for pre-training using hp_seq_len"""
        # For pretraining, use hp_seq_len (hyperparameter sequence length)
        pretraining_config = self._create_pretraining_config()
        pretraining_calc = TransformerCalculator(pretraining_config)
        # Get memory and FLOPS components
        memory_components = pretraining_calc._calculate_forward_memory(include_kv_cache=False)
        flops_components = pretraining_calc._calculate_forward_flops()
        
        # Calculate reuse for each stage
        pre_attention_reuse = self._calculate_pre_attention_reuse(memory_components, flops_components)
        attention_reuse = self._calculate_attention_reuse(memory_components, flops_components)
        post_attention_reuse = self._calculate_post_attention_reuse(memory_components, flops_components)
        
        total_reuse = pre_attention_reuse + attention_reuse + post_attention_reuse
        
        return ReuseComponents(
            pre_attention=pre_attention_reuse,
            attention=attention_reuse,
            post_attention=post_attention_reuse,
            total=total_reuse
        )
    
    def _calculate_prefill_reuse(self) -> ReuseComponents:
        """Calculate arithmetic intensity for prefill phase"""
        return self._calculate_pretraining_reuse()  # Same as pretraining for forward pass
    
    def _calculate_decode_reuse(self) -> ReuseComponents:
        """Calculate arithmetic intensity for decode phase"""
        # For decode, use effective last-token context length
        total_context_length = self.config.sequence_length + max(self.config.decode_len - 1, 0)
        
        # Create config for the peak context length
        decode_config = ConfigBuilder.for_decode(self.config, total_context_length)
        
        decode_calc = TransformerCalculator(decode_config)
        
        # Get memory and FLOPS components for the last token
        memory_components = decode_calc._calculate_forward_memory(include_kv_cache=True)
        flops_components = decode_calc._calculate_decode_step_flops(decode_config)
        
        # Calculate reuse for each stage with decode-specific FLOPS
        pre_attention_reuse = decode_calc._calculate_pre_attention_reuse(memory_components, flops_components)
        attention_reuse = decode_calc._calculate_decode_attention_reuse(memory_components, flops_components)
        post_attention_reuse = decode_calc._calculate_post_attention_reuse(memory_components, flops_components)
        
        total_reuse = pre_attention_reuse + attention_reuse + post_attention_reuse
        
        return ReuseComponents(
            pre_attention=pre_attention_reuse,
            attention=attention_reuse,
            post_attention=post_attention_reuse,
            total=total_reuse
        )
    
    def _calculate_pre_attention_reuse(self, memory_components: MemoryComponents, flops_components: FLOPSComponents) -> float:
        """Calculate arithmetic intensity for pre-attention stage"""
        # For MoE models, use GPU memory components (only activated experts)
        if self.config.model_type == ModelType.MOE:
            gpu_memory_components = self._get_moe_gpu_memory_components(memory_components)
            pre_attention_memory = gpu_memory_components.embeddings + gpu_memory_components.layer_norm + gpu_memory_components.projections
        else:
            # For dense models, use standard memory calculation
            pre_attention_memory = memory_components.embeddings + memory_components.layer_norm + memory_components.projections
        
        pre_attention_flops = flops_components.embeddings + flops_components.layer_norm + flops_components.projections
        
        if pre_attention_memory == 0:
            return 0.0
        
        return pre_attention_flops / pre_attention_memory
    
    def _calculate_attention_reuse(self, memory_components: MemoryComponents, flops_components: FLOPSComponents) -> float:
        """Calculate arithmetic intensity for attention stage"""
        # For MoE models, use GPU memory components (bytes moved)
        if self.config.model_type == ModelType.MOE:
            gpu_memory_components = self._get_moe_gpu_memory_components(memory_components)
            # For MoE, use GPU memory (bytes moved) for arithmetic intensity
            attention_memory = gpu_memory_components.attention
        else:
            # For dense models, use standard memory calculation
            attention_memory = memory_components.attention + memory_components.activations
        
        attention_flops = flops_components.attention
        
        if attention_memory == 0:
            return 0.0
        
        return attention_flops / attention_memory
    
    def _calculate_post_attention_reuse(self, memory_components: MemoryComponents, flops_components: FLOPSComponents) -> float:
        """Calculate arithmetic intensity for post-attention stage"""
        # Post-attention includes: MLP, final layer norm
        if self.config.model_type == ModelType.MOE:
            # For MoE models, use GPU memory components (only activated experts)
            gpu_memory_components = self._get_moe_gpu_memory_components(memory_components)
            post_attention_memory = gpu_memory_components.mlp + gpu_memory_components.layer_norm
        else:
            # For dense models, use standard memory calculation
            post_attention_memory = memory_components.mlp + memory_components.layer_norm
        
        post_attention_flops = flops_components.mlp + flops_components.layer_norm
        
        if post_attention_memory == 0:
            return 0.0
        
        return post_attention_flops / post_attention_memory
    
    def _calculate_prefill_flops(self) -> FLOPSComponents:
        """Calculate FLOPS for prefill phase"""
        return self._calculate_forward_flops()
    
    def _calculate_decode_flops(self) -> FLOPSComponents:
        """Calculate FLOPS for decode phase (auto-regressive generation)"""
        # For decode, we need to calculate FLOPS for each token generation step
        # Each step processes the full context length (sequence_length + current_position)
        # We sum up FLOPS for all decode_len steps
        
        total_flops = FLOPSComponents()
        
        # Calculate FLOPS for each token generation step
        for step in range(1, self.config.decode_len + 1):
            current_context_length = self.config.sequence_length + step
            
            # Create config for this step
            step_config = ModelConfig(
                num_layers=self.config.num_layers,
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                intermediate_size=self.config.intermediate_size,
                vocab_size=self.config.vocab_size,
                model_type=self.config.model_type,
                num_experts=self.config.num_experts,
                expert_capacity_factor=self.config.expert_capacity_factor,
                top_k=self.config.top_k,
                sequence_length=current_context_length,
                batch_size=self.config.batch_size,
                decode_len=self.config.decode_len,
                dtype=self.config.dtype
            )
            
            step_calc = TransformerCalculator(step_config)
            
            # For decode mode, use matrix-vector operations instead of matrix-matrix
            # This matches the decode reuse calculation for consistency
            step_flops = self._calculate_decode_step_flops(step_config)
            
            # Accumulate FLOPS
            total_flops.attention += step_flops.attention
            total_flops.projections += step_flops.projections
            total_flops.mlp += step_flops.mlp
            total_flops.embeddings += step_flops.embeddings
            total_flops.layer_norm += step_flops.layer_norm
        
        # Calculate total
        total_flops.total = sum([
            total_flops.attention, total_flops.projections, total_flops.mlp,
            total_flops.embeddings, total_flops.layer_norm
        ])
        
        return total_flops
    
    def _calculate_decode_step_flops(self, step_config: ModelConfig) -> FLOPSComponents:
        """Calculate FLOPS for a single decode step using matrix-vector operations"""
        seq_len = step_config.sequence_length
        batch_size = step_config.batch_size
        hidden_size = step_config.hidden_size
        num_heads = step_config.num_attention_heads
        num_kv_heads = step_config.num_key_value_heads
        intermediate_size = step_config.intermediate_size
        num_layers = step_config.num_layers
        
        # Math_Used.md: L × [5Bd² + 2BdS] for decode
        # Attention FLOPS: 2BdS (Q @ K^T for 1 token against S cached keys)
        attention_flops = 2 * batch_size * hidden_size * seq_len
        
        # Projection FLOPS: 3Bd² (Q, K, V projections for 1 token)
        # Q projection: Bd²
        q_flops = batch_size * hidden_size * hidden_size
        
        # K, V projections (scaled by num_kv_heads for GQA/MQA)
        kv_scale = num_kv_heads / num_heads
        k_flops = batch_size * hidden_size * hidden_size * kv_scale
        v_flops = batch_size * hidden_size * hidden_size * kv_scale
        
        # Output projection: Bd²
        out_flops = batch_size * hidden_size * hidden_size
        
        projection_flops = q_flops + k_flops + v_flops + out_flops
        
        # MLP FLOPS: 2Bd² (Up + Down projections for 1 token)
        if step_config.model_type == ModelType.MOE:
            # MoE: routing + expert computation
            routing_flops = batch_size * step_config.num_experts * hidden_size
            expert_flops = step_config.top_k * batch_size * hidden_size * intermediate_size
            mlp_flops = routing_flops + expert_flops
        else:
            # Dense: 2Bd² (Up + Down projections)
            mlp_flops = 2 * batch_size * hidden_size * intermediate_size
        
        # Embedding FLOPS: matrix-vector operations
        embedding_flops = batch_size * hidden_size * step_config.vocab_size
        
        # Layer norm FLOPS: element-wise operations
        layer_norm_flops = 2 * batch_size * hidden_size
        
        # Scale by number of layers
        attention_flops *= num_layers
        projection_flops *= num_layers
        mlp_flops *= num_layers
        layer_norm_flops *= num_layers
        
        total_flops = attention_flops + projection_flops + mlp_flops + embedding_flops + layer_norm_flops
        
        return FLOPSComponents(
            attention=attention_flops,
            projections=projection_flops,
            mlp=mlp_flops,
            embeddings=embedding_flops,
            layer_norm=layer_norm_flops,
            total=total_flops
        )
    
    def _calculate_forward_flops(self) -> FLOPSComponents:
        """Calculate forward pass FLOPS"""
        seq_len = self.config.sequence_length
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        intermediate_size = self.config.intermediate_size
        num_layers = self.config.num_layers
        
        # Attention FLOPS
        attention_flops = self._calculate_attention_flops(seq_len, batch_size, hidden_size, num_heads, num_kv_heads)
        
        # Projection FLOPS
        projection_flops = self._calculate_projection_flops(seq_len, batch_size, hidden_size, num_heads, num_kv_heads)
        
        # MLP FLOPS
        mlp_flops = self._calculate_mlp_flops(seq_len, batch_size, hidden_size, intermediate_size)
        
        # Embedding FLOPS
        embedding_flops = self._calculate_embedding_flops(seq_len, batch_size, hidden_size)
        
        # Layer norm FLOPS
        layer_norm_flops = self._calculate_layer_norm_flops(seq_len, batch_size, hidden_size)
        
        # Scale by number of layers
        total_flops = FLOPSComponents(
            attention=attention_flops * num_layers,
            projections=projection_flops * num_layers,
            mlp=mlp_flops * num_layers,
            embeddings=embedding_flops,
            layer_norm=layer_norm_flops * num_layers,
            total=0.0
        )
        
        total_flops.total = sum([
            total_flops.attention, total_flops.projections, total_flops.mlp,
            total_flops.embeddings, total_flops.layer_norm
        ])
        
        return total_flops
    
    def _calculate_attention_flops(self, seq_len: int, batch_size: int, hidden_size: int,
                                 num_heads: int, num_kv_heads: int) -> float:
        """
        Calculate attention mechanism FLOPS (Floating Point Operations)
        
        This function calculates the number of floating-point operations required for
        the attention mechanism, including QK^T computation and attention score application.
        
        Mathematical Foundation:
        - Attention FLOPS are quadratic in sequence length: O(seq_len²)
        - QK^T computation: seq_len * batch_size * num_heads * seq_len * head_dim
        - Attention application: seq_len * batch_size * num_heads * seq_len * head_dim
        - Total: 2 * seq_len² * batch_size * num_heads * head_dim
        
        Formula:
        - QK^T FLOPS: seq_len * batch_size * num_heads * seq_len * (hidden_size / num_heads)
        - Attention FLOPS: seq_len * batch_size * num_heads * seq_len * (hidden_size / num_heads)
        - Total: 2 * seq_len² * batch_size * hidden_size
        
        Note: Q, K, V projections are counted separately in projection FLOPS to avoid double counting.
        
        Args:
            seq_len: Sequence length (number of tokens)
            batch_size: Batch size (number of sequences processed in parallel)
            hidden_size: Model hidden dimension (d_model)
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA/MQA)
        
        Returns:
            float: Total attention FLOPS
        
        Complexity: O(seq_len²) - quadratic in sequence length
        """
        # Note: Q, K, V projections are counted in projection FLOPS, not here
        # to avoid double counting
        
        # Attention scores: seq_len^2 * batch_size * num_heads * hidden_size_per_head
        hidden_per_head = hidden_size // num_heads
        attention_scores_flops = seq_len * seq_len * batch_size * num_heads * hidden_per_head
        
        # Attention output: seq_len * batch_size * hidden_size^2
        attention_output_flops = seq_len * batch_size * hidden_size * hidden_size
        
        return attention_scores_flops + attention_output_flops
    
    def _calculate_projection_flops(self, seq_len: int, batch_size: int, hidden_size: int, 
                                   num_heads: int, num_kv_heads: int) -> float:
        """Calculate projection layer FLOPS"""
        # Q projection (full size)
        q_flops = seq_len * batch_size * hidden_size * hidden_size
        
        # K, V projections (scaled by num_kv_heads for GQA/MQA)
        kv_scale = num_kv_heads / num_heads
        k_flops = seq_len * batch_size * hidden_size * hidden_size * kv_scale
        v_flops = seq_len * batch_size * hidden_size * hidden_size * kv_scale
        
        # Output projection
        out_flops = seq_len * batch_size * hidden_size * hidden_size
        
        return q_flops + k_flops + v_flops + out_flops
    
    def _calculate_mlp_flops(self, seq_len: int, batch_size: int, hidden_size: int,
                           intermediate_size: int) -> float:
        """Calculate MLP FLOPS"""
        if self.config.model_type == ModelType.MOE:
            # Use MoECalculator for MoE models
            context = MemoryContext(
                seq_len=seq_len,
                batch_size=batch_size,
                hidden_size=hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                intermediate_size=intermediate_size,
                dtype_bytes=self.config.dtype_bytes
            )
            moe_calc = MoECalculator(self.config)
            return moe_calc.calculate_flops(context)
        else:
            # Dense MLP: up_proj + down_proj
            up_proj_flops = seq_len * batch_size * hidden_size * intermediate_size
            down_proj_flops = seq_len * batch_size * intermediate_size * hidden_size
            return up_proj_flops + down_proj_flops
    
    
    def _calculate_embedding_flops(self, seq_len: int, batch_size: int, hidden_size: int) -> float:
        """Calculate embedding FLOPS"""
        # Input embeddings lookup (simplified)
        return seq_len * batch_size * hidden_size
    
    def _calculate_layer_norm_flops(self, seq_len: int, batch_size: int, hidden_size: int) -> float:
        """Calculate layer normalization FLOPS"""
        # Pre-attention and pre-MLP layer norms
        # Each layer norm operation involves multiply-add operations (2 FLOPS per element)
        return 2 * seq_len * batch_size * hidden_size * 2
    
    def generate_table(self, mode: OperationMode, table_type: str = "memory") -> str:
        """Generate formatted table output"""
        if table_type == "all":
            memory_components = self.calculate_memory(mode)
            flops_components = self.calculate_flops(mode)
            reuse_components = self.calculate_reuse(mode)
            
            # Add hardware comparison for comprehensive analysis
            formatter = TableFormatter(self.config)
            # Pass validation result to formatter if available
            if hasattr(self, '_validation_result') and self._validation_result:
                formatter._validation_result = self._validation_result
            # Map data types for hardware comparison
            dtype_mapping = {
                'fp32': 'fp32',
                'fp16': 'fp16', 
                'bf16': 'fp16',  # BF16 uses similar FLOPS as FP16
                'int8': 'fp16',  # Use FP16 as approximation
                'int4': 'fp16',  # Use FP16 as approximation
                'fp8': 'fp16'    # Use FP16 as approximation
            }
            mapped_dtype = dtype_mapping.get(self.config.dtype.value, 'fp16')
            hardware_comparison = formatter.format_hardware_comparison(
                memory_components.total / (1024**3),  # Convert to GB
                flops_components.total,
                mapped_dtype
            )
            
            all_tables = self._format_all_tables(memory_components, flops_components, reuse_components, mode)
            return all_tables + "\n\n" + hardware_comparison
        else:
            # Use TableFormatter for individual tables
            formatter = TableFormatter(self.config)
            # Pass validation result to formatter if available
            if hasattr(self, '_validation_result') and self._validation_result:
                formatter._validation_result = self._validation_result
            if table_type == "memory":
                components = self.calculate_memory(mode)
                return formatter.format_memory_table(components, mode.value)
            elif table_type == "flops":
                components = self.calculate_flops(mode)
                return formatter.format_flops_table(components, mode.value)
            elif table_type == "reuse":
                components = self.calculate_reuse(mode)
                return formatter.format_reuse_table(components, mode.value)
            else:
                raise ValueError("table_type must be 'memory', 'flops', 'reuse', or 'all'")
    
    def _format_bytes_human(self, bytes_value: float) -> str:
        """Format bytes in human readable format (KB, MB, GB, TB, PB, EB)"""
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
        unit_index = 0
        value = bytes_value
        
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        
        return f"{value:.1f} {units[unit_index]}"
    
    def _format_flops_human(self, flops_value: float) -> str:
        """Format FLOPS in human readable format (FLOP, KFLOP, MFLOP, GFLOP, TFLOP, PFLOP, EFLOP)"""
        if flops_value == 0:
            return "0 FLOP"
        
        units = ['FLOP', 'KFLOP', 'MFLOP', 'GFLOP', 'TFLOP', 'PFLOP', 'EFLOP']
        unit_index = 0
        value = flops_value
        
        while value >= 1000 and unit_index < len(units) - 1:
            value /= 1000
            unit_index += 1
        
        return f"{value:.1f} {units[unit_index]}"
    
    def _format_parameters_human(self, param_count: int) -> str:
        """Format parameter count in human readable format (M, B, T)"""
        if param_count == 0:
            return "0"
        
        if param_count < 1_000_000:
            return f"{param_count:,}"
        elif param_count < 1_000_000_000:
            return f"{param_count / 1_000_000:.1f}M"
        elif param_count < 1_000_000_000_000:
            return f"{param_count / 1_000_000_000:.1f}B"
        else:
            return f"{param_count / 1_000_000_000_000:.1f}T"

    
    
    def _format_all_tables(self, memory_components: MemoryComponents, flops_components: FLOPSComponents, 
                          reuse_components: ReuseComponents, mode: OperationMode) -> str:
        """Format all three tables together"""
        formatter = TableFormatter(self.config)
        memory_table = formatter.format_memory_table(memory_components, mode.value)
        flops_table = formatter.format_flops_table(flops_components, mode.value)
        reuse_table = formatter.format_reuse_table(reuse_components, mode.value)
        
        return memory_table + "\n" + flops_table + "\n" + reuse_table
    
    def _format_reuse_human(self, reuse_value: float) -> str:
        """Format arithmetic intensity in human readable format"""
        if reuse_value == 0:
            return "0 FLOPS/B"
        
        if reuse_value < 1:
            return f"{reuse_value:.3f} FLOPS/B"
        elif reuse_value < 1000:
            return f"{reuse_value:.1f} FLOPS/B"
        elif reuse_value < 1000000:
            return f"{reuse_value/1000:.1f} KFLOPS/B"
        else:
            return f"{reuse_value/1000000:.1f} MFLOPS/B"
    


def create_qwen3_omni_config() -> ModelConfig:
    """Create configuration for Qwen3-Omni-30B-A3B-Instruct model"""
    return ModelConfig(
        num_layers=32,  # Estimated for 30B model
        hidden_size=4096,  # Typical for 30B models
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA configuration
        intermediate_size=11008,  # Typical for 30B models
        vocab_size=151936,  # Qwen vocabulary size
        model_type=ModelType.MOE,
        num_experts=8,  # A3B configuration
        expert_capacity_factor=1.0,
        top_k=2,
        sequence_length=2048,
        batch_size=1,
        decode_len=64,  # Default decode length
        dtype=DataType.BF16  # Use DataType enum instead of hardcoded bytes
    )


if __name__ == "__main__":
    # Example usage
    config = create_qwen3_omni_config()
    calculator = TransformerCalculator(config)
    
    print("Qwen3-Omni-30B-A3B-Instruct Analysis")
    print("=" * 50)
    
    # Memory analysis
    print(calculator.generate_table(OperationMode.PRETRAINING, "memory"))
    print(calculator.generate_table(OperationMode.PREFILL, "memory"))
    print(calculator.generate_table(OperationMode.DECODE, "memory"))
    
    # FLOPS analysis
    print(calculator.generate_table(OperationMode.PRETRAINING, "flops"))
    print(calculator.generate_table(OperationMode.PREFILL, "flops"))
    print(calculator.generate_table(OperationMode.DECODE, "flops"))
