"""
Advanced MoE Optimizations and Routing Calculations

⚠️  EXPERIMENTAL MODULE - Advanced MoE Features

This module provides experimental MoE-specific optimizations including:
- Expert routing and selection algorithms (DEMONSTRATIVE)
- Capacity management and load balancing (SIMPLIFIED)
- Expert communication patterns (APPROXIMATED)
- Advanced memory optimizations for MoE models (EXPERIMENTAL)

IMPORTANT LIMITATIONS:
- Routing strategies are demonstrative and simplified
- Communication costs are approximated (not production-ready)
- Load balancing uses basic variance calculations
- Capacity management is idealized (no real-world constraints)

For production use, implement proper routing algorithms or use
the simplified MoE calculations in the core transformer_calculator.py
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from transformer_calculator import (
    ModelConfig, ModelType, DataType, MemoryContext, CalculationContext, OperationMode
)


class RoutingStrategy(Enum):
    """Expert routing strategies"""
    TOP_K = "top_k"
    TOP_K_GATING = "top_k_gating"
    SWITCH = "switch"
    GSHARD = "gshard"
    EXPERT_CHOICE = "expert_choice"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for MoE"""
    UNIFORM = "uniform"
    CAPACITY_BASED = "capacity_based"
    ADAPTIVE = "adaptive"
    EXPERT_CHOICE = "expert_choice"


@dataclass
class ExpertRoutingConfig:
    """Configuration for expert routing"""
    strategy: RoutingStrategy = RoutingStrategy.TOP_K
    top_k: int = 2
    capacity_factor: float = 1.0
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.UNIFORM
    expert_choice_rounds: int = 1
    gating_noise: float = 0.0
    aux_loss_weight: float = 0.01


@dataclass
class ExpertMetrics:
    """Metrics for expert performance"""
    expert_id: int
    load: float  # Current load (0.0 to 1.0)
    capacity: float  # Expert capacity
    utilization: float  # Load / capacity
    communication_cost: float  # Communication overhead
    memory_usage: float  # Memory usage in bytes
    flops_usage: float  # FLOPS usage


@dataclass
class RoutingResult:
    """Result of expert routing"""
    selected_experts: List[int]
    routing_weights: List[float]
    load_balance_loss: float
    communication_cost: float
    routing_flops: float
    routing_memory: float


class MoERouter:
    """Advanced MoE routing and expert selection"""
    
    def __init__(self, config: ExpertRoutingConfig):
        self.config = config
    
    def route_experts(self, context: CalculationContext, 
                     num_experts: int, top_k: int) -> RoutingResult:
        """Route tokens to experts using the configured strategy"""
        
        if self.config.strategy == RoutingStrategy.TOP_K:
            return self._top_k_routing(context, num_experts, top_k)
        elif self.config.strategy == RoutingStrategy.TOP_K_GATING:
            return self._top_k_gating_routing(context, num_experts, top_k)
        elif self.config.strategy == RoutingStrategy.SWITCH:
            return self._switch_routing(context, num_experts, top_k)
        elif self.config.strategy == RoutingStrategy.GSHARD:
            return self._gshard_routing(context, num_experts, top_k)
        elif self.config.strategy == RoutingStrategy.EXPERT_CHOICE:
            return self._expert_choice_routing(context, num_experts, top_k)
        else:
            raise ValueError(f"Unknown routing strategy: {self.config.strategy}")
    
    def _top_k_routing(self, context: CalculationContext, 
                      num_experts: int, top_k: int) -> RoutingResult:
        """Standard top-k routing"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        
        # Each token selects top_k experts
        selected_experts = [list(range(top_k)) for _ in range(seq_len * batch_size)]
        routing_weights = [1.0 / top_k] * top_k
        
        # Calculate routing costs
        routing_flops = seq_len * batch_size * num_experts * 2  # Gating network
        routing_memory = seq_len * batch_size * num_experts * context.dtype_bytes
        
        # Load balancing loss (encourage uniform expert usage)
        # Flatten all selected experts for proper load balancing calculation
        all_selected_experts = [expert for token_experts in selected_experts for expert in token_experts]
        load_balance_loss = self._calculate_load_balance_loss(all_selected_experts, num_experts)
        
        # Communication cost (expert-to-expert communication)
        communication_cost = self._calculate_communication_cost(all_selected_experts, num_experts)
        
        return RoutingResult(
            selected_experts=selected_experts,  # Return all token selections
            routing_weights=routing_weights,
            load_balance_loss=load_balance_loss,
            communication_cost=communication_cost,
            routing_flops=routing_flops,
            routing_memory=routing_memory
        )
    
    def _top_k_gating_routing(self, context: CalculationContext, 
                             num_experts: int, top_k: int) -> RoutingResult:
        """Top-k routing with gating network"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        
        # Gating network: hidden_size -> num_experts
        gating_flops = seq_len * batch_size * context.hidden_size * num_experts
        gating_memory = seq_len * batch_size * num_experts * context.dtype_bytes
        
        # Top-k selection
        selection_flops = seq_len * batch_size * num_experts * math.log2(num_experts)
        
        # Softmax normalization
        softmax_flops = seq_len * batch_size * top_k * 3  # exp, sum, div
        
        total_routing_flops = gating_flops + selection_flops + softmax_flops
        total_routing_memory = gating_memory + seq_len * batch_size * top_k * context.dtype_bytes
        
        # Load balancing with auxiliary loss
        load_balance_loss = self._calculate_auxiliary_loss(num_experts, top_k)
        
        return RoutingResult(
            selected_experts=list(range(top_k)),
            routing_weights=[1.0 / top_k] * top_k,
            load_balance_loss=load_balance_loss,
            communication_cost=0.0,  # Simplified
            routing_flops=total_routing_flops,
            routing_memory=total_routing_memory
        )
    
    def _switch_routing(self, context: CalculationContext, 
                       num_experts: int, top_k: int) -> RoutingResult:
        """Switch Transformer routing (top-1 with load balancing)"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        
        # Switch uses top-1 routing with strong load balancing
        routing_flops = seq_len * batch_size * context.hidden_size * num_experts
        routing_memory = seq_len * batch_size * num_experts * context.dtype_bytes
        
        # Switch loss (encourages load balancing)
        switch_loss = self._calculate_switch_loss(num_experts)
        
        return RoutingResult(
            selected_experts=[0],  # Top-1
            routing_weights=[1.0],
            load_balance_loss=switch_loss,
            communication_cost=0.0,
            routing_flops=routing_flops,
            routing_memory=routing_memory
        )
    
    def _gshard_routing(self, context: CalculationContext, 
                       num_experts: int, top_k: int) -> RoutingResult:
        """GShard routing with capacity constraints"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        
        # GShard uses capacity-aware routing
        capacity = int(seq_len * batch_size * self.config.capacity_factor)
        
        # Routing with capacity constraints
        routing_flops = seq_len * batch_size * context.hidden_size * num_experts
        capacity_flops = seq_len * batch_size * num_experts * math.log2(capacity)
        
        total_routing_flops = routing_flops + capacity_flops
        total_routing_memory = seq_len * batch_size * num_experts * context.dtype_bytes
        
        # Capacity-based load balancing
        load_balance_loss = self._calculate_capacity_loss(num_experts, capacity)
        
        return RoutingResult(
            selected_experts=list(range(top_k)),
            routing_weights=[1.0 / top_k] * top_k,
            load_balance_loss=load_balance_loss,
            communication_cost=0.0,
            routing_flops=total_routing_flops,
            routing_memory=total_routing_memory
        )
    
    def _expert_choice_routing(self, context: CalculationContext, 
                              num_experts: int, top_k: int) -> RoutingResult:
        """Expert Choice routing (experts choose tokens)"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        
        # Expert Choice: experts select their top tokens
        rounds = self.config.expert_choice_rounds
        
        # Per-expert gating
        expert_gating_flops = num_experts * seq_len * batch_size * context.hidden_size
        expert_selection_flops = num_experts * seq_len * batch_size * math.log2(seq_len * batch_size)
        
        total_routing_flops = rounds * (expert_gating_flops + expert_selection_flops)
        total_routing_memory = rounds * num_experts * seq_len * batch_size * context.dtype_bytes
        
        # Expert Choice naturally balances load
        load_balance_loss = 0.0  # Expert Choice is inherently balanced
        
        return RoutingResult(
            selected_experts=list(range(top_k)),
            routing_weights=[1.0 / top_k] * top_k,
            load_balance_loss=load_balance_loss,
            communication_cost=0.0,
            routing_flops=total_routing_flops,
            routing_memory=total_routing_memory
        )
    
    def _calculate_load_balance_loss(self, selected_experts: List[int], 
                                    num_experts: int) -> float:
        """
        Calculate load balancing loss (SIMPLIFIED APPROXIMATION)
        
        ASSUMPTIONS:
        - Uses basic variance calculation (not production-ready)
        - Assumes uniform expert capacity (ignores real capacity constraints)
        - No consideration of expert utilization patterns
        - Simplified to encourage uniform distribution only
        
        LIMITATIONS:
        - Real load balancing involves complex capacity management
        - Should consider expert utilization, capacity constraints
        - Production systems use sophisticated load balancing algorithms
        """
        # Simplified load balancing loss
        expert_counts = [0] * num_experts
        for expert in selected_experts:
            if isinstance(expert, int) and 0 <= expert < num_experts:
                expert_counts[expert] += 1
        
        # Variance-based loss (encourage uniform distribution)
        mean_count = sum(expert_counts) / num_experts
        variance = sum((count - mean_count) ** 2 for count in expert_counts) / num_experts
        return variance
    
    def _calculate_auxiliary_loss(self, num_experts: int, top_k: int) -> float:
        """Calculate auxiliary loss for load balancing"""
        # Auxiliary loss encourages uniform expert usage
        return self.config.aux_loss_weight * num_experts / top_k
    
    def _calculate_switch_loss(self, num_experts: int) -> float:
        """Calculate Switch Transformer loss"""
        # Switch loss with noise for load balancing
        return self.config.aux_loss_weight * num_experts * (1 + self.config.gating_noise)
    
    def _calculate_capacity_loss(self, num_experts: int, capacity: int) -> float:
        """Calculate capacity-based loss"""
        # Capacity loss encourages efficient capacity usage
        return self.config.aux_loss_weight * num_experts / capacity
    
    def _calculate_communication_cost(self, selected_experts: List[int], 
                                   num_experts: int) -> float:
        """
        Calculate expert-to-expert communication cost (SIMPLIFIED APPROXIMATION)
        
        ASSUMPTIONS:
        - Assumes all-to-all communication between selected experts
        - Ignores network topology and routing constraints
        - No consideration of communication bandwidth limitations
        - Simplified to basic combinatorial calculation
        
        LIMITATIONS:
        - Real communication involves complex network routing
        - Should consider network topology, bandwidth constraints
        - Production systems use sophisticated communication patterns
        - Ignores expert placement and data locality
        """
        # Simplified communication cost
        # Flatten the list if it contains nested lists
        flat_experts = []
        for expert in selected_experts:
            if isinstance(expert, list):
                flat_experts.extend(expert)
            else:
                flat_experts.append(expert)
        
        unique_experts = len(set(flat_experts))
        return unique_experts * (unique_experts - 1) / 2  # All-to-all communication


class MoECapacityManager:
    """
    ⚠️  EXPERIMENTAL: Manages expert capacity and load balancing
    
    This class provides simplified capacity management calculations.
    Real-world MoE systems require sophisticated capacity management
    with dynamic load balancing and expert utilization tracking.
    
    LIMITATIONS:
    - Simplified capacity calculations (not production-ready)
    - No dynamic load balancing
    - Ignores expert utilization patterns
    - Assumes uniform expert capacity
    """
    
    def __init__(self, num_experts: int, capacity_factor: float = 1.0):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_metrics: List[ExpertMetrics] = []
        
        # Initialize expert metrics
        for expert_id in range(num_experts):
            self.expert_metrics.append(ExpertMetrics(
                expert_id=expert_id,
                load=0.0,
                capacity=1.0,
                utilization=0.0,
                communication_cost=0.0,
                memory_usage=0.0,
                flops_usage=0.0
            ))
    
    def calculate_optimal_capacity(self, context: CalculationContext, 
                                top_k: int) -> float:
        """Calculate optimal expert capacity"""
        total_tokens = context.seq_len * context.batch_size
        tokens_per_expert = total_tokens * top_k / self.num_experts
        optimal_capacity = tokens_per_expert * self.capacity_factor
        return optimal_capacity
    
    def update_expert_metrics(self, routing_result: RoutingResult, 
                            context: CalculationContext):
        """Update expert metrics based on routing results"""
        for expert_id in routing_result.selected_experts:
            if expert_id < len(self.expert_metrics):
                expert = self.expert_metrics[expert_id]
                expert.load += 1.0
                expert.utilization = expert.load / expert.capacity
                expert.communication_cost += routing_result.communication_cost
                expert.memory_usage += routing_result.routing_memory
                expert.flops_usage += routing_result.routing_flops
    
    def get_load_balance_score(self) -> float:
        """Get overall load balance score (0.0 = perfect, 1.0 = worst)"""
        if not self.expert_metrics:
            return 0.0
        
        utilizations = [expert.utilization for expert in self.expert_metrics]
        mean_utilization = sum(utilizations) / len(utilizations)
        variance = sum((util - mean_utilization) ** 2 for util in utilizations) / len(utilizations)
        return variance ** 0.5  # Standard deviation
    
    def get_communication_efficiency(self) -> float:
        """Get communication efficiency score"""
        if not self.expert_metrics:
            return 1.0
        
        total_communication = sum(expert.communication_cost for expert in self.expert_metrics)
        total_work = sum(expert.flops_usage for expert in self.expert_metrics)
        
        if total_work == 0:
            return 1.0
        
        return 1.0 - (total_communication / total_work)


class AdvancedMoECalculator:
    """
    ⚠️  EXPERIMENTAL: Advanced MoE calculator with routing optimizations
    
    This class provides demonstrative routing strategies and simplified
    capacity management. The calculations are approximated and should
    not be used for production systems without proper validation.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.routing_config = ExpertRoutingConfig(
            strategy=RoutingStrategy.TOP_K,
            top_k=config.top_k,
            capacity_factor=config.expert_capacity_factor
        )
        self.router = MoERouter(self.routing_config)
        self.capacity_manager = MoECapacityManager(
            config.num_experts, 
            config.expert_capacity_factor
        )
    
    def calculate_advanced_moe_memory(self, context: CalculationContext) -> Dict[str, float]:
        """Calculate advanced MoE memory with routing optimizations"""
        
        # Get routing result
        routing_result = self.router.route_experts(
            context, self.config.num_experts, self.config.top_k
        )
        
        # Update capacity manager
        self.capacity_manager.update_expert_metrics(routing_result, context)
        
        # Calculate memory components
        memory_components = {}
        
        # Routing memory
        memory_components['routing'] = routing_result.routing_memory
        
        # Expert memory (HBM storage)
        expert_memory = self._calculate_expert_memory(context)
        memory_components['expert_storage'] = expert_memory
        
        # Activated expert memory (GPU)
        activated_memory = self._calculate_activated_expert_memory(context, routing_result)
        memory_components['activated_experts'] = activated_memory
        
        # Communication memory
        communication_memory = self._calculate_communication_memory(routing_result)
        memory_components['communication'] = communication_memory
        
        # Load balancing overhead
        load_balance_memory = self._calculate_load_balance_memory()
        memory_components['load_balancing'] = load_balance_memory
        
        return memory_components
    
    def calculate_advanced_moe_flops(self, context: CalculationContext) -> Dict[str, float]:
        """Calculate advanced MoE FLOPS with routing optimizations"""
        
        # Get routing result
        routing_result = self.router.route_experts(
            context, self.config.num_experts, self.config.top_k
        )
        
        # Calculate FLOPS components
        flops_components = {}
        
        # Routing FLOPS
        flops_components['routing'] = routing_result.routing_flops
        
        # Expert computation FLOPS
        expert_flops = self._calculate_expert_flops(context, routing_result)
        flops_components['expert_computation'] = expert_flops
        
        # Communication FLOPS
        communication_flops = self._calculate_communication_flops(routing_result)
        flops_components['communication'] = communication_flops
        
        # Load balancing FLOPS
        load_balance_flops = self._calculate_load_balance_flops()
        flops_components['load_balancing'] = load_balance_flops
        
        return flops_components
    
    def _calculate_expert_memory(self, context: CalculationContext) -> float:
        """Calculate expert storage memory in HBM"""
        num_experts = self.config.num_experts
        hidden_size = context.hidden_size
        intermediate_size = context.intermediate_size
        dtype_bytes = context.dtype_bytes
        
        # Each expert: hidden_size -> intermediate_size -> hidden_size
        expert_params = hidden_size * intermediate_size + intermediate_size * hidden_size
        total_expert_memory = num_experts * expert_params * dtype_bytes
        
        return total_expert_memory
    
    def _calculate_activated_expert_memory(self, context: CalculationContext, 
                                          routing_result: RoutingResult) -> float:
        """Calculate memory for activated experts on GPU"""
        top_k = self.config.top_k
        capacity = int(context.seq_len * context.batch_size * self.config.expert_capacity_factor)
        intermediate_size = context.intermediate_size
        dtype_bytes = context.dtype_bytes
        
        # Only activated experts are moved to GPU
        activated_memory = top_k * capacity * intermediate_size * dtype_bytes
        
        return activated_memory
    
    def _calculate_communication_memory(self, routing_result: RoutingResult) -> float:
        """Calculate communication memory overhead"""
        # Communication buffers for expert-to-expert communication
        # More realistic: communication cost represents number of expert pairs
        # Each pair needs buffer space proportional to expert size
        expert_buffer_size = 1024 * 1024  # 1MB per expert pair (realistic for large models)
        return routing_result.communication_cost * expert_buffer_size
    
    def _calculate_load_balance_memory(self) -> float:
        """Calculate load balancing memory overhead"""
        # Memory for load balancing statistics and auxiliary losses
        return self.config.num_experts * 64  # Simplified
    
    def _calculate_expert_flops(self, context: CalculationContext, 
                              routing_result: RoutingResult) -> float:
        """Calculate expert computation FLOPS"""
        seq_len = context.seq_len
        batch_size = context.batch_size
        hidden_size = context.hidden_size
        intermediate_size = context.intermediate_size
        top_k = self.config.top_k
        
        # Expert computation: up_proj + down_proj
        up_proj_flops = seq_len * batch_size * hidden_size * intermediate_size
        down_proj_flops = seq_len * batch_size * intermediate_size * hidden_size
        expert_flops_per_token = up_proj_flops + down_proj_flops
        
        # Scale by number of activated experts
        total_expert_flops = expert_flops_per_token * top_k
        
        return total_expert_flops
    
    def _calculate_communication_flops(self, routing_result: RoutingResult) -> float:
        """Calculate communication FLOPS"""
        # Communication FLOPS for expert coordination
        return routing_result.communication_cost * 1000  # Simplified
    
    def _calculate_load_balance_flops(self) -> float:
        """Calculate load balancing FLOPS"""
        # FLOPS for load balancing calculations
        return self.config.num_experts * 100  # Simplified
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics"""
        return {
            'load_balance_score': self.capacity_manager.get_load_balance_score(),
            'communication_efficiency': self.capacity_manager.get_communication_efficiency(),
            'expert_utilization': [expert.utilization for expert in self.capacity_manager.expert_metrics],
            'total_communication_cost': sum(expert.communication_cost for expert in self.capacity_manager.expert_metrics),
            'total_memory_usage': sum(expert.memory_usage for expert in self.capacity_manager.expert_metrics),
            'total_flops_usage': sum(expert.flops_usage for expert in self.capacity_manager.expert_metrics)
        }


def create_advanced_moe_config(base_config: ModelConfig, 
                             routing_strategy: RoutingStrategy = RoutingStrategy.TOP_K,
                             load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.UNIFORM) -> ModelConfig:
    """Create an advanced MoE configuration with routing optimizations"""
    
    # Create new config with MoE optimizations
    advanced_config = ModelConfig(
        num_layers=base_config.num_layers,
        hidden_size=base_config.hidden_size,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        vocab_size=base_config.vocab_size,
        model_type=ModelType.MOE,
        num_experts=base_config.num_experts,
        top_k=base_config.top_k,
        expert_capacity_factor=base_config.expert_capacity_factor,
        dtype=base_config.dtype,
        sequence_length=base_config.sequence_length,
        batch_size=base_config.batch_size,
        decode_len=base_config.decode_len
    )
    
    return advanced_config


def benchmark_moe_routing_strategies(config: ModelConfig, 
                                   context: CalculationContext) -> Dict[str, Any]:
    """Benchmark different MoE routing strategies"""
    
    strategies = [
        RoutingStrategy.TOP_K,
        RoutingStrategy.TOP_K_GATING,
        RoutingStrategy.SWITCH,
        RoutingStrategy.GSHARD,
        RoutingStrategy.EXPERT_CHOICE
    ]
    
    results = {}
    
    for strategy in strategies:
        # Create calculator with this routing strategy
        routing_config = ExpertRoutingConfig(strategy=strategy)
        router = MoERouter(routing_config)
        
        # Get routing result
        routing_result = router.route_experts(context, config.num_experts, config.top_k)
        
        # Store results
        results[strategy.value] = {
            'routing_flops': routing_result.routing_flops,
            'routing_memory': routing_result.routing_memory,
            'load_balance_loss': routing_result.load_balance_loss,
            'communication_cost': routing_result.communication_cost
        }
    
    return results


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
        model_type=ModelType.MOE,
        num_experts=8,
        top_k=2,
        expert_capacity_factor=1.0,
        dtype=DataType.BF16
    )
    
    context = CalculationContext.from_config(config, OperationMode.PREFILL)
    
    # Create advanced MoE calculator
    advanced_calc = AdvancedMoECalculator(config)
    
    # Calculate advanced memory and FLOPS
    memory_components = advanced_calc.calculate_advanced_moe_memory(context)
    flops_components = advanced_calc.calculate_advanced_moe_flops(context)
    
    print("Advanced MoE Memory Components:")
    for component, memory in memory_components.items():
        print(f"  {component}: {memory/1e9:.2f} GB")
    
    print("\nAdvanced MoE FLOPS Components:")
    for component, flops in flops_components.items():
        print(f"  {component}: {flops/1e12:.2f} TFLOP")
    
    # Get routing metrics
    metrics = advanced_calc.get_routing_metrics()
    print(f"\nRouting Metrics:")
    print(f"  Load Balance Score: {metrics['load_balance_score']:.3f}")
    print(f"  Communication Efficiency: {metrics['communication_efficiency']:.3f}")
    
    # Benchmark routing strategies
    benchmark_results = benchmark_moe_routing_strategies(config, context)
    print(f"\nRouting Strategy Benchmark:")
    for strategy, results in benchmark_results.items():
        print(f"  {strategy}: {results['routing_flops']/1e9:.2f} GFLOP, {results['routing_memory']/1e6:.2f} MB")
