"""
Table Formatting Utilities

This module provides formatting utilities for tables and human-readable numbers
used in the transformer calculator output.
"""

from typing import List, Tuple, Dict, Any
try:
    from .hardware_specs import get_hardware_comparison, format_hardware_summary, get_hardware_json
except ImportError:
    # Fallback for when run as script
    from hardware_specs import get_hardware_comparison, format_hardware_summary, get_hardware_json


class TableFormatter:
    """Formats calculation results into human-readable tables"""
    
    def __init__(self, config):
        self.config = config
    
    def format_header(self, mode: str, table_type: str) -> str:
        """Format table header"""
        mode_names = {
            'pretraining': 'PRETRAINING',
            'prefill': 'PREFILL', 
            'decode': 'DECODE'
        }
        
        table_names = {
            'memory': 'MEMORY ANALYSIS',
            'flops': 'FLOPS ANALYSIS',
            'reuse': 'ARITHMETIC INTENSITY (REUSE)',
            'all': 'COMPREHENSIVE ANALYSIS'
        }
        
        mode_name = mode_names.get(mode, mode.upper())
        table_name = table_names.get(table_type, table_type.upper())
        
        return f"TRANSFORMER {table_name} - {mode_name}"
    
    def format_model_info(self) -> str:
        """Format model information section"""
        model_type = self.config.model_type.value.upper()
        total_params = self._calculate_total_parameters()
        param_str = self._format_parameters_human(total_params)
        
        info = f"""
Model: {model_type}
Model Parameters: {total_params:,} ({param_str})
Layers: {self.config.num_layers}, Hidden Size: {self.config.hidden_size}
Sequence Length: {self.config.sequence_length}, Batch Size: {self.config.batch_size}
Attention Heads: {self.config.num_attention_heads} (KV: {self.config.num_key_value_heads}), Intermediate Size: {self.config.intermediate_size}
"""
        
        if self.config.model_type.value == 'moe':
            info += f"MoE: {self.config.num_experts} experts, top-{self.config.top_k}, capacity: {self.config.expert_capacity_factor}\n"
        
        info += f"Vocab Size: {self.config.vocab_size:,}, Precision: {self.config.dtype.value.upper()} ({self.config.dtype_bytes} bytes)"
        
        return info
    
    def _calculate_total_parameters(self) -> int:
        """Calculate total model parameters"""
        # Import here to avoid circular imports
        from transformer_calculator import calculate_model_parameters
        return calculate_model_parameters(self.config)
    
    def _format_parameters_human(self, param_count: int) -> str:
        """Format parameter count in human readable format"""
        if param_count < 1000000:
            return f"{param_count / 1000:.1f}K"
        elif param_count < 1000000000:
            return f"{param_count / 1000000:.1f}M"
        elif param_count < 1000000000000:
            return f"{param_count / 1000000000:.1f}B"
        elif param_count < 1000000000000000:
            return f"{param_count / 1000000000000:.1f}T"
        else:
            return f"{param_count / 1000000000000000:.1f}P"  # Peta-parameters
    
    def _format_bytes_human(self, bytes_value: float) -> str:
        """Format bytes in human readable format"""
        if bytes_value == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
        value = bytes_value
        unit_index = 0
        
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        
        return f"{value:.1f} {units[unit_index]}"
    
    def _format_flops_human(self, flops_value: float) -> str:
        """Format FLOPS in human readable format"""
        if flops_value == 0:
            return "0 FLOP"
        
        units = ["FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP", "EFLOP"]
        value = flops_value
        unit_index = 0
        
        while value >= 1000 and unit_index < len(units) - 1:
            value /= 1000
            unit_index += 1
        
        return f"{value:.1f} {units[unit_index]}"
    
    def _format_reuse_human(self, reuse_value: float) -> str:
        """Format reuse value in human readable format"""
        if reuse_value == 0:
            return "0 FLOPS/B"
        
        if reuse_value >= 1000000:
            return f"{reuse_value / 1000000:.1f} MFLOPS/B"
        elif reuse_value >= 1000:
            return f"{reuse_value / 1000:.1f} KFLOPS/B"
        else:
            return f"{reuse_value:.1f} FLOPS/B"
    
    def format_memory_table(self, memory_components, mode: str) -> str:
        """Format memory components into a table"""
        header = self.format_header(mode, 'memory')
        model_info = self.format_model_info()
        
        # Calculate percentages
        total = memory_components.total
        components = [
            ("Attention", memory_components.attention),
            ("Projections", memory_components.projections),
            ("MLP", memory_components.mlp),
            ("Embeddings", memory_components.embeddings),
            ("Layer Norm", memory_components.layer_norm),
            ("Weights", memory_components.weights),
            ("KV Cache", memory_components.kv_cache)
        ]
        
        # Column widths
        name_w = 20
        value_w = 20
        percent_w = 10

        table = f"""
{'='*80}
{header}
{'='*80}
{model_info}
{'='*80}
{('Component'):<{name_w}} {('Memory'):<{value_w}} {('Percent'):>{percent_w}}
{'-'*80}"""
        
        for name, value in components:
            percentage = (value / total * 100) if total > 0 else 0
            perc_str = f"{percentage:.2f}%"
            table += f"""
{name:<{name_w}} {self._format_bytes_human(value):<{value_w}} {perc_str:>{percent_w}}"""
        
        total_perc_str = "100.00%"
        table += f"""
{'-'*80}
{'TOTAL':<{name_w}} {self._format_bytes_human(total):<{value_w}} {total_perc_str:>{percent_w}}
{'='*80}
"""
        
        return table
    
    def format_flops_table(self, flops_components, mode: str) -> str:
        """Format FLOPS components into a table"""
        header = self.format_header(mode, 'flops')
        model_info = self.format_model_info()
        
        # Calculate percentages
        total = flops_components.total
        components = [
            ("Embeddings", flops_components.embeddings),
            ("Layer Norm", flops_components.layer_norm),
            ("Projections", flops_components.projections),
            ("Attention", flops_components.attention),
            ("MLP", flops_components.mlp)
        ]
        
        # Column widths
        name_w = 20
        value_w = 20
        percent_w = 10

        table = f"""
{'='*80}
{header}
{'='*80}
{model_info}
{'='*80}
{('Component'):<{name_w}} {('FLOPS'):<{value_w}} {('Percent'):>{percent_w}}
{'-'*80}"""
        
        for name, value in components:
            percentage = (value / total * 100) if total > 0 else 0
            perc_str = f"{percentage:.2f}%"
            table += f"""
{name:<{name_w}} {self._format_flops_human(value):<{value_w}} {perc_str:>{percent_w}}"""
        
        total_perc_str = "100.00%"
        table += f"""
{'-'*80}
{'TOTAL':<{name_w}} {self._format_flops_human(total):<{value_w}} {total_perc_str:>{percent_w}}
{'='*80}
"""
        
        return table
    
    def format_reuse_table(self, reuse_components, mode: str) -> str:
        """Format reuse components into a table"""
        header = self.format_header(mode, 'reuse')
        model_info = self.format_model_info()
        
        components = [
            ("Pre-Attention", reuse_components.pre_attention),
            ("Attention", reuse_components.attention),
            ("Post-Attention", reuse_components.post_attention)
        ]
        
        # Column widths
        stage_w = 20
        intensity_w = 20
        human_w = 15
        efficiency_w = 12
        
        table = f"""
{'='*80}
{header}
{'='*80}
{model_info}
{'='*80}
{('Stage'):<{stage_w}} {('Reuse'):<{intensity_w}} {('Human Readable'):<{human_w}} {('Efficiency'):>{efficiency_w}}
{'-'*80}"""
        
        for name, value in components:
            efficiency = "High" if value > 100 else "Medium" if value > 10 else "Low"
            table += f"""
{name:<{stage_w}} {value:<{intensity_w}.2f} {self._format_reuse_human(value):<{human_w}} {efficiency:>{efficiency_w}}"""
        
        total_efficiency = "High" if reuse_components.total > 100 else "Medium" if reuse_components.total > 10 else "Low"
        table += f"""
{'-'*80}
{'TOTAL':<{stage_w}} {reuse_components.total:<{intensity_w}.2f} {self._format_reuse_human(reuse_components.total):<{human_w}} {total_efficiency:>{efficiency_w}}
{'='*80}
"""
        
        return table
    
    def format_hardware_comparison(self, memory_gb: float, flops: float, 
                                 data_type: str = "fp16") -> str:
        """
        Format hardware capability comparison
        
        Args:
            memory_gb: Model memory requirement in GB
            flops: Model FLOPS requirement
            data_type: Data type for FLOPS comparison
        
        Returns:
            Formatted hardware comparison string
        """
        comparison_results = get_hardware_comparison(memory_gb, flops, data_type)
        return format_hardware_summary(comparison_results)
    
    def get_hardware_comparison_data(self, memory_gb: float, flops: float,
                                   data_type: str = "fp16") -> Dict[str, Any]:
        """
        Get hardware comparison data for web interface
        
        Args:
            memory_gb: Model memory requirement in GB
            flops: Model FLOPS requirement
            data_type: Data type for FLOPS comparison
        
        Returns:
            Hardware comparison data as dictionary
        """
        comparison_results = get_hardware_comparison(memory_gb, flops, data_type)
        return get_hardware_json(comparison_results)
