"""
Test suite for the Model Calculator project.

This suite focuses on validating correct behaviour of the refactored
modules. It covers edge cases and ensures that bugs do not creep into
core functionality. To run these tests execute:

    python3 test_model_calculator.py

Make sure the project root is on the Python path when running the
tests (e.g. by setting `PYTHONPATH=.`).
"""

import unittest

from transformer_calculator import (
    ModelConfig,
    ModelType,
    DataType,
    TransformerCalculator,
    OperationMode,
    ValidationRules,
    ConfigBuilder,
    MoECalculator,
)
from datatypes import DataType as DT
from table_formatter import TableFormatter
from context import MemoryContext, CalculationContext
from moe_optimizations import MoECapacityManager, RoutingResult


def make_sample_config(model_type=ModelType.DENSE) -> ModelConfig:
    """Helper to create a minimal model configuration for testing."""
    return ModelConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        vocab_size=100,
        model_type=model_type,
        num_experts=4 if model_type == ModelType.MOE else 1,
        expert_capacity_factor=1.0,
        top_k=2 if model_type == ModelType.MOE else 1,
        sequence_length=16,
        batch_size=1,
        decode_len=8,
        dtype=DT.BF16,
    )


class TestModelCalculator(unittest.TestCase):
    """Test suite for the Model Calculator project."""
    
    def test_config_builder_rejects_dtype_bytes_override(self):
        """ConfigBuilder.from_base_config should not accept dtype_bytes overrides."""
        base = make_sample_config()
        # Attempting to override dtype_bytes should raise a TypeError because
        # ModelConfig does not accept a dtype_bytes argument.
        with self.assertRaises(TypeError):
            ConfigBuilder.from_base_config(base_config=base, dtype_bytes=4)


    def test_parameter_formatting_consistency(self):
        """Ensure that parameter formatting uses consistent thresholds."""
        cfg = make_sample_config()
        # Add a simple Model with 2 layers etc. Parameter count is small (<1M)
        total_params = TransformerCalculator(cfg)._calculate_total_parameters()
        # TableFormatter formats small values in K units
        tf = TableFormatter(cfg)
        human_tf = tf._format_parameters_human(total_params)
        # TransformerCalculator formats small values with comma separator
        human_calc = TransformerCalculator(cfg)._format_parameters_human(total_params)
        # When total parameters < 1_000_000, TableFormatter uses K and TransformerCalculator returns comma separated.
        # They should not both claim units like M or B.
        self.assertFalse(human_tf.endswith("M") or human_tf.endswith("B"), "TableFormatter wrongly labels small counts")
        self.assertTrue(',' in human_calc or human_calc.isdigit(), "TransformerCalculator should format small counts directly")


    def test_moe_calculator_decode_uses_seq_len_one(self):
        """Decode memory for MoE should not scale with full sequence length."""
        cfg = make_sample_config(model_type=ModelType.MOE)
        moe_calc = MoECalculator(cfg)
        # Create two contexts with different seq_len values
        ctx_small = MemoryContext(
            seq_len=16,
            batch_size=1,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            dtype_bytes=cfg.dtype_bytes,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            mode='decode',
            model_type=cfg.model_type.value,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            expert_capacity_factor=cfg.expert_capacity_factor,
            moe_intermediate_size=cfg.moe_intermediate_size,
        )
        ctx_large = MemoryContext(
            seq_len=64,
            batch_size=1,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            dtype_bytes=cfg.dtype_bytes,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            mode='decode',
            model_type=cfg.model_type.value,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            expert_capacity_factor=cfg.expert_capacity_factor,
            moe_intermediate_size=cfg.moe_intermediate_size,
        )
        mem_small = moe_calc.calculate_hbm_memory(ctx_small)
        mem_large = moe_calc.calculate_hbm_memory(ctx_large)
        # Because decode uses seq_len=1 internally, memory should be identical for both contexts
        self.assertAlmostEqual(mem_small, mem_large, places=10)


    def test_moe_router_update_expert_metrics_handles_nested_experts(self):
        """Update should flatten nested selected_experts lists or raise a TypeError."""
        cfg = make_sample_config(model_type=ModelType.MOE)
        manager = MoECapacityManager(num_experts=cfg.num_experts)
        # selected_experts with nested lists as produced by top_k routing
        nested_selected = [[0, 1], [2, 3]]
        routing_result = RoutingResult(
            selected_experts=nested_selected,
            routing_weights=[0.5, 0.5],
            load_balance_loss=0.0,
            communication_cost=0.0,
            routing_flops=0.0,
            routing_memory=0.0,
        )
        # update_expert_metrics should raise TypeError when comparing list < int
        with self.assertRaises(TypeError):
            manager.update_expert_metrics(routing_result, CalculationContext(
                seq_len=16,
                batch_size=1,
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                intermediate_size=cfg.intermediate_size,
                dtype_bytes=cfg.dtype_bytes,
                num_layers=cfg.num_layers,
                vocab_size=cfg.vocab_size,
                model_type=cfg.model_type.value,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                expert_capacity_factor=cfg.expert_capacity_factor,
                moe_intermediate_size=cfg.moe_intermediate_size,
            ))


    def test_validation_rules_detect_invalid_attention_config(self):
        """ValidationRules should catch when num_kv_heads > num_attention_heads."""
        cfg = make_sample_config()
        # Set invalid kv heads
        cfg.num_key_value_heads = cfg.num_attention_heads + 1
        errors = ValidationRules.validate_attention_config(cfg)
        self.assertTrue(any("cannot exceed" in err for err in errors))

    # New tests for additional errors discovered in the latest code
    def test_decode_flops_should_scale_with_decode_len(self):
        """
        In decode mode, total FLOPS should scale with the decode length (decode_len).

        A model with a longer decode_len should incur proportionally more FLOPS than
        one generating only a single token. Due to a bug in DecodeStrategy, the
        calculated FLOPS are currently scaled by seq_len (which is fixed at 1 for
        decode contexts) rather than decode_len. This test will fail until the bug
        is fixed.
        """
        # Config with decode_len=1
        cfg_small = make_sample_config()
        cfg_small.decode_len = 1
        calc_small = TransformerCalculator(cfg_small)
        flops_small = calc_small.calculate_flops(OperationMode.DECODE)
        # Config with larger decode_len
        cfg_large = make_sample_config()
        cfg_large.decode_len = 10
        calc_large = TransformerCalculator(cfg_large)
        flops_large = calc_large.calculate_flops(OperationMode.DECODE)
        # Under correct implementation, total FLOPS should increase with decode_len
        # Here we assert that large decode length yields strictly more FLOPS
        self.assertGreater(flops_large.total, flops_small.total, 
                          "Decode FLOPS did not scale with decode_len; this indicates a bug in DecodeStrategy.calculate_flops")


if __name__ == '__main__':
    unittest.main()
