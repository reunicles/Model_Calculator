"""
Hugging Face Model Analysis Examples

This script demonstrates how to use the transformer calculator with models
fetched directly from Hugging Face Hub.
"""

from transformer_calculator import TransformerCalculator, OperationMode
from hf_model_fetcher import create_model_config_from_hf


def analyze_hf_model(model_path: str, model_name: str):
    """Analyze a Hugging Face model"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {model_name.upper()}")
    print(f"Model Path: {model_path}")
    print(f"{'='*80}")
    
    # Fetch model configuration
    config = create_model_config_from_hf(
        model_path,
        sequence_length=2048,
        batch_size=1,
        decode_len=64
    )
    
    if not config:
        print(f"Failed to fetch model configuration for {model_path}")
        return
    
    # Create calculator
    calculator = TransformerCalculator(config)
    
    # Analyze different operation modes
    modes = [OperationMode.PRETRAINING, OperationMode.PREFILL, OperationMode.DECODE]
    
    for mode in modes:
        print(f"\n{mode.value.upper()} MODE ANALYSIS")
        print("-" * 40)
        
        # Memory analysis
        print(calculator.generate_table(mode, "memory"))
        
        # FLOPS analysis
        print(calculator.generate_table(mode, "flops"))


def compare_models():
    """Compare different models from Hugging Face"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    models = [
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B"),
        ("openai/gpt-oss-120b", "GPT-OSS-120B"),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen3-Omni-30B")
    ]
    
    print(f"{'Model':<30} {'Parameters':<15} {'Memory (MB)':<15} {'FLOPS':<20}")
    print("-" * 85)
    
    for model_path, model_name in models:
        config = create_model_config_from_hf(model_path, sequence_length=1024, batch_size=1)
        
        if config:
            calculator = TransformerCalculator(config)
            
            # Get memory and FLOPS for prefill
            memory = calculator.calculate_memory(OperationMode.PREFILL)
            flops = calculator.calculate_flops(OperationMode.PREFILL)
            
            memory_mb = memory.total / (1024 * 1024)
            flops_str = f"{flops.total:.2e}"
            params = f"{calculator._calculate_total_parameters():,}"
            
            print(f"{model_name:<30} {params:<15} {memory_mb:<15.2f} {flops_str:<20}")
        else:
            print(f"{model_name:<30} {'Failed':<15} {'N/A':<15} {'N/A':<20}")


def analyze_decode_len_impact():
    """Analyze impact of decode length on different models"""
    print(f"\n{'='*80}")
    print("DECODE LENGTH IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    model_path = "openai/gpt-oss-120b"
    decode_lens = [32, 64, 128, 256, 512]
    
    print(f"Model: {model_path}")
    print(f"{'Decode Length':<15} {'Memory (MB)':<15} {'FLOPS':<20}")
    print("-" * 55)
    
    for decode_len in decode_lens:
        config = create_model_config_from_hf(
            model_path,
            sequence_length=1024,
            batch_size=1,
            decode_len=decode_len
        )
        
        if config:
            calculator = TransformerCalculator(config)
            
            # Get memory and FLOPS for decode
            memory = calculator.calculate_memory(OperationMode.DECODE)
            flops = calculator.calculate_flops(OperationMode.DECODE)
            
            memory_mb = memory.total / (1024 * 1024)
            flops_str = f"{flops.total:.2e}"
            
            print(f"{decode_len:<15} {memory_mb:<15.2f} {flops_str:<20}")


def analyze_sequence_length_impact():
    """Analyze impact of sequence length on different models"""
    print(f"\n{'='*80}")
    print("SEQUENCE LENGTH IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    models = [
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B"),
        ("openai/gpt-oss-120b", "GPT-OSS-120B")
    ]
    
    sequence_lengths = [512, 1024, 2048, 4096]
    
    for model_path, model_name in models:
        print(f"\n{model_name}:")
        print(f"{'Seq Length':<12} {'Memory (MB)':<15} {'FLOPS':<20}")
        print("-" * 50)
        
        for seq_len in sequence_lengths:
            config = create_model_config_from_hf(
                model_path,
                sequence_length=seq_len,
                batch_size=1,
                decode_len=64
            )
            
            if config:
                calculator = TransformerCalculator(config)
                
                # Get memory and FLOPS for prefill
                memory = calculator.calculate_memory(OperationMode.PREFILL)
                flops = calculator.calculate_flops(OperationMode.PREFILL)
                
                memory_mb = memory.total / (1024 * 1024)
                flops_str = f"{flops.total:.2e}"
                
                print(f"{seq_len:<12} {memory_mb:<15.2f} {flops_str:<20}")


if __name__ == "__main__":
    print("Hugging Face Model Analysis Examples")
    print("=" * 50)
    
    # Analyze individual models
    analyze_hf_model("Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen3-VL-235B")
    analyze_hf_model("openai/gpt-oss-120b", "GPT-OSS-120B")
    
    # Compare models
    compare_models()
    
    # Analyze decode length impact
    analyze_decode_len_impact()
    
    # Analyze sequence length impact
    analyze_sequence_length_impact()
