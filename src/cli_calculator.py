"""
Command Line Interface for Transformer Calculator

This provides a simple CLI for calculating memory and FLOPS for transformer models.
"""

import argparse
import sys
from typing import Optional
try:
    from .transformer_calculator import (
        TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
    )
    from .hf_model_fetcher import create_model_config_from_hf
except ImportError:
    # Fallback for when run as script
    from transformer_calculator import (
        TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType
    )
    from hf_model_fetcher import create_model_config_from_hf


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Calculate HBM memory and GPU FLOPS for transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Analyze Hugging Face models
          python cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --mode prefill --table memory
          python cli_calculator.py --hf-model "https://huggingface.co/openai/gpt-oss-120b" --mode decode --table all
          
          # Use custom sequence length
          python cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --seq-len 4096 --mode prefill --table reuse
          
          # Decode with custom length
          python cli_calculator.py --hf-model "openai/gpt-oss-120b" --decode-len 128 --mode decode --table all
        """
    )
    
    # Model selection
    parser.add_argument("--hf-model", required=True,
                       help="Hugging Face model path (e.g., 'Qwen/Qwen3-VL-235B-A22B-Instruct' or 'https://huggingface.co/openai/gpt-oss-120b')")
    
    # Note: All model architecture parameters are automatically extracted from Hugging Face config.json
    
    # Sequence parameters
    parser.add_argument("--seq-len", type=int,
                       help="Sequence length for prefill/decode modes (defaults to model's max_position_embeddings)")
    parser.add_argument("--hp-seq-len", type=int, default=8192,
                       help="Hyperparameter sequence length for pretraining mode (default: 8192)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--decode-len", type=int, default=64,
                       help="Number of tokens to generate in decode phase")
    
    # Operation parameters
    parser.add_argument("--mode", choices=["pretraining", "prefill", "decode"], 
                       default="prefill", help="Operation mode")
    parser.add_argument("--table", choices=["memory", "flops", "reuse", "all"], 
                       default="all", help="Table type to generate")
    
    # Precision
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16', 'int8', 'int4', 'fp8'],
                       help="Data type for calculations (auto-detected from model if not specified)")
    parser.add_argument("--force-dtype", action="store_true",
                       help="Force use of specified dtype even if model config suggests different")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create ModelConfig from Hugging Face model"""
    print(f"Fetching model configuration from Hugging Face: {args.hf_model}")
    
    # Get both sequence lengths in one download
    model_max_seq_len, model_hp_seq_len = get_model_sequence_info(args.hf_model)
    
    if not model_max_seq_len:
        model_max_seq_len = 2048  # Default fallback
        print(f"Warning: Could not determine model's max sequence length, using default: {model_max_seq_len}")
    
    # Get the model's max sequence length if not specified
    sequence_length = args.seq_len
    if sequence_length is None:
        sequence_length = model_max_seq_len
        print(f"Using model's max sequence length: {sequence_length}")
    else:
        # Validate user-specified sequence length
        if sequence_length > model_max_seq_len:
            print(f"Error: Sequence length ({sequence_length}) exceeds model's maximum supported length ({model_max_seq_len})")
            print(f"Please use --seq-len {model_max_seq_len} or lower")
            sys.exit(1)
        print(f"Using specified sequence length: {sequence_length}")
    
    # Handle hyperparameter sequence length with early validation
    if model_hp_seq_len:
        print(f"Using model's hyperparameter sequence length: {model_hp_seq_len}")
        # Use the extracted hp_seq_len, but allow user override
        hp_seq_len = args.hp_seq_len if args.hp_seq_len != 8192 else model_hp_seq_len
    else:
        print(f"Could not determine model's hyperparameter sequence length, using default: 8192")
        hp_seq_len = args.hp_seq_len  # Use user-specified value or default 8192
    
    # EARLY VALIDATION: Check hyperparameter sequence length immediately
    if hp_seq_len > model_max_seq_len:
        print(f"\n❌ ERROR: Hyperparameter sequence length validation failed!")
        print(f"   Requested HP sequence length: {hp_seq_len}")
        print(f"   Model's maximum supported length: {model_max_seq_len}")
        print(f"   Difference: {hp_seq_len - model_max_seq_len} tokens over limit")
        print(f"\n💡 SOLUTION: Use --hp-seq-len {model_max_seq_len} or lower")
        print(f"   Example: --hp-seq-len {min(model_max_seq_len, 8192)}")
        sys.exit(1)
    
    # Use cached fetcher for better performance
    fetcher = get_global_fetcher()
    hf_info = fetcher.fetch_model_config(args.hf_model)
    
    if not hf_info:
        print(f"Error: Could not fetch model configuration for {args.hf_model}")
        sys.exit(1)
    
    # Use the new from_hf_config method
    config = ModelConfig.from_hf_config(
        hf_info.config,
        model_name=args.hf_model,
        sequence_length=sequence_length,
        batch_size=args.batch_size,
        decode_len=args.decode_len
    )
    
    # Set the hyperparameter sequence length for pretraining
    if config:
        config.hp_seq_len = hp_seq_len
        
        # Handle data type
        if args.dtype:
            # User specified a data type
            user_dtype = DataType.from_string(args.dtype)
            if args.force_dtype:
                # Force use user-specified dtype
                config.dtype = user_dtype
                print(f"Using forced data type: {user_dtype.value} ({user_dtype.bytes} bytes)")
            else:
                # Check if model config suggests different dtype
                from transformer_calculator import DataTypeDetector
                # Get the original HF config for detection
                try:
                    from huggingface_hub import hf_hub_download
                    import json
                    model_id = args.hf_model.replace('https://huggingface.co/', '').replace('@', '')
                    config_path = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
                    with open(config_path, 'r') as f:
                        hf_config = json.load(f)
                    detected_dtype = DataTypeDetector.detect_from_config(hf_config)
                    if detected_dtype != user_dtype:
                        print(f"Model config suggests {detected_dtype.value}, but using user-specified {user_dtype.value}")
                except:
                    print(f"Could not detect model's preferred data type, using user-specified {user_dtype.value}")
                config.dtype = user_dtype
        else:
            # Auto-detect data type from model
            from transformer_calculator import DataTypeDetector
            try:
                from huggingface_hub import hf_hub_download
                import json
                model_id = args.hf_model.replace('https://huggingface.co/', '').replace('@', '')
                config_path = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
                with open(config_path, 'r') as f:
                    hf_config = json.load(f)
                detected_dtype = DataTypeDetector.detect_from_config(hf_config)
                config.dtype = detected_dtype
                print(f"Auto-detected data type: {detected_dtype.value} ({detected_dtype.bytes} bytes)")
            except:
                print("Could not auto-detect data type, using default BF16")
                config.dtype = DataType.BF16
    
    if not config:
        print("Failed to fetch model from Hugging Face.")
        sys.exit(1)
    
    return config


def get_model_max_sequence_length(model_path: str) -> Optional[int]:
    """Get the model's maximum sequence length from config.json"""
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        model_id = model_path.replace('https://huggingface.co/', '').replace('@', '')
        
        # Download config.json
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            repo_type="model"
        )
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for max_position_embeddings in text_config or root
        if 'text_config' in config:
            text_config = config['text_config']
            max_pos = text_config.get('max_position_embeddings')
        else:
            max_pos = config.get('max_position_embeddings')
        
        return max_pos
        
    except Exception as e:
        print(f"Warning: Could not determine max sequence length: {e}")
        return None


def get_model_hp_sequence_length(model_path: str) -> Optional[int]:
    """Get the model's hyperparameter sequence length from config.json"""
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        model_id = model_path.replace('https://huggingface.co/', '').replace('@', '')
        
        # Download config.json
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            repo_type="model"
        )
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for original_max_position_embeddings in rope_scaling (DeepSeek V3 style)
        rope_scaling = config.get('rope_scaling', {})
        if rope_scaling and 'original_max_position_embeddings' in rope_scaling:
            return rope_scaling['original_max_position_embeddings']
        
        # Check for other common hyperparameter sequence length indicators
        hp_seq_len_indicators = [
            'original_max_position_embeddings',
            'training_max_position_embeddings', 
            'pretraining_max_position_embeddings',
            'base_max_position_embeddings'
        ]
        
        for indicator in hp_seq_len_indicators:
            if indicator in config:
                return config[indicator]
        
        # Check in text_config for multimodal models
        text_config = config.get('text_config', {})
        if text_config:
            rope_scaling = text_config.get('rope_scaling', {})
            if rope_scaling and 'original_max_position_embeddings' in rope_scaling:
                return rope_scaling['original_max_position_embeddings']
            
            for indicator in hp_seq_len_indicators:
                if indicator in text_config:
                    return text_config[indicator]
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not determine hyperparameter sequence length: {e}")
        return None


def main():
    """Main CLI function"""
    args = parse_args()
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Create calculator
        calculator = TransformerCalculator(config)
        
        # Determine operation mode
        mode = OperationMode(args.mode)
        
        # Generate tables
        if args.table in ["memory", "all"]:
            print(calculator.generate_table(mode, "memory"))
        
        if args.table in ["flops", "all"]:
            print(calculator.generate_table(mode, "flops"))
        
        if args.table in ["reuse", "all"]:
            print(calculator.generate_table(mode, "reuse"))
        
        # Model parameters are now included in the tables above
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# Global fetcher instance for caching
_global_fetcher = None

def get_global_fetcher():
    """Get or create global HFModelFetcher instance for caching"""
    global _global_fetcher
    if _global_fetcher is None:
        from hf_model_fetcher import HFModelFetcher
        _global_fetcher = HFModelFetcher()
    return _global_fetcher

def get_model_sequence_info(model_path: str) -> tuple[Optional[int], Optional[int]]:
    """Get both max sequence length and hyperparameter sequence length from config.json in one download"""
    try:
        # Use global fetcher for caching
        fetcher = get_global_fetcher()
        hf_info = fetcher.fetch_model_config(model_path)
        
        if not hf_info:
            return None, None
        
        config = hf_info.config
        
        # Check for max_position_embeddings in text_config or root
        if 'text_config' in config:
            text_config = config['text_config']
            max_pos = text_config.get('max_position_embeddings')
            
            # Check for hyperparameter sequence length in rope_scaling
            rope_scaling = text_config.get('rope_scaling', {})
            hp_seq_len = rope_scaling.get('original_max_position_embeddings')
        else:
            max_pos = config.get('max_position_embeddings')
            
            # Check for hyperparameter sequence length in rope_scaling
            rope_scaling = config.get('rope_scaling', {})
            hp_seq_len = rope_scaling.get('original_max_position_embeddings')
        
        return max_pos, hp_seq_len
        
    except Exception as e:
        print(f"Warning: Could not determine sequence lengths: {e}")
        return None, None


if __name__ == "__main__":
    main()
