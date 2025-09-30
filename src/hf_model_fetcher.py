"""
Hugging Face Model Configuration Fetcher

This module provides functionality to fetch model configurations from Hugging Face Hub
and convert them to our ModelConfig format.
"""

import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
try:
    from .transformer_calculator import ModelConfig, ModelType, DataType, DataTypeDetector
except ImportError:
    # Fallback for when run as script
    from transformer_calculator import ModelConfig, ModelType, DataType, DataTypeDetector


@dataclass
class HFModelInfo:
    """Information about a Hugging Face model"""
    model_id: str
    config: Dict[str, Any]
    model_type: str
    architecture: str
    is_moe: bool = False
    num_experts: int = 1
    top_k: int = 1


class HFModelFetcher:
    """Fetches and parses model configurations from Hugging Face Hub"""
    
    def __init__(self):
        self.cache = {}
        self.config_cache = {}  # Cache for downloaded configs
    
    def parse_model_path(self, model_path: str) -> str:
        """Parse model path from URL or direct path"""
        # Handle URLs like https://huggingface.co/openai/gpt-oss-120b
        if model_path.startswith('https://huggingface.co/'):
            model_path = model_path.replace('https://huggingface.co/', '')
        elif model_path.startswith('@'):
            model_path = model_path[1:]  # Remove @ prefix
        
        return model_path.strip()
    
    def fetch_model_config(self, model_path: str) -> Optional[HFModelInfo]:
        """Fetch model configuration from Hugging Face Hub with caching"""
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
        
        model_id = self.parse_model_path(model_path)
        
        # Check cache first
        if model_id in self.config_cache:
            print(f"Using cached config for {model_id}")
            config = self.config_cache[model_id]
        else:
            try:
                # Download the config.json file directly
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    repo_type="model"
                )
                
                # Load the config.json file
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Cache the config
                self.config_cache[model_id] = config
                print(f"Cached config for {model_id}")
            except Exception as e:
                print(f"Error downloading config for {model_id}: {e}")
                return None
        
        # Determine model architecture
        architecture = self._detect_architecture(config)
        is_moe = self._is_moe_model(config)
        
        # Extract MoE parameters if applicable
        num_experts, top_k = self._extract_moe_params(config)
        
        return HFModelInfo(
            model_id=model_id,
            config=config,
            model_type=architecture,
            architecture=architecture,
            is_moe=is_moe,
            num_experts=num_experts,
            top_k=top_k
        )
    
    def _detect_architecture(self, config: Dict[str, Any]) -> str:
        """Detect the model architecture from config"""
        architectures = config.get('architectures', [])
        if not architectures:
            return 'unknown'
        
        arch = architectures[0].lower()
        
        # Map common architectures
        if 'qwen' in arch:
            return 'qwen'
        elif 'gpt' in arch:
            return 'gpt'
        elif 'llama' in arch:
            return 'llama'
        elif 'mistral' in arch:
            return 'mistral'
        elif 'mixtral' in arch:
            return 'mixtral'
        elif 'gemma' in arch:
            return 'gemma'
        else:
            return arch
    
    def _is_moe_model(self, config: Dict[str, Any]) -> bool:
        """Check if model is MoE (Mixture of Experts)"""
        # Check for MoE indicators with actual values
        moe_indicators = [
            'num_experts',
            'num_local_experts',
            'n_routed_experts',  # DeepSeek V3 parameter
            'expert_capacity',
            'router_aux_loss_coef',
            'moe',
            'mixtral'
        ]
        
        for indicator in moe_indicators:
            if indicator in config:
                # Check if the value is actually > 1 for expert counts
                if indicator in ['num_experts', 'num_local_experts', 'n_routed_experts']:
                    if config[indicator] > 1:
                        return True
                else:
                    return True
        
        # Check architecture name only if we have actual MoE parameters
        architectures = config.get('architectures', [])
        for arch in architectures:
            if any(moe_term in arch.lower() for moe_term in ['moe', 'mixtral', 'switch', 'deepseek']):
                # Only consider it MoE if we have actual expert parameters
                if any(key in config for key in ['num_experts', 'num_local_experts', 'n_routed_experts', 'expert_capacity']):
                    return True
        
        return False
    
    def _extract_moe_params(self, config: Dict[str, Any]) -> Tuple[int, int]:
        """Extract MoE parameters from config"""
        # Handle different MoE parameter naming conventions
        num_experts = config.get('num_experts', 
                               config.get('num_local_experts', 
                                        config.get('n_routed_experts', 1)))
        top_k = config.get('num_experts_per_tok', 
                          config.get('top_k', 1))
        
        return num_experts, top_k
    
    def convert_to_model_config(self, hf_info: HFModelInfo, 
                               sequence_length: int = 2048,
                               batch_size: int = 1,
                               decode_len: int = 64) -> Optional[ModelConfig]:
        """Convert Hugging Face model info to our ModelConfig"""
        if not hf_info:
            return None
        
        config = hf_info.config
        
        # Handle multimodal models (like Qwen3-VL) that have text_config
        if 'text_config' in config:
            print(f"Detected multimodal model with text_config")
            text_config = config['text_config']
            hidden_size = text_config.get('hidden_size', text_config.get('d_model', 4096))
            num_layers = text_config.get('num_hidden_layers', text_config.get('n_layer', 32))
            num_heads = text_config.get('num_attention_heads', text_config.get('n_head', 32))
            vocab_size = text_config.get('vocab_size', 50257)
            num_kv_heads = text_config.get('num_key_value_heads', num_heads)
            intermediate_size = text_config.get('intermediate_size', text_config.get('ffn_dim', 4 * hidden_size))
            
            # Check for MoE in text config
            is_moe = self._is_moe_model(text_config)
            if is_moe:
                num_experts, top_k = self._extract_moe_params(text_config)
                moe_intermediate_size = text_config.get('moe_intermediate_size', intermediate_size)
            else:
                num_experts, top_k = 1, 1
                moe_intermediate_size = None
                
            expert_capacity_factor = text_config.get('expert_capacity_factor', 1.0)
            
            print(f"Using text_config parameters:")
            print(f"  hidden_size: {hidden_size}")
            print(f"  num_layers: {num_layers}")
            print(f"  num_attention_heads: {num_heads}")
            print(f"  vocab_size: {vocab_size}")
            print(f"  intermediate_size: {intermediate_size}")
            if is_moe:
                print(f"  MoE: {num_experts} experts, top-{top_k}")
            
        else:
            # Handle standard transformer models
            hidden_size = config.get('hidden_size', config.get('d_model', 4096))
            num_layers = config.get('num_hidden_layers', config.get('n_layer', 32))
            num_heads = config.get('num_attention_heads', config.get('n_head', 32))
            vocab_size = config.get('vocab_size', 50257)
            num_kv_heads = config.get('num_key_value_heads', num_heads)
            intermediate_size = config.get('intermediate_size', config.get('ffn_dim', 4 * hidden_size))
            
            # Check for MoE
            is_moe = hf_info.is_moe
            if is_moe:
                num_experts, top_k = hf_info.num_experts, hf_info.top_k
                moe_intermediate_size = config.get('moe_intermediate_size', intermediate_size)
            else:
                num_experts, top_k = 1, 1
                moe_intermediate_size = None
                
            expert_capacity_factor = config.get('expert_capacity_factor', 1.0)
        
        # Determine model type
        model_type = ModelType.MOE if is_moe else ModelType.DENSE
        
        # Check for missing parameters and show defaults
        missing_params = []
        defaults_used = {}
        
        # Check for missing basic parameters
        if not any(key in (config.get('text_config', config) if 'text_config' in config else config) for key in ['hidden_size', 'd_model']):
            missing_params.append('hidden_size')
            defaults_used['hidden_size'] = hidden_size
        if not any(key in (config.get('text_config', config) if 'text_config' in config else config) for key in ['num_hidden_layers', 'n_layer']):
            missing_params.append('num_layers')
            defaults_used['num_layers'] = num_layers
        if not any(key in (config.get('text_config', config) if 'text_config' in config else config) for key in ['num_attention_heads', 'n_head']):
            missing_params.append('num_attention_heads')
            defaults_used['num_attention_heads'] = num_heads
        if not any(key in (config.get('text_config', config) if 'text_config' in config else config) for key in ['vocab_size']):
            missing_params.append('vocab_size')
            defaults_used['vocab_size'] = vocab_size
        if not any(key in (config.get('text_config', config) if 'text_config' in config else config) for key in ['intermediate_size', 'ffn_dim']):
            missing_params.append('intermediate_size')
            defaults_used['intermediate_size'] = intermediate_size
        
        # Special handling for models with incomplete configs
        if missing_params:
            print(f"\n⚠️  WARNING: Missing parameters for {hf_info.model_id}")
            print("Missing parameters:", ", ".join(missing_params))
            print("Using intelligent defaults based on model name:")
            for param, value in defaults_used.items():
                print(f"  {param}: {value}")
            print("Consider providing model parameters manually for more accurate calculations.")
            
            # Use model name to infer some parameters
            model_name = hf_info.model_id.lower()
            if '235b' in model_name or '235' in model_name:
                hidden_size = 5120  # Typical for 235B models
                num_layers = 64
                num_heads = 40
                print(f"  Inferred from model name: hidden_size=5120, num_layers=64, num_heads=40")
            elif '120b' in model_name or '120' in model_name:
                hidden_size = 4096  # Typical for 120B models
                num_layers = 32
                num_heads = 32
                print(f"  Inferred from model name: hidden_size=4096, num_layers=32, num_heads=32")
            elif '30b' in model_name or '30' in model_name:
                hidden_size = 4096  # Typical for 30B models
                num_layers = 32
                num_heads = 32
                print(f"  Inferred from model name: hidden_size=4096, num_layers=32, num_heads=32")
        
        # Detect data type from config and model name
        detector = DataTypeDetector()
        detected_dtype = detector.detect_from_config(config)
        
        # Create ModelConfig with detected data type
        model_config = ModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            model_type=model_type,
            num_experts=num_experts,
            expert_capacity_factor=expert_capacity_factor,
            top_k=top_k,
            moe_intermediate_size=moe_intermediate_size,
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len,
            dtype=detected_dtype  # Use detected data type
        )
        
        # Print data type detection info
        print(f"Data type detected: {detected_dtype.value} ({model_config.dtype_bytes} bytes)")
        
        return model_config


def create_model_config_from_hf(model_path: str, 
                               sequence_length: int = 2048,
                               batch_size: int = 1,
                               decode_len: int = 64) -> Optional[ModelConfig]:
    """Convenience function to create ModelConfig from Hugging Face model path"""
    fetcher = HFModelFetcher()
    hf_info = fetcher.fetch_model_config(model_path)
    
    if not hf_info:
        return None
    
    return fetcher.convert_to_model_config(hf_info, sequence_length, batch_size, decode_len)


# Predefined model configurations for known models
PREDEFINED_MODELS = {
    'qwen3-omni': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
    'qwen3-vl-235b': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
    'gpt-oss-20b': 'openai/gpt-oss-20b',
}


def get_predefined_model_path(model_name: str) -> Optional[str]:
    """Get Hugging Face path for predefined model names"""
    return PREDEFINED_MODELS.get(model_name.lower())


if __name__ == "__main__":
    # Test the fetcher
    test_models = [
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "openai/gpt-oss-120b"
    ]
    
    fetcher = HFModelFetcher()
    
    for model_path in test_models:
        print(f"\nFetching {model_path}...")
        hf_info = fetcher.fetch_model_config(model_path)
        
        if hf_info:
            print(f"Model ID: {hf_info.model_id}")
            print(f"Architecture: {hf_info.architecture}")
            print(f"Is MoE: {hf_info.is_moe}")
            if hf_info.is_moe:
                print(f"Experts: {hf_info.num_experts}, Top-K: {hf_info.top_k}")
            
            # Convert to our config
            config = fetcher.convert_to_model_config(hf_info)
            if config:
                print(f"Converted config: {config.num_layers} layers, {config.hidden_size} hidden size")
        else:
            print("Failed to fetch model info")
