"""
Enhanced Web Interface for Transformer Calculator

Integrates HuggingFace model fetching with calculation logic for a unified experience.
Users can either:
1. Enter a HuggingFace model path and auto-populate parameters
2. Manually configure model parameters
3. Mix both approaches (start with HF model, then modify parameters)
"""

try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    print("❌ Flask not installed. Install with: pip3 install flask")
    print("   The web interface requires Flask to run.")
    print("   You can still use the CLI calculator without Flask.")
    FLASK_AVAILABLE = False
    sys.exit(1)

import json
import os
import sys
from typing import Dict, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .transformer_calculator import (
        TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
        DataTypeDetector, calculate_model_parameters
    )
    from .hf_model_fetcher import create_model_config_from_hf
except ImportError:
    # Fallback for when run as script
    from transformer_calculator import (
        TransformerCalculator, ModelConfig, ModelType, OperationMode, DataType,
        DataTypeDetector, calculate_model_parameters
    )
    from hf_model_fetcher import create_model_config_from_hf


app = Flask(__name__)

# Force template reloading to avoid caching issues
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    """Main calculator interface"""
    return render_template('enhanced_v2_index.html')


@app.route('/api/hf-models', methods=['GET'])
def get_hf_models():
    """Get list of popular HuggingFace models"""
    popular_models = [
        {
            'name': 'Qwen3-VL-235B-Thinking',
            'path': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
            'type': 'MoE',
            'size': '235B',
            'description': 'Qwen3-VL 235B parameter vision-language model with thinking capabilities'
        },
        {
            'name': 'DeepSeek-V3.1-Terminus',
            'path': 'deepseek-ai/DeepSeek-V3.1-Terminus',
            'type': 'MoE',
            'size': '685B',
            'description': 'DeepSeek V3.1 Terminus 685B parameter model with enhanced reasoning'
        },
        {
            'name': 'GPT-OSS-120B',
            'path': 'openai/gpt-oss-120b',
            'type': 'MoE',
            'size': '120B',
            'description': 'OpenAI GPT-OSS 120B parameter model for powerful reasoning and agentic tasks'
        },
        {
            'name': 'Falcon-H1-34B',
            'path': 'tiiuae/Falcon-H1-34B-Instruct',
            'type': 'Dense',
            'size': '34B',
            'description': 'Falcon-H1 34B parameter hybrid Transformer+Mamba model for efficient reasoning'
        },
        {
            'name': 'Qwen2.5-72B',
            'path': 'Qwen/Qwen2.5-72B',
            'type': 'Dense',
            'size': '72B',
            'description': 'Qwen 2.5 72B parameter dense model (publicly available)'
        },
        {
            'name': 'Qwen2.5-14B',
            'path': 'Qwen/Qwen2.5-14B',
            'type': 'Dense',
            'size': '14B',
            'description': 'Qwen 2.5 14B parameter dense model (publicly available)'
        },
        {
            'name': 'Qwen2.5-7B',
            'path': 'Qwen/Qwen2.5-7B',
            'type': 'Dense',
            'size': '7B',
            'description': 'Qwen 2.5 7B parameter dense model (publicly available)'
        },
        {
            'name': 'Phi-3.5-mini',
            'path': 'microsoft/Phi-3.5-mini-instruct',
            'type': 'Dense',
            'size': '3.8B',
            'description': 'Microsoft Phi-3.5 mini model (publicly available)'
        }
    ]
    
    return jsonify({'success': True, 'models': popular_models})


@app.route('/api/fetch-model', methods=['POST'])
def fetch_model():
    """Fetch HuggingFace model configuration"""
    try:
        data = request.get_json()
        hf_path = data.get('hf_path', '').strip()
        sequence_length = int(data.get('sequence_length', 2048))
        batch_size = int(data.get('batch_size', 1))
        decode_len = int(data.get('decode_len', 64))
        
        if not hf_path:
            return jsonify({'success': False, 'error': 'No HuggingFace model path provided'}), 400
        
        # Fetch model configuration using existing fetcher
        config = create_model_config_from_hf(
            hf_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len
        )
        
        if not config:
            return jsonify({'success': False, 'error': 'Failed to fetch model configuration'}), 400
        
        # Calculate model parameters
        total_params = calculate_model_parameters(config)
        
        # Format model information
        model_info = {
            'hf_path': hf_path,
            'model_type': config.model_type.value,
            'num_layers': config.num_layers,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': config.num_key_value_heads,
            'intermediate_size': config.intermediate_size,
            'vocab_size': config.vocab_size,
            'dtype': config.dtype.value,
            'dtype_bytes': config.dtype_bytes,
            'total_parameters': total_params,
            'total_parameters_formatted': _format_parameters_human(total_params),
            'sequence_length': config.sequence_length,
            'batch_size': config.batch_size,
            'decode_len': config.decode_len
        }
        
        # Add MoE parameters if applicable
        if config.model_type == ModelType.MOE:
            model_info.update({
                'num_experts': config.num_experts,
                'top_k': config.top_k,
                'expert_capacity_factor': config.expert_capacity_factor,
                'moe_intermediate_size': config.moe_intermediate_size
            })
        
        return jsonify({'success': True, 'model_info': model_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """API endpoint for calculations"""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_type = data.get('model_type', 'dense')
        num_layers = int(data.get('num_layers', 32))
        hidden_size = int(data.get('hidden_size', 4096))
        num_attention_heads = int(data.get('num_attention_heads', 32))
        intermediate_size = int(data.get('intermediate_size', 11008))
        vocab_size = int(data.get('vocab_size', 32000))
        dtype = data.get('dtype', 'bf16')
        sequence_length = int(data.get('sequence_length', 2048))
        batch_size = int(data.get('batch_size', 1))
        decode_len = int(data.get('decode_len', 64))
        
        # MoE parameters
        num_experts = int(data.get('num_experts', 8))
        top_k = int(data.get('top_k', 2))
        expert_capacity_factor = float(data.get('expert_capacity_factor', 1.0))
        
        # Operation mode
        mode = data.get('mode', 'prefill')
        
        # Create model configuration
        config = ModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            model_type=ModelType(model_type),
            dtype=DataType(dtype),
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        # Create calculator
        calculator = TransformerCalculator(config)
        
        # Run calculations
        operation_mode = OperationMode(mode)
        memory = calculator.calculate_memory(operation_mode)
        flops = calculator.calculate_flops(operation_mode)
        reuse = calculator.calculate_reuse(operation_mode)
        
        # Calculate total parameters
        total_params = calculate_model_parameters(config)
        
        # Format results
        results = {
            'model_info': {
                'model_type': config.model_type.value,
                'total_parameters': total_params,
                'total_parameters_formatted': _format_parameters_human(total_params),
                'num_layers': config.num_layers,
                'hidden_size': config.hidden_size,
                'num_attention_heads': config.num_attention_heads,
                'num_key_value_heads': config.num_key_value_heads,
                'intermediate_size': config.intermediate_size,
                'vocab_size': config.vocab_size,
                'dtype': config.dtype.value,
                'dtype_bytes': config.dtype_bytes,
                'sequence_length': config.sequence_length,
                'batch_size': config.batch_size,
                'mode': mode
            },
            'memory': {
                'attention': memory.attention,
                'projections': memory.projections,
                'mlp': memory.mlp,
                'embeddings': memory.embeddings,
                'layer_norm': memory.layer_norm,
                'activations': memory.activations,
                'weights': memory.weights,
                'kv_cache': memory.kv_cache,
                'gradients': memory.gradients,
                'optimizer_states': memory.optimizer_states,
                'total': memory.total
            },
            'flops': {
                'attention': flops.attention,
                'projections': flops.projections,
                'mlp': flops.mlp,
                'embeddings': flops.embeddings,
                'layer_norm': flops.layer_norm,
                'total': flops.total
            },
            'reuse': {
                'pre_attention': reuse.pre_attention,
                'attention': reuse.attention,
                'post_attention': reuse.post_attention,
                'total': reuse.total
            }
        }
        
        # Add MoE info if applicable
        if config.model_type == ModelType.MOE:
            results['model_info'].update({
                'num_experts': config.num_experts,
                'top_k': config.top_k,
                'expert_capacity_factor': config.expert_capacity_factor,
                'moe_intermediate_size': config.moe_intermediate_size
            })
        
        # Add hardware comparison
        try:
            from .table_formatter import TableFormatter
        except ImportError:
            # Fallback for when run as script
            from table_formatter import TableFormatter
        formatter = TableFormatter(config)
        hardware_comparison = formatter.get_hardware_comparison_data(
            memory.total / (1024**3),  # Convert to GB
            flops.total,
            config.dtype.value
        )
        results['hardware_comparison'] = hardware_comparison
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/hf-model', methods=['POST'])
def fetch_hf_model():
    """API endpoint for fetching Hugging Face models"""
    try:
        data = request.get_json()
        hf_path = data.get('hf_path', '')
        sequence_length = int(data.get('sequence_length', 2048))
        batch_size = int(data.get('batch_size', 1))
        decode_len = int(data.get('decode_len', 64))
        
        if not hf_path:
            return jsonify({'success': False, 'error': 'No Hugging Face path provided'}), 400
        
        # Fetch model configuration
        config = create_model_config_from_hf(
            hf_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len
        )
        
        if not config:
            return jsonify({'success': False, 'error': 'Failed to fetch model configuration'}), 400
        
        # Return configuration
        config_data = {
            'num_layers': config.num_layers,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'intermediate_size': config.intermediate_size,
            'vocab_size': config.vocab_size,
            'model_type': config.model_type.value,
            'dtype': config.dtype.value,
            'num_experts': getattr(config, 'num_experts', 1),
            'top_k': getattr(config, 'top_k', 1),
            'expert_capacity_factor': getattr(config, 'expert_capacity_factor', 1.0),
            'hf_path': hf_path
        }
        
        return jsonify({'success': True, 'config': config_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/calculate-hf', methods=['POST'])
def calculate_hf_model():
    """API endpoint for direct HuggingFace model calculations"""
    try:
        data = request.get_json()
        hf_path = data.get('hf_path', '')
        sequence_length = int(data.get('sequence_length', 2048))
        batch_size = int(data.get('batch_size', 1))
        decode_len = int(data.get('decode_len', 64))
        mode = data.get('mode', 'prefill')
        
        if not hf_path:
            return jsonify({'success': False, 'error': 'No Hugging Face path provided'}), 400
        
        # Fetch model configuration
        config = create_model_config_from_hf(
            hf_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            decode_len=decode_len
        )
        
        if not config:
            return jsonify({'success': False, 'error': 'Failed to fetch model configuration'}), 400
        
        # Create calculator
        calculator = TransformerCalculator(config)
        
        # Run calculations
        operation_mode = OperationMode(mode)
        memory = calculator.calculate_memory(operation_mode)
        flops = calculator.calculate_flops(operation_mode)
        reuse = calculator.calculate_reuse(operation_mode)
        
        # Format results
        results = {
            'memory': {
                'attention': memory.attention,
                'projections': memory.projections,
                'mlp': memory.mlp,
                'embeddings': memory.embeddings,
                'layer_norm': memory.layer_norm,
                'activations': memory.activations,
                'weights': memory.weights,
                'kv_cache': memory.kv_cache,
                'gradients': memory.gradients,
                'optimizer_states': memory.optimizer_states,
                'total': memory.total
            },
            'flops': {
                'attention': flops.attention,
                'projections': flops.projections,
                'mlp': flops.mlp,
                'embeddings': flops.embeddings,
                'layer_norm': flops.layer_norm,
                'total': flops.total
            },
            'reuse': {
                'pre_attention': reuse.pre_attention,
                'attention': reuse.attention,
                'post_attention': reuse.post_attention,
                'total': reuse.total
            },
            'config': {
                'model_type': config.model_type.value,
                'num_layers': config.num_layers,
                'hidden_size': config.hidden_size,
                'dtype': config.dtype.value,
                'sequence_length': config.sequence_length,
                'batch_size': config.batch_size,
                'hf_path': hf_path
            }
        }
        
        # Add hardware comparison
        try:
            from .table_formatter import TableFormatter
        except ImportError:
            # Fallback for when run as script
            from table_formatter import TableFormatter
        formatter = TableFormatter(config)
        hardware_comparison = formatter.get_hardware_comparison_data(
            memory.total / (1024**3),  # Convert to GB
            flops.total,
            config.dtype.value
        )
        results['hardware_comparison'] = hardware_comparison
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """API endpoint for running benchmarks"""
    try:
        # Import benchmark suite
        from benchmark_suite import BenchmarkSuite
        
        suite = BenchmarkSuite()
        suite.create_standard_validation_cases()
        results = suite.run_all_validations()
        
        # Format results
        benchmark_results = []
        for result in results:
            benchmark_results.append({
                'model_name': result.model_name,
                'accuracy_score': result.accuracy_score,
                'execution_time': result.execution_time,
                'memory_gb': result.memory.total / 1e9,
                'flops_tflop': result.flops.total / 1e12,
                'reuse': result.reuse.total
            })
        
        return jsonify({'success': True, 'results': benchmark_results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


def _format_parameters_human(param_count: int) -> str:
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
        return f"{param_count / 1000000000000000:.1f}P"


def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🌐 Starting Enhanced Transformer Calculator Web Interface")
    print("=" * 60)
    
    # Find available port (avoiding port 5000 which is used by macOS AirPlay)
    port = find_available_port(5001)  # Start from 5001 to avoid AirPlay
    if port is None:
        print("❌ No available ports found. Please close some applications and try again.")
        sys.exit(1)
    
    print(f"Access the interface at: http://localhost:{port}")
    print(f"API endpoints available at: http://localhost:{port}/api/")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use.")
            print("💡 Try running: python3 start_web_interface.py")
            print("   This will automatically find an available port.")
        else:
            print(f"❌ Error starting web interface: {e}")
        sys.exit(1)


def main():
    """Main entry point for the web interface"""
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🌐 Starting Enhanced Transformer Calculator Web Interface")
    print("=" * 60)
    
    # Find available port (avoiding port 5000 which is used by macOS AirPlay)
    port = find_available_port(5001)  # Start from 5001 to avoid AirPlay
    if port is None:
        print("❌ No available ports found. Please close some applications and try again.")
        sys.exit(1)
    
    print(f"Access the interface at: http://localhost:{port}")
    print(f"API endpoints available at: http://localhost:{port}/api/")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use.")
            print("💡 Try running: python3 start_web_interface.py")
            print("   This will automatically find an available port.")
        else:
            print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

