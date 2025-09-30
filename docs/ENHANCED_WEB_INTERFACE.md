# Enhanced Web Interface for Transformer Calculator

## Overview

The Enhanced Web Interface provides a unified experience for calculating transformer model memory and FLOPS requirements, with seamless HuggingFace model integration.

## Key Features

### 🤗 HuggingFace Integration
- **Direct Model Loading**: Enter any HuggingFace model path (e.g., `microsoft/DialoGPT-medium`)
- **Auto-Population**: Automatically fills in model parameters from `config.json`
- **Unified Calculations**: Uses the same calculation logic as the CLI tool
- **Real-time Updates**: See results immediately after loading a model

### 🎯 Enhanced User Experience
- **Tabbed Interface**: Separate sections for manual configuration and advanced parameters
- **Smart Defaults**: Sensible default values for all parameters
- **MoE Support**: Full support for Mixture of Experts models
- **Responsive Design**: Works on desktop and mobile devices

### 🔧 Advanced Features
- **Multiple Operation Modes**: Pretraining, Prefill, and Decode
- **Benchmark Suite**: Run comprehensive benchmarks
- **Real-time Calculations**: Instant results as you modify parameters
- **Export Results**: Copy results for further analysis

## Usage

### Quick Start
1. **Start the server**:
   ```bash
   python3 start_enhanced_web.py
   ```

2. **Access the interface**:
   - Open your browser to `http://localhost:5001`
   - The interface will automatically find an available port

### HuggingFace Model Loading
1. **Enter Model Path**: In the HuggingFace section, enter a model path like:
   - `microsoft/DialoGPT-medium`
   - `meta-llama/Llama-2-7b-hf`
   - `Qwen/Qwen2.5-7B`

2. **Click "Load Model"**: The interface will:
   - Fetch the model configuration
   - Auto-populate all form fields
   - Display model information
   - Automatically run calculations

3. **View Results**: See comprehensive results including:
   - Memory usage breakdown
   - FLOPS requirements
   - Arithmetic intensity analysis

### Manual Configuration
1. **Basic Parameters**: Set core model parameters
2. **Advanced Parameters**: Configure sequence length, batch size, operation mode
3. **MoE Parameters**: For Mixture of Experts models
4. **Calculate**: Get instant results

## API Endpoints

### `/api/calculate`
Calculate memory and FLOPS for manually configured models.

**Request**:
```json
{
  "model_type": "dense",
  "num_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  "vocab_size": 32000,
  "dtype": "bf16",
  "sequence_length": 2048,
  "batch_size": 1,
  "mode": "prefill",
  "decode_len": 64,
  "num_experts": 8,
  "top_k": 2,
  "expert_capacity_factor": 1.0
}
```

**Response**:
```json
{
  "success": true,
  "results": {
    "memory": {
      "attention": 1234567890,
      "projections": 987654321,
      "mlp": 4567890123,
      "embeddings": 234567890,
      "layer_norm": 12345678,
      "activations": 345678901,
      "weights": 12345678901,
      "kv_cache": 0,
      "gradients": 12345678901,
      "optimizer_states": 24691357802,
      "total": 49382715604
    },
    "flops": {
      "attention": 1234567890123,
      "projections": 987654321098,
      "mlp": 4567890123456,
      "embeddings": 234567890123,
      "layer_norm": 12345678901,
      "total": 7000000000000
    },
    "reuse": {
      "pre_attention": 123.45,
      "attention": 789.01,
      "post_attention": 234.56,
      "total": 1147.02
    },
    "config": {
      "model_type": "dense",
      "num_layers": 32,
      "hidden_size": 4096,
      "dtype": "bf16",
      "sequence_length": 2048,
      "batch_size": 1
    }
  }
}
```

### `/api/hf-model`
Fetch and configure a HuggingFace model.

**Request**:
```json
{
  "hf_path": "microsoft/DialoGPT-medium",
  "sequence_length": 2048,
  "batch_size": 1,
  "decode_len": 64
}
```

**Response**:
```json
{
  "success": true,
  "config": {
    "num_layers": 24,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "vocab_size": 50257,
    "model_type": "dense",
    "dtype": "fp32",
    "num_experts": 1,
    "top_k": 1,
    "expert_capacity_factor": 1.0,
    "hf_path": "microsoft/DialoGPT-medium"
  }
}
```

### `/api/calculate-hf`
Direct calculation for HuggingFace models.

**Request**:
```json
{
  "hf_path": "microsoft/DialoGPT-medium",
  "sequence_length": 2048,
  "batch_size": 1,
  "decode_len": 64,
  "mode": "prefill"
}
```

### `/api/benchmark`
Run comprehensive benchmarks.

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "model_name": "Small Dense",
      "accuracy_score": 0.95,
      "execution_time": 0.123,
      "memory_gb": 1.23,
      "flops_tflop": 4.56,
      "reuse": 789.01
    }
  ]
}
```

## Supported Models

### Dense Models
- **GPT-style**: `microsoft/DialoGPT-medium`, `gpt2`
- **LLaMA**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- **Qwen**: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-72B`
- **Custom**: Any HuggingFace model with standard transformer architecture

### MoE Models
- **Mixtral**: `mistralai/Mixtral-8x7B-v0.1`
- **Qwen MoE**: `Qwen/Qwen2.5-MoE-A14B`
- **Custom MoE**: Any HuggingFace MoE model

## Configuration Options

### Model Parameters
- **Model Type**: Dense or MoE (Mixture of Experts)
- **Architecture**: Number of layers, hidden size, attention heads
- **Vocabulary**: Vocabulary size and data type
- **MoE**: Number of experts, top-k selection, capacity factor

### Operation Parameters
- **Sequence Length**: Input sequence length
- **Batch Size**: Number of sequences processed simultaneously
- **Operation Mode**: Pretraining, Prefill, or Decode
- **Decode Length**: Number of tokens to generate (decode mode)

### Optimization Parameters
- **Data Type**: FP32, FP16, BF16, INT8, INT4
- **Flash Attention**: Automatic optimization for large sequences
- **Flash Decode**: Automatic optimization for inference

## Results Interpretation

### Memory Usage
- **Attention**: Memory for attention mechanism
- **Projections**: Q, K, V, and output projection weights
- **MLP**: Multi-layer perceptron weights
- **Embeddings**: Input and output embedding weights
- **Layer Norm**: Layer normalization weights
- **Weights**: Total model weights
- **KV Cache**: Key-value cache for inference

### FLOPS Requirements
- **Attention**: Attention mechanism computation
- **Projections**: Projection layer computation
- **MLP**: MLP layer computation
- **Embeddings**: Embedding layer computation
- **Layer Norm**: Layer normalization computation

### Arithmetic Intensity
- **Pre-Attention**: Embeddings, layer norm, projections
- **Attention**: Attention mechanism efficiency
- **Post-Attention**: MLP and final layer norm
- **Efficiency**: High (>100), Medium (10-100), Low (<10)

## Troubleshooting

### Common Issues
1. **Port Already in Use**: The interface automatically finds available ports
2. **Model Loading Failed**: Check HuggingFace model path and internet connection
3. **Calculation Errors**: Verify model parameters are reasonable

### Debug Mode
```bash
# Run with debug output
FLASK_DEBUG=1 python3 web_interface_enhanced.py
```

### Logs
Check the console output for detailed error messages and debugging information.

## Development

### Adding New Features
1. **Backend**: Modify `web_interface_enhanced.py`
2. **Frontend**: Update `templates/enhanced_index.html`
3. **API**: Add new endpoints following the existing pattern

### Testing
```bash
# Run tests
python3 -m pytest tests/

# Run specific test
python3 -m pytest tests/test_web_interface.py
```

## Performance

### Optimization
- **Caching**: Model configurations are cached for faster loading
- **Async**: Non-blocking calculations for better responsiveness
- **Compression**: Results are compressed for faster transmission

### Scalability
- **Concurrent Users**: Supports multiple simultaneous users
- **Memory Usage**: Efficient memory management for large models
- **Response Time**: Sub-second response times for most calculations

## Security

### Best Practices
- **Input Validation**: All inputs are validated and sanitized
- **Error Handling**: Graceful error handling prevents crashes
- **Rate Limiting**: Built-in rate limiting for API endpoints

### Recommendations
- **Production**: Use a production WSGI server (Gunicorn, uWSGI)
- **HTTPS**: Enable HTTPS for production deployments
- **Authentication**: Add authentication for sensitive deployments

## Contributing

### Code Style
- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use modern ES6+ features
- **HTML/CSS**: Use semantic HTML and modern CSS

### Testing
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test API endpoints
- **UI Tests**: Test user interface interactions

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues and questions:
1. **GitHub Issues**: Create an issue on the project repository
2. **Documentation**: Check the comprehensive documentation
3. **Community**: Join the discussion forums

---

**Happy Calculating! 🚀**

