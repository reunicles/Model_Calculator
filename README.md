# Transformer Memory & FLOPS Calculator

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Integrated-orange.svg)](https://huggingface.co)
[![MoE](https://img.shields.io/badge/MoE-Supported-purple.svg)](https://github.com/reunicles/Model_Calculator)

**Comprehensive analysis of transformer model memory usage and compute requirements**

[Features](#-features) • [Quick Start](#-quick-start) • [Web Interface](#-web-interface) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 🚀 Features

### Core Capabilities
- **🔍 Multi-Architecture Support**: Dense and MoE (Mixture of Experts) models
- **⚡ Operation Modes**: Pretraining, Prefill, and Decode phases
- **📊 Component Analysis**: Detailed breakdown of attention, MLP, embeddings, and more
- **🧠 Memory Optimization**: Flash Attention and Flash Decode support
- **🤗 HuggingFace Integration**: Automatic model configuration fetching

### Advanced Features
- **🎯 Web Interface**: Interactive analysis with popular model selection
- **📈 Performance Analysis**: Arithmetic intensity and efficiency metrics
- **🔧 Data Type Support**: FP32, FP16, BF16, INT8, INT4 with auto-detection
- **🧪 Comprehensive Testing**: 20+ unit tests and real-world validation
- **📚 Professional Documentation**: Detailed mathematical foundations

---

## ⚡ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/reunicles/Model_Calculator.git
cd Model_Calculator

# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

### 2. CLI Usage
```bash
# Analyze any HuggingFace model
python3 transformer_calculator.py --hf-model "Qwen/Qwen2.5-7B" --mode prefill --table all

# MoE model analysis
python3 transformer_calculator.py --hf-model "deepseek-ai/DeepSeek-V3.1-Terminus" --mode decode --table memory

# Custom sequence length
python3 transformer_calculator.py --hf-model "Qwen/Qwen2.5-7B" --seq-len 4096 --mode prefill --table all
```

### 3. Python API
```python
from src.transformer_calculator import TransformerCalculator, ModelConfig, ModelType, OperationMode

# Create configuration
config = ModelConfig(
    num_layers=24,
    hidden_size=2048,
    num_attention_heads=16,
    sequence_length=2048,
    batch_size=4,
    model_type=ModelType.DENSE
)

# Create calculator and analyze
calculator = TransformerCalculator(config)
print(calculator.generate_table(OperationMode.PREFILL, "all"))
```

---

## 🌐 Web Interface

### Enhanced Interactive Analysis
```bash
# Start the web interface
python3 start_web.py

# Access at http://localhost:5001
```

#### 🎯 Features
- **🤗 Popular Models**: Pre-configured models with one-click selection
- **📊 Model Parameters**: Automatic parameter display and configuration
- **🔧 Custom Models**: Support for any HuggingFace model
- **📈 Real-time Analysis**: Interactive calculations with instant results

#### 🚀 Popular Models Available
| Model | Type | Size | Description |
|-------|------|------|-------------|
| **Qwen3-VL-235B-Thinking** | MoE | 235B | Vision-language with thinking capabilities |
| **DeepSeek-V3.1-Terminus** | MoE | 685B | Enhanced reasoning model |
| **GPT-OSS-120B** | MoE | 120B | OpenAI reasoning model |
| **Falcon-H1-34B** | Dense | 34B | Hybrid Transformer+Mamba |
| **Qwen2.5-72B** | Dense | 72B | Large dense model |
| **Qwen2.5-14B** | Dense | 14B | Medium dense model |
| **Qwen2.5-7B** | Dense | 7B | Small dense model |
| **Phi-3.5-mini** | Dense | 3.8B | Microsoft compact model |

---

## 📊 Operation Modes

### Pretraining
- **Forward Pass**: Standard forward computation
- **Backward Pass**: Gradient computation (1.5× forward memory)
- **Gradients**: Parameter gradients storage (FP32)
- **Optimizer States**: Adam optimizer states

### Prefill
- **Parallel Processing**: Entire sequence processed simultaneously
- **Memory**: Peak usage during sequence processing
- **Flash Attention**: O(seq_len) memory instead of O(seq_len²)

### Decode
- **Autoregressive Generation**: Sequential token generation
- **Memory**: Peak usage during generation
- **Flash Decode**: O(1) memory per token

---

## 🧠 Memory Components

### Detailed Breakdown
1. **Attention**: Q, K, V matrices and attention scores
2. **Projections**: Linear projection layers
3. **MLP**: Multi-layer perceptron (dense or MoE)
4. **Embeddings**: Input and positional embeddings
5. **Layer Norm**: Layer normalization parameters
6. **Activations**: Intermediate activations
7. **Weights**: Model parameters storage
8. **KV Cache**: Key-value cache for inference

### Flash Attention Optimization
| Sequence Length | Standard Memory | Flash Memory | Reduction |
|----------------|----------------|--------------|-----------|
| 1K | O(S²) | O(S) | ~1000× |
| 4K | O(S²) | O(S) | ~4000× |
| 16K | O(S²) | O(S) | ~16000× |
| 65K | O(S²) | O(S) | ~65000× |

---

## 🔀 MoE (Mixture of Experts)

### Expert Architecture
- **Expert Weight Sharing**: Weights shared across all layers
- **Routing**: Top-k expert selection per token
- **Load Balancing**: Expert utilization optimization
- **Memory Efficiency**: ~60× reduction vs naive storage

### MoE Configuration
```python
config = ModelConfig(
    model_type=ModelType.MOE,
    num_experts=8,                # Number of experts
    top_k=2,                     # Top-k expert selection
    expert_capacity_factor=1.0,   # Capacity factor
    moe_intermediate_size=8192,   # Expert MLP dimension
)
```

---

## 📈 Performance Analysis

### Arithmetic Intensity
- **High Reuse**: >100 FLOPS/B (compute-bound)
- **Medium Reuse**: 10-100 FLOPS/B (balanced)
- **Low Reuse**: <10 FLOPS/B (memory-bound)

### Real-World Examples
```bash
# 7B model analysis
python3 cli_calculator.py --hf-model "Qwen/Qwen2.5-7B" --mode prefill --table all

# MoE model with custom parameters
python3 cli_calculator.py --hf-model "deepseek-ai/DeepSeek-V3.1-Terminus" --batch-size 4 --mode decode --table memory
```

---

## 🧪 Testing & Validation

### Comprehensive Test Suite
```bash
# Run all tests
python3 run_all_tests.py

# Unit tests
python3 run_tests.py

# Performance benchmarking
python3 benchmark_suite.py

# Real-world validation
python3 real_world_validation.py
```

### Test Coverage
- **Unit Tests**: 20+ comprehensive test cases
- **Integration Tests**: Real HuggingFace model validation
- **Performance Tests**: Benchmarking and profiling
- **Validation Tests**: Realistic range checking

---

## 📚 Documentation

### Mathematical Foundations
See [Math_Used.md](Math_Used.md) for detailed mathematical equations:
- **Memory Calculations**: Component-wise breakdown
- **FLOPS Calculations**: Operation-wise analysis
- **Arithmetic Intensity**: Performance optimization
- **Flash Attention**: Memory optimization techniques
- **MoE Calculations**: Expert routing and load balancing

### Project Structure
```
├── README.md                   # Main documentation
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── transformer_calculator.py   # CLI entry point
├── start_web.py               # Web interface entry point
├── run_tests.py               # Test runner entry point
├── scripts/                   # Entry point scripts
│   ├── transformer_calculator.py
│   ├── start_web.py
│   ├── web_interface.py
│   └── run_tests.py
├── src/                       # Core source code
│   ├── transformer_calculator.py
│   ├── cli_calculator.py
│   ├── web_interface_enhanced.py
│   ├── hf_model_fetcher.py
│   ├── table_formatter.py
│   └── templates/
├── tests/                     # Test suites
│   ├── test_*.py
│   ├── benchmark_suite.py
│   └── real_world_validation.py
├── docs/                      # Documentation
│   └── Math_Used.md
├── examples/                  # Usage examples
│   ├── example_usage.py
│   └── hf_examples.py
└── archive/                   # Archived files
```

---

## 🔧 Advanced Usage

### Data Type Support
```bash
# Auto-detect from model
python3 cli_calculator.py --hf-model "meta-llama/Llama-2-7b-hf" --mode prefill --table all

# Force specific precision
python3 cli_calculator.py --hf-model "meta-llama/Llama-2-7b-hf" --dtype int8 --mode prefill --table all
```

### Custom Configuration
```python
# Advanced MoE configuration
config = ModelConfig(
    num_layers=32,
    hidden_size=4096,
    model_type=ModelType.MOE,
    num_experts=8,
    top_k=2,
    use_flash_attention=True,
    use_flash_decode=True,
    flash_attention_block_size=4096,
    flash_decode_block_size=1024,
)
```

---

## 📊 Output Examples

### Memory Analysis
```
================================================================================
TRANSFORMER MEMORY ANALYSIS - PREFILL
================================================================================
Component            Memory                  Percent
--------------------------------------------------------------------------------
Attention            953.7 MB                 20.00%
Projections          190.7 MB                  4.00%
MLP                  2.8 GB                   60.00%
Embeddings           95.4 MB                   2.00%
Layer Norm           47.7 MB                   1.00%
Weights              476.8 MB                 10.00%
KV Cache              143.1 MB                 3.00%
--------------------------------------------------------------------------------
TOTAL                4.7 GB                  100.00%
================================================================================
```

### FLOPS Analysis
```
================================================================================
TRANSFORMER FLOPS ANALYSIS - PREFILL
================================================================================
Component            FLOPS                    Percent
--------------------------------------------------------------------------------
Attention            32.3 TFLOP               14.70%
Projections          25.8 TFLOP               11.80%
MLP                  161.5 TFLOP              73.50%
Embeddings           16.8 MFLOP                0.00%
Layer Norm           3.2 GFLOP                 0.00%
--------------------------------------------------------------------------------
TOTAL                220.0 TFLOP             100.00%
================================================================================
```

---

## 🚀 Examples

### Example 1: Model Comparison
```python
# Compare different model sizes
models = [
    ("Qwen/Qwen2.5-7B", "7B Dense"),
    ("Qwen/Qwen2.5-14B", "14B Dense"),
    ("Qwen/Qwen2.5-72B", "72B Dense"),
    ("deepseek-ai/DeepSeek-V3.1-Terminus", "685B MoE")
]

for model_path, description in models:
    print(f"\n=== {description} ===")
    # Run analysis...
```

### Example 2: Sequence Length Analysis
```python
# Analyze memory vs sequence length
sequence_lengths = [512, 1024, 2048, 4096, 8192]

for seq_len in sequence_lengths:
    config = ModelConfig(
        num_layers=12,
        hidden_size=1024,
        sequence_length=seq_len,
        batch_size=1
    )
    
    calculator = TransformerCalculator(config)
    memory = calculator.calculate_memory(OperationMode.PREFILL)
    print(f"Seq Length: {seq_len}, Memory: {memory.total / (1024**2):.2f} MB")
```

### Example 3: MoE Analysis
```python
# Analyze MoE model efficiency
config = ModelConfig(
    num_layers=32,
    hidden_size=4096,
    model_type=ModelType.MOE,
    num_experts=8,
    top_k=2,
    sequence_length=2048,
    batch_size=4
)

calculator = TransformerCalculator(config)
reuse = calculator.calculate_reuse(OperationMode.PREFILL)
print(f"Total Arithmetic Intensity: {reuse.total:.2f} FLOPS/B")
```

---

## 🔧 Troubleshooting

### Common Issues

#### Web Interface
```bash
# Port conflicts
python3 web_interface_enhanced.py  # Auto-finds available port

# Missing Flask
pip install flask
```

#### Model Access
```bash
# Gated models require authentication
# Use publicly available models instead
python3 cli_calculator.py --hf-model "Qwen/Qwen2.5-7B" --mode prefill --table all
```

#### Performance
```bash
# For large models, use smaller batch sizes
python3 cli_calculator.py --hf-model "Qwen/Qwen2.5-72B" --batch-size 1 --mode prefill --table memory
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install mypy pytest

# Run type checking
python3 -m mypy transformer_calculator.py

# Run tests
python3 run_tests.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Key Achievements

### Mathematical Accuracy
- ✅ **Validated Equations**: All formulas tested against real models
- ✅ **Real-world Testing**: Comprehensive validation suite
- ✅ **Memory Optimization**: Flash Attention and Flash Decode support
- ✅ **MoE Support**: Advanced expert routing and optimization

### Code Quality
- ✅ **Professional Documentation**: Comprehensive docstrings and examples
- ✅ **Type Safety**: Static type checking with MyPy
- ✅ **Unit Testing**: 20+ comprehensive test cases
- ✅ **Validation**: Real-world model testing and benchmarking

### User Experience
- ✅ **HuggingFace Integration**: Automatic model configuration
- ✅ **Web Interface**: Interactive analysis with popular models
- ✅ **CLI Efficiency**: Optimized command-line interface
- ✅ **Human-Readable Output**: Formatted tables with proper units

---

<div align="center">

**Built with ❤️ for the AI community**

[GitHub](https://github.com/reunicles/Model_Calculator) • [Issues](https://github.com/reunicles/Model_Calculator/issues) • [Discussions](https://github.com/reunicles/Model_Calculator/discussions)

</div>