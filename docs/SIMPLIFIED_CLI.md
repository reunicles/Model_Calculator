# Simplified CLI - Hugging Face Only

## Overview

The transformer calculator has been simplified to focus exclusively on Hugging Face models, providing a streamlined interface that automatically extracts all necessary parameters from `config.json` files.

## Key Simplifications

### 1. Hugging Face Only
- **Removed**: `--model` parameter for predefined models
- **Removed**: Custom model architecture parameters
- **Required**: `--hf-model` parameter for Hugging Face model path
- **Automatic**: All parameters extracted from `config.json`

### 2. Automatic Max Sequence Length
- **Default**: Uses model's `max_position_embeddings` from config
- **Override**: `--seq-len` parameter for custom sequence length
- **Smart Detection**: Handles both standard and multimodal models

### 3. Simplified Interface

**Before (Complex):**
```bash
python cli_calculator.py --layers 24 --hidden-size 2048 --heads 16 --seq-len 2048 --batch-size 4 --mode prefill --table memory
```

**After (Simple):**
```bash
python cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --mode prefill --table memory
```

## Usage Examples

### Basic Analysis
```bash
# Uses model's max sequence length automatically
python cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --mode prefill --table memory

# Decode analysis
python cli_calculator.py --hf-model "openai/gpt-oss-120b" --mode decode --table both
```

### Custom Sequence Length
```bash
# Override max sequence length
python cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --seq-len 4096 --mode prefill --table memory
```

### Decode with Custom Length
```bash
# Custom decode length
python cli_calculator.py --hf-model "openai/gpt-oss-120b" --decode-length 128 --mode decode --table both
```

## Automatic Parameter Detection

The calculator now automatically extracts:

- **Architecture**: Layers, hidden size, attention heads
- **MoE Configuration**: Expert counts, top-k selection  
- **Sequence Length**: Model's maximum supported length
- **Vocabulary**: Actual vocabulary size
- **Precision**: Model's data type
- **Multimodal Support**: Text component parameters

## Benefits

1. **🎯 Simplicity**: Single parameter to specify model
2. **🔍 Accuracy**: Real parameters from `config.json`
3. **⚡ Speed**: No manual parameter specification
4. **🔧 Reliability**: Works with any Hugging Face model
5. **📊 Precision**: Accurate memory and FLOPS calculations

## Model Support

- **Standard Transformers**: GPT, Llama, Mistral, etc.
- **Multimodal Models**: Qwen3-VL, GPT-4V, etc.
- **MoE Models**: Mixtral, Qwen3-Omni, GPT-OSS, etc.
- **Vision-Language Models**: Complete text component analysis

## Example Output

```
Fetching model configuration from Hugging Face: Qwen/Qwen3-VL-235B-A22B-Instruct
Using model's max sequence length: 262144
Detected multimodal model with text_config
Using text_config parameters:
  hidden_size: 4096
  num_layers: 94
  num_attention_heads: 64
  vocab_size: 151936
  intermediate_size: 12288
  MoE: 128 experts, top-8
```

The simplified CLI provides a much cleaner and more reliable interface for analyzing transformer models from Hugging Face Hub!
