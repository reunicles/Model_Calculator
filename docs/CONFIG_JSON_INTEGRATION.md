# Config.json Integration for Accurate Model Parameters

## Overview

The transformer calculator now uses the canonical `config.json` file from Hugging Face model repositories to fetch complete and accurate model parameters. This eliminates the need for intelligent defaults and provides precise calculations.

## Key Improvements

### 1. Direct Config.json Access

Instead of using incomplete model info, the calculator now:
- Downloads the `config.json` file directly from Hugging Face Hub
- Extracts all model parameters from the canonical configuration
- Provides accurate parameter values without assumptions

### 2. Multimodal Model Support

The calculator now properly handles multimodal models like Qwen3-VL:
- Detects `text_config` in the configuration
- Extracts parameters from the text component
- Supports vision-language models correctly

### 3. Complete Parameter Detection

Modern models now provide all necessary parameters:

**Qwen3-VL-235B-A22B-Instruct:**
```
Model: MOE
Layers: 94, Hidden Size: 4096
Attention Heads: 64 (KV: 4), Intermediate Size: 12288
MoE: 128 experts, top-8, capacity: 1.0
Vocab Size: 151,936, Precision: 16-bit
```

**GPT-OSS-120B:**
```
Model: MOE
Layers: 36, Hidden Size: 2880
Attention Heads: 64 (KV: 8), Intermediate Size: 2880
MoE: 128 experts, top-4, capacity: 1.0
Vocab Size: 201,088, Precision: 16-bit
```

### 4. No More Missing Parameter Warnings

Models with complete `config.json` files show no warnings:
- All parameters are accurately extracted
- No intelligent defaults needed
- Full transparency about model architecture

## Architecture Detection

The calculator now properly detects:

- **Standard Transformers**: Direct parameter extraction
- **Multimodal Models**: Uses `text_config` for text component
- **MoE Models**: Extracts expert counts and top-k values
- **GQA/MQA**: Handles grouped query attention correctly

## Example Usage

```bash
# Modern models with complete configs (no warnings)
python3 cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --mode prefill --table memory
python3 cli_calculator.py --hf-model "https://huggingface.co/openai/gpt-oss-120b" --mode decode --table both

# Older models still show warnings for missing parameters
python3 cli_calculator.py --hf-model "microsoft/DialoGPT-medium" --mode prefill --table memory
```

## Benefits

1. **🎯 Accuracy**: Uses actual model parameters instead of estimates
2. **🔍 Transparency**: No hidden assumptions about model architecture
3. **📊 Precision**: Correct memory and FLOPS calculations
4. **🚀 Performance**: Accurate resource planning for deployment
5. **🛠️ Reliability**: Works with any Hugging Face model

## Technical Details

The calculator now:
- Downloads `config.json` using `hf_hub_download()`
- Parses JSON configuration directly
- Handles multimodal model structures
- Extracts MoE parameters accurately
- Provides complete parameter transparency

This ensures that memory and FLOPS calculations are based on the actual model architecture rather than intelligent guesses, providing much more accurate results for production use cases.
