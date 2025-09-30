# Enhanced Output with Missing Parameter Detection

## Overview

The transformer calculator now provides enhanced output that clearly indicates when parameters are missing and shows the defaults being used. This makes the analysis more transparent and helps users understand when the calculator is making assumptions.

## Key Features

### 1. Missing Parameter Warnings

When model configurations are incomplete, the calculator now shows:

```
⚠️  WARNING: Missing parameters for Qwen/Qwen3-VL-235B-A22B-Instruct
Missing parameters: hidden_size, num_layers, num_attention_heads, vocab_size, intermediate_size
Using intelligent defaults based on model name:
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32
  vocab_size: 50257
  intermediate_size: 16384
Consider providing model parameters manually for more accurate calculations.
  Inferred from model name: hidden_size=5120, num_layers=64, num_heads=40
```

### 2. Enhanced Table Output

The analysis tables now include comprehensive parameter information:

```
============================================================
TRANSFORMER MEMORY ANALYSIS - PREFILL
============================================================
Model: DENSE
Layers: 64, Hidden Size: 5120
Sequence Length: 2048, Batch Size: 1
Attention Heads: 40 (KV: 32), Intermediate Size: 16384
Vocab Size: 50,257, Precision: 16-bit
============================================================
```

### 3. Intelligent Defaults

The calculator uses intelligent defaults based on:

- **Model name patterns**: Detects model size from names (235B, 120B, 30B)
- **Architecture inference**: Determines appropriate parameters for different model sizes
- **Fallback values**: Uses reasonable defaults when inference fails

### 4. Complete vs. Incomplete Configurations

**Complete Configuration (Qwen3-Omni hardcoded):**
```
Using hardcoded Qwen3-Omni configuration...

============================================================
TRANSFORMER MEMORY ANALYSIS - PREFILL
============================================================
Model: MOE
Layers: 32, Hidden Size: 4096
Sequence Length: 2048, Batch Size: 1
Attention Heads: 32 (KV: 8), Intermediate Size: 11008
MoE: 8 experts, top-2, capacity: 1.0
Vocab Size: 151,936, Precision: 16-bit
============================================================
```

**Incomplete Configuration (Hugging Face fetch):**
```
⚠️  WARNING: Missing parameters for Qwen/Qwen3-VL-235B-A22B-Instruct
Missing parameters: hidden_size, num_layers, num_attention_heads, vocab_size, intermediate_size
...
```

## Benefits

1. **Transparency**: Users know exactly when defaults are being used
2. **Accuracy**: Clear indication of parameter confidence levels
3. **Debugging**: Easy to identify configuration issues
4. **Trust**: Users can verify the assumptions being made
5. **Flexibility**: Option to provide manual parameters for accuracy

## Usage Examples

```bash
# Models with complete configurations (no warnings)
python3 cli_calculator.py --model qwen3-omni --mode prefill --table memory

# Models with missing parameters (shows warnings and defaults)
python3 cli_calculator.py --hf-model "Qwen/Qwen3-VL-235B-A22B-Instruct" --mode prefill --table memory

# Custom models (no warnings, user-specified parameters)
python3 cli_calculator.py --layers 24 --hidden-size 2048 --heads 16 --seq-len 2048 --batch-size 4
```

## Parameter Detection

The calculator checks for these key parameters:

- `hidden_size` / `d_model`
- `num_hidden_layers` / `n_layer`
- `num_attention_heads` / `n_head`
- `vocab_size`
- `intermediate_size` / `ffn_dim`
- `num_key_value_heads` (for GQA/MQA)
- MoE parameters (`num_experts`, `top_k`, etc.)

When any of these are missing, the calculator provides intelligent defaults and clearly indicates the assumptions being made.
