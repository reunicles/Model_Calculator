# Mathematical Foundations

> **Comprehensive mathematical foundations for transformer memory and compute analysis**

[![Transformer Calculator](https://img.shields.io/badge/Transformer-Calculator-blue.svg)](https://github.com/reunicles/Model_Calculator)
[![Mathematical Accuracy](https://img.shields.io/badge/Mathematical-Accuracy-green.svg)](https://github.com/reunicles/Model_Calculator)
[![MoE Support](https://img.shields.io/badge/MoE-Supported-orange.svg)](https://github.com/reunicles/Model_Calculator)

## 🎯 Quick Reference

### Core Symbols
| Symbol | Meaning | Example |
|--------|---------|---------|
| **B** | Batch size | 1, 32, 128 |
| **S** | Sequence length | 2048, 8192, 65536 |
| **L** | Number of layers | 32, 80, 94 |
| **H** | Attention heads | 32, 64, 128 |
| **d** | Hidden dimension | 4096, 8192 |
| **d_h** | Head dimension (d/H) | 128, 64 |
| **d_ff** | MLP dimension | 4×d, 8×d |

### Master Equations
```math
Prefill FLOPs = L × [5BSd² + 2BS²d]
Decode FLOPs  = L × [5Bd² + 2BdS]
KV Cache     = B × S × H × (d_k + d_v) × bytes
```

---

## 📊 Per-Layer Operations

### Standard Transformer Layer
| Operation | FLOPs | Memory | Arithmetic Intensity |
|-----------|-------|--------|---------------------|
| **QKV Projections** | `3BSd²` | `3BSd × bytes` | `d/bytes` |
| **Attention Scores** | `2BSHd_hS` | `2BSHd_hS × bytes` | `1/bytes` |
| **Attention × V** | `2BSHd_hS` | `2BSHd_hS × bytes` | `1/bytes` |
| **Output Projection** | `BSd²` | `BSd × bytes` | `d/bytes` |
| **MLP Up** | `BSdd_ff` | `BSdd_ff × bytes` | `d_ff/bytes` |
| **MLP Down** | `BSdd_ff` | `BSdd_ff × bytes` | `d_ff/bytes` |

**Total per layer**: Sum all operations  
**Total for model**: Multiply by L layers

---

## 🔄 Operation Modes

### Prefill (Parallel Processing)
- **Memory**: Peak usage during sequence processing
- **FLOPS**: `L × [5BSd² + 2BS²d]`
- **Attention**: O(S²) - quadratic scaling
- **Flash Attention**: O(S) - linear scaling

### Decode (Autoregressive)
- **Memory**: Peak usage during generation
- **FLOPS**: `L × [5Bd² + 2BdS]` per token
- **Attention**: O(S) - linear scaling
- **Flash Decode**: O(1) - constant scaling

### Pretraining (Full Training)
- **Memory**: Forward + Backward + Gradients + Optimizer
- **FLOPS**: 3× forward pass (forward + backward)
- **Gradients**: FP32 parameter gradients
- **Optimizer**: Adam states (momentum + variance)

---

## 🧠 Memory Components

### 1. Attention Memory
```math
Standard: S² × B × H × bytes
Flash:    block_size × B × H × bytes × factor
```

### 2. MLP Memory
```math
Dense: 2BSdd_ff × bytes
MoE:   K × util × 2BSd × d_ff_moe × bytes
```

### 3. KV Cache
```math
KV_bytes = B × S × H × (d_k + d_v) × bytes_kv
Read/Write = 2 × KV_bytes per token
```

### 4. Expert Weights (MoE)
```math
Expert_weights = E × 2 × d × d_ff_moe × bytes
# Shared across all layers (no L multiplication)
```

---

## ⚡ Flash Attention Optimization

### Memory Reduction
| Sequence Length | Standard Memory | Flash Memory | Reduction |
|----------------|----------------|--------------|-----------|
| 1K | O(S²) | O(S) | ~1000× |
| 4K | O(S²) | O(S) | ~4000× |
| 16K | O(S²) | O(S) | ~16000× |
| 65K | O(S²) | O(S) | ~65000× |

### Sequence Length Bucketing
```math
factor = {
    1.1  if S ≤ 1K
    1.2  if S ≤ 4K  
    1.5  if S ≤ 16K
    2.0  if S ≤ 65K
    2.5  if S > 65K
}
```

---

## 🔀 MoE (Mixture of Experts)

### Expert Utilization
```math
util = K / E
# K = top_k experts per token
# E = total number of experts
```

### MoE Memory Optimization
```math
HBM_Storage = E × 2 × d × d_ff_moe × bytes
GPU_Memory = K × util × 2 × d × d_ff_moe × bytes
```

### Capacity Factor
```math
capacity = S × B × K × capacity_factor
# Handles dropped tokens vs padding overhead
```

---

## 📈 Arithmetic Intensity Analysis

### Performance Ranges
| Mode | Attention | MLP | Total |
|------|-----------|-----|-------|
| **Pretraining** | 1K-1M FLOPS/B | 100-10K FLOPS/B | 10K-1M FLOPS/B |
| **Prefill** | 1K-1M FLOPS/B | 100-10K FLOPS/B | 10K-1M FLOPS/B |
| **Decode** | 0.1-10 FLOPS/B | 1-100 FLOPS/B | 1-100 FLOPS/B |

### Efficiency Classification
- **High**: >100 FLOPS/B (compute-bound)
- **Medium**: 10-100 FLOPS/B (balanced)
- **Low**: <10 FLOPS/B (memory-bound)

---

## 🎯 Data Types

### Precision Support
| Type | Bytes | Usage | Memory Impact |
|------|-------|-------|---------------|
| **FP32** | 4 | Full precision | 4× baseline |
| **FP16/BF16** | 2 | Standard | 2× baseline |
| **INT8** | 1 | Quantized | 1× baseline |
| **INT4** | 0.5 | Extreme quantization | 0.5× baseline |

### KV Cache Optimization
```math
KV_FP16 = B × S × H × (d_k + d_v) × 2
KV_INT8 = B × S × H × (d_k + d_v) × 1
# 2× memory reduction with INT8
```

---

## 🚀 Real-World Examples

### Model Scaling
| Parameters | Layers | Hidden | Heads | Typical S | Memory (GB) |
|-----------|--------|--------|-------|-----------|-------------|
| **7B** | 32 | 4096 | 32 | 2K-8K | 14-56 |
| **13B** | 40 | 5120 | 40 | 2K-8K | 26-104 |
| **70B** | 80 | 8192 | 64 | 2K-32K | 140-1120 |
| **MoE-8×7B** | 32 | 4096 | 32 | 2K-8K | 56-224 |

### Flash Attention Impact
```math
# Example: 65K context length
Standard_Memory = 65K² × B × H × bytes ≈ 4.2B × B × H × bytes
Flash_Memory = 4096 × B × H × bytes × 2.5 ≈ 10K × B × H × bytes
Reduction = 420,000×
```

---

## 🔧 Implementation Notes

### Critical Considerations
1. **Attention Scaling**: O(S²) → O(S) with Flash Attention
2. **MoE Expert Sharing**: Weights shared across layers
3. **Memory Traffic**: Use GPU memory for MoE arithmetic intensity
4. **KV Cache**: Full sequence for storage, 1 token for computation
5. **Flash Decode**: O(S) → O(1) for inference optimization

### Validation Ranges
- **Memory**: 1MB - 10TB (realistic model sizes)
- **FLOPS**: 1M - 1E FLOPS (from 1B to 1T parameters)
- **Arithmetic Intensity**: 0.1 - 1M FLOPS/B (memory to compute bound)

---

## 📚 Further Reading

- [Transformer Calculator Documentation](../README.md)
- [CLI Usage Guide](../README.md#-cli-usage)
- [Web Interface Guide](../README.md#-enhanced-web-interface)
- [MoE Advanced Features](../README.md#-advanced-moe-optimizations)

---

*This document provides the mathematical foundations for accurate transformer model analysis. For implementation details and usage examples, see the main [README.md](../README.md).*