# Human-Readable Formatting Enhancement

## Overview

Added human-readable columns to the transformer calculator output to make memory and FLOPS values more intuitive and easier to understand.

## Memory Formatting

### Units Used
- **B**: Bytes
- **KB**: Kilobytes (1,024 bytes)
- **MB**: Megabytes (1,024 KB)
- **GB**: Gigabytes (1,024 MB)
- **TB**: Terabytes (1,024 GB)
- **PB**: Petabytes (1,024 TB)
- **EB**: Exabytes (1,024 PB)

### Examples
```
Component            Memory (MB)     Human Readable  Percentage  
--------------------------------------------------------------------------------
Attention            2176.00         2.1 GB          48.2        %
Projections          128.00          128.0 MB        2.8         %
MLP                  801.00          801.0 MB         17.7       %
Embeddings           1219.00         1.2 GB          27.0        %
Layer Norm           64.00           64.0 MB         1.4         %
Activations          128.00          128.0 MB        2.8         %
--------------------------------------------------------------------------------
TOTAL                4516.00         4.4 GB          100.0       %
```

## FLOPS Formatting

### Units Used
- **FLOP**: Floating Point Operations
- **KFLOP**: Kilo-FLOP (1,000 FLOP)
- **MFLOP**: Mega-FLOP (1,000 KFLOP)
- **GFLOP**: Giga-FLOP (1,000 MFLOP)
- **TFLOP**: Tera-FLOP (1,000 GFLOP)
- **PFLOP**: Peta-FLOP (1,000 TFLOP)
- **EFLOP**: Exa-FLOP (1,000 PFLOP)

### Examples
```
Component            FLOPS                Human Readable  Percentage  
--------------------------------------------------------------------------------
Attention            3.23e+13             32.3 TFLOP      14.7        %
Projections          2.58e+13             25.8 TFLOP      11.8        %
MLP                  1.61e+14             161.5 TFLOP     73.5        %
Embeddings           1.68e+07             16.8 MFLOP      0.0         %
Layer Norm           3.15e+09             3.2 GFLOP       0.0         %
--------------------------------------------------------------------------------
TOTAL                2.20e+14             219.6 TFLOP     100.0       %
```

## Implementation Details

### Helper Functions Added

```python
def _format_bytes_human(self, bytes_value: float) -> str:
    """Format bytes in human readable format (KB, MB, GB, TB, PB, EB)"""
    if bytes_value == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
    unit_index = 0
    value = bytes_value
    
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    
    return f"{value:.1f} {units[unit_index]}"

def _format_flops_human(self, flops_value: float) -> str:
    """Format FLOPS in human readable format (FLOP, KFLOP, MFLOP, GFLOP, TFLOP, PFLOP, EFLOP)"""
    if flops_value == 0:
        return "0 FLOP"
    
    units = ['FLOP', 'KFLOP', 'MFLOP', 'GFLOP', 'TFLOP', 'PFLOP', 'EFLOP']
    unit_index = 0
    value = flops_value
    
    while value >= 1000 and unit_index < len(units) - 1:
        value /= 1000
        unit_index += 1
    
    return f"{value:.1f} {units[unit_index]}"
```

### Table Format Updates

- **Memory Tables**: Added "Human Readable" column
- **FLOPS Tables**: Added "Human Readable" column
- **Wider Tables**: Increased table width to accommodate new columns
- **Consistent Formatting**: Single decimal point precision for human-readable values

## Benefits

1. **🎯 Intuitive**: Easy to understand at a glance
2. **📊 Scalable**: Works across all model sizes
3. **🔍 Precise**: Maintains exact values in original columns
4. **📱 Readable**: Better for reports and presentations
5. **⚡ Quick**: Instant understanding of magnitude

## Scale Examples

### Small Models (DialoGPT-medium)
- Memory: 856.6 MB
- FLOPS: 13.6 TFLOP

### Medium Models (Qwen3-VL with 4K context)
- Memory: 4.4 GB
- FLOPS: 219.6 TFLOP

### Large Models (Qwen3-VL with 256K context)
- Memory: 8.1 TB
- FLOPS: 40.1 PFLOP

The human-readable formatting makes the transformer calculator output much more accessible and easier to interpret! 🚀
