"""
Hardware Specifications for GPU Node and Rack Comparisons

This module provides specifications for high-end GPU nodes and racks
to help users understand the scale of their models relative to
actual hardware capabilities.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HardwareSpecs:
    """Hardware specifications for GPU nodes and racks"""
    name: str
    memory_gb: float
    flops_fp16: float  # FLOPS in FP16
    flops_fp32: float  # FLOPS in FP32
    description: str


# GPU Node Specifications
# Based on official NVIDIA specifications from https://www.nvidia.com/en-us/data-center/gb300-nvl72/
HARDWARE_SPECS = {
    "gb300_single_gpu": HardwareSpecs(
        name="GB300 Single GPU",
        memory_gb=288.0,  # 288 GB HBM3e per GPU
        flops_fp16=15.0e15,  # 15 PetaFLOPS FP4
        flops_fp32=7.5e15,  # 7.5 PetaFLOPS FP32 (estimated)
        description="Single NVIDIA GB300 GPU with 288GB HBM3e, 8TB/s memory bandwidth"
    ),
    "gb300_nvl72_rack": HardwareSpecs(
        name="GB300-NVL72 Rack",
        memory_gb=20736.0,  # 72 GPUs × 288 GB HBM3e
        flops_fp16=1080.0e15,  # 72 × 15 PFLOPS FP4
        flops_fp32=540.0e15,  # 72 × 7.5 PFLOPS FP32
        description="Full rack with 72 GB300 GPUs, 36 Grace CPUs"
    ),
    "gb300_node_estimate": HardwareSpecs(
        name="GB300 Node (Estimated)",
        memory_gb=576.0,  # Estimated 2 GPUs per node
        flops_fp16=30.0e15,  # 2 × 15 PetaFLOPS FP4
        flops_fp32=15.0e15,  # 2 × 7.5 PetaFLOPS FP32
        description="Estimated single node with 2 GB300 GPUs"
    ),
    "vr200_single_gpu": HardwareSpecs(
        name="VR200 Single GPU",
        memory_gb=288.0,  # 288 GB HBM4 per GPU
        flops_fp16=33.3e15,  # 33.3 PetaFLOPS FP4
        flops_fp32=16.65e15,  # Estimated FP32 performance
        description="Single NVIDIA VR200 GPU with 288GB HBM4, 20.5TB/s memory bandwidth"
    ),
    "vr200_nvl144_rack": HardwareSpecs(
        name="VR200-NVL144 Rack",
        memory_gb=20736.0,  # 72 GPUs × 288 GB HBM4
        flops_fp16=2397.6e15,  # 72 × 33.3 PFLOPS FP4
        flops_fp32=1198.8e15,  # 72 × 16.65 PFLOPS FP32
        description="Full rack with 72 VR200 GPUs, 144 nodes total"
    ),
    "vr200_node_estimate": HardwareSpecs(
        name="VR200 Node (Estimated)", 
        memory_gb=576.0,  # Estimated 2 GPUs per node
        flops_fp16=66.6e15,  # 2 × 33.3 PetaFLOPS FP4
        flops_fp32=33.3e15,  # 2 × 16.65 PetaFLOPS FP32
        description="Estimated single node with 2 VR200 GPUs"
    )
}


def get_hardware_comparison(model_memory_gb: float, model_flops: float, 
                          data_type: str = "fp16") -> Dict[str, Any]:
    """
    Compare model requirements against hardware capabilities
    
    Args:
        model_memory_gb: Model memory requirement in GB
        model_flops: Model FLOPS requirement
        data_type: Data type for FLOPS comparison ("fp16" or "fp32")
    
    Returns:
        Dictionary with hardware comparison results
    """
    flops_key = f"flops_{data_type}"
    
    results = {}
    
    for spec_name, spec in HARDWARE_SPECS.items():
        # Memory comparison
        memory_ratio = model_memory_gb / spec.memory_gb
        memory_fits = memory_ratio <= 1.0
        
        # FLOPS comparison
        flops_ratio = model_flops / getattr(spec, flops_key)
        flops_fits = flops_ratio <= 1.0
        
        # Overall fit
        overall_fits = memory_fits and flops_fits
        
        results[spec_name] = {
            "name": spec.name,
            "description": spec.description,
            "memory_gb": spec.memory_gb,
            "flops": getattr(spec, flops_key),
            "memory_ratio": memory_ratio,
            "flops_ratio": flops_ratio,
            "memory_fits": memory_fits,
            "flops_fits": flops_fits,
            "overall_fits": overall_fits,
            "memory_utilization": f"{memory_ratio:.1%}",
            "flops_utilization": f"{flops_ratio:.1%}"
        }
    
    return results


def format_hardware_summary(comparison_results: Dict[str, Any]) -> str:
    """
    Format hardware comparison results into a readable summary
    
    Args:
        comparison_results: Results from get_hardware_comparison()
    
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 80)
    lines.append("HARDWARE CAPABILITY COMPARISON")
    lines.append("=" * 80)
    
    # Single GPU comparisons
    lines.append("\n🎯 SINGLE GPU CAPABILITIES:")
    lines.append("-" * 50)
    
    for spec_name in ["gb300_single_gpu", "vr200_single_gpu"]:
        if spec_name in comparison_results:
            result = comparison_results[spec_name]
            status = "✅ FITS" if result["overall_fits"] else "❌ EXCEEDS"
            lines.append(f"{result['name']}: {status}")
            lines.append(f"  Memory: {result['memory_utilization']} of {result['memory_gb']:.0f} GB")
            lines.append(f"  FLOPS:  {result['flops_utilization']} of {result['flops']/1e15:.1f} PFLOPS")
            lines.append("")
    
    # Node-level comparisons
    lines.append("📊 NODE-LEVEL CAPABILITIES:")
    lines.append("-" * 50)
    
    for spec_name in ["gb300_node_estimate", "vr200_node_estimate"]:
        if spec_name in comparison_results:
            result = comparison_results[spec_name]
            status = "✅ FITS" if result["overall_fits"] else "❌ EXCEEDS"
            lines.append(f"{result['name']}: {status}")
            lines.append(f"  Memory: {result['memory_utilization']} of {result['memory_gb']:.0f} GB")
            lines.append(f"  FLOPS:  {result['flops_utilization']} of {result['flops']/1e15:.1f} PFLOPS")
            lines.append("")
    
    # Rack-level comparisons
    lines.append("🏢 RACK-LEVEL CAPABILITIES:")
    lines.append("-" * 50)
    
    for spec_name in ["gb300_nvl72_rack", "vr200_nvl144_rack"]:
        if spec_name in comparison_results:
            result = comparison_results[spec_name]
            status = "✅ FITS" if result["overall_fits"] else "❌ EXCEEDS"
            lines.append(f"{result['name']}: {status}")
            lines.append(f"  Memory: {result['memory_utilization']} of {result['memory_gb']:.0f} GB")
            lines.append(f"  FLOPS:  {result['flops_utilization']} of {result['flops']/1e15:.1f} PFLOPS")
            lines.append("")
    
    # Summary recommendations
    lines.append("💡 RECOMMENDATIONS:")
    lines.append("-" * 30)
    
    # Find the smallest hardware that fits
    fitting_hardware = []
    for spec_name, result in comparison_results.items():
        if result["overall_fits"]:
            fitting_hardware.append((spec_name, result))
    
    if fitting_hardware:
        # Sort by total capacity (memory + flops)
        fitting_hardware.sort(key=lambda x: x[1]["memory_gb"] + x[1]["flops"]/1e12)
        smallest = fitting_hardware[0][1]
        lines.append(f"✅ Minimum required: {smallest['name']}")
        lines.append(f"   Memory utilization: {smallest['memory_utilization']}")
        lines.append(f"   FLOPS utilization: {smallest['flops_utilization']}")
    else:
        lines.append("❌ Model exceeds all available hardware capabilities")
        lines.append("   Consider model optimization or distributed training")
    
    return "\n".join(lines)


def get_hardware_json(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hardware comparison results to JSON-serializable format
    
    Args:
        comparison_results: Results from get_hardware_comparison()
    
    Returns:
        JSON-serializable dictionary
    """
    json_results = {}
    
    for spec_name, result in comparison_results.items():
        json_results[spec_name] = {
            "name": result["name"],
            "description": result["description"],
            "memory_gb": result["memory_gb"],
            "flops": result["flops"],
            "memory_ratio": result["memory_ratio"],
            "flops_ratio": result["flops_ratio"],
            "memory_fits": result["memory_fits"],
            "flops_fits": result["flops_fits"],
            "overall_fits": result["overall_fits"],
            "memory_utilization": result["memory_utilization"],
            "flops_utilization": result["flops_utilization"]
        }
    
    return json_results
