"""
4-bit Quantized AdamW Optimizer using RTN min-max scaling.

This optimizer reduces memory usage by quantizing the momentum states
(exp_avg and exp_avg_sq) to 4-bit precision, achieving ~4x memory savings
for optimizer states compared to standard FP32 AdamW.
"""

import math
import torch
from torch.optim import AdamW


def quantize_4bit(tensor):
    """
    Quantize tensor to 4-bit using min-max scaling, preserving tensor type.
    
    Args:
        tensor: Input tensor to quantize (can be DTensor or regular tensor)
    
    Returns:
        Tuple of (quantized_tensor, min_val, max_val, scale)
    """
    # Work with the local tensor if it's a DTensor
    if hasattr(tensor, '_local_tensor'):
        local_tensor = tensor._local_tensor
        device = local_tensor.device
        dtype = local_tensor.dtype
    else:
        local_tensor = tensor
        device = tensor.device
        dtype = tensor.dtype
    
    min_val = local_tensor.min()
    max_val = local_tensor.max()
    
    # Avoid division by zero
    scale = (max_val - min_val) / 15.0  # 2^4 - 1 = 15
    if scale == 0:
        scale = torch.tensor(1.0, device=device, dtype=dtype)
    
    # Quantize to 4-bit integers (0-15)
    quantized_local = torch.clamp(torch.round((local_tensor - min_val) / scale), 0, 15).to(torch.uint8)
    
    # Preserve DTensor structure if original was DTensor
    if hasattr(tensor, '_local_tensor'):
        from torch.distributed.tensor import DTensor
        quantized = DTensor.from_local(quantized_local, tensor.device_mesh, tensor.placements)
    else:
        quantized = quantized_local
    
    return quantized, min_val, max_val, scale


def dequantize_4bit(quantized, min_val, max_val, scale):
    """
    Dequantize 4-bit tensor back to float, preserving tensor structure.
    
    Args:
        quantized: 4-bit quantized tensor
        min_val: Minimum value used for quantization
        max_val: Maximum value used for quantization
        scale: Scale factor used for quantization
    
    Returns:
        Dequantized tensor in original precision
    """
    # Work with local tensor if DTensor
    if hasattr(quantized, '_local_tensor'):
        local_quantized = quantized._local_tensor.float()
        result_local = local_quantized * scale + min_val
        
        # Reconstruct DTensor
        from torch.distributed.tensor import DTensor
        result = DTensor.from_local(result_local, quantized.device_mesh, quantized.placements)
    else:
        result = quantized.float() * scale + min_val
    
    return result


class Quantized4BitAdamW(AdamW):
    """
    AdamW optimizer with 4-bit quantized optimizer states using RTN min-max scaling.
    
    This optimizer quantizes the momentum states (exp_avg and exp_avg_sq) to 4-bit
    precision after each update, reducing memory usage by approximately 4x compared
    to standard FP32 AdamW.
    
    Memory savings:
    - Standard AdamW: 8 bytes per parameter (2 states * 4 bytes each)
    - 4-bit AdamW: ~2 bytes per parameter (2 states * 1 byte each, storing 4-bit values in uint8)
    - Theoretical optimal with bit packing: 1 byte per parameter (2 states * 0.5 bytes each)
    
    Note: Current implementation stores one 4-bit value per byte for simplicity.
    Further optimization could pack two 4-bit values per byte.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with 4-bit quantized states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize quantized states with proper structure
                    # FSDP creates optimizer states with same structure as the tensor they track
                    # So we create dummy states and immediately quantize them
                    dummy_exp_avg = torch.zeros_like(p)
                    dummy_exp_avg_sq = torch.zeros_like(p)
                    
                    exp_avg_q, exp_avg_min, exp_avg_max, exp_avg_scale = quantize_4bit(dummy_exp_avg)
                    exp_avg_sq_q, exp_avg_sq_min, exp_avg_sq_max, exp_avg_sq_scale = quantize_4bit(dummy_exp_avg_sq)
                    
                    state['exp_avg_quantized'] = exp_avg_q
                    state['exp_avg_min'] = exp_avg_min
                    state['exp_avg_max'] = exp_avg_max
                    state['exp_avg_scale'] = exp_avg_scale
                    state['exp_avg_sq_quantized'] = exp_avg_sq_q
                    state['exp_avg_sq_min'] = exp_avg_sq_min
                    state['exp_avg_sq_max'] = exp_avg_sq_max
                    state['exp_avg_sq_scale'] = exp_avg_sq_scale
                
                # Dequantize current states
                exp_avg = dequantize_4bit(
                    state['exp_avg_quantized'], 
                    state['exp_avg_min'], 
                    state['exp_avg_max'], 
                    state['exp_avg_scale']
                )
                exp_avg_sq = dequantize_4bit(
                    state['exp_avg_sq_quantized'], 
                    state['exp_avg_sq_min'], 
                    state['exp_avg_sq_max'], 
                    state['exp_avg_sq_scale']
                )
                
                state['step'] += 1
                
                # Weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update momentum terms
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Re-quantize states
                exp_avg_q, exp_avg_min, exp_avg_max, exp_avg_scale = quantize_4bit(exp_avg)
                exp_avg_sq_q, exp_avg_sq_min, exp_avg_sq_max, exp_avg_sq_scale = quantize_4bit(exp_avg_sq)
                
                state['exp_avg_quantized'] = exp_avg_q
                state['exp_avg_min'] = exp_avg_min
                state['exp_avg_max'] = exp_avg_max
                state['exp_avg_scale'] = exp_avg_scale
                state['exp_avg_sq_quantized'] = exp_avg_sq_q
                state['exp_avg_sq_min'] = exp_avg_sq_min
                state['exp_avg_sq_max'] = exp_avg_sq_max
                state['exp_avg_sq_scale'] = exp_avg_sq_scale

        return loss
    
    def get_memory_stats(self):
        """
        Get actual memory usage statistics by measuring tensor sizes.
        
        Returns:
            Dictionary containing memory usage statistics including:
            - total_parameters: Total number of parameters
            - estimated_fp32_gb: Estimated memory for FP32 AdamW
            - actual_quantized_gb: Actual memory used by 4-bit quantization
            - memory_saved_gb: Memory saved compared to FP32
            - memory_reduction_percent: Percentage memory reduction
            - compression_ratio: Compression ratio achieved
        """
        total_params = 0
        actual_quantized_bytes = 0
        estimated_fp32_bytes = 0
        
        for group in self.param_groups:
            for p in group['params']:
                param_count = p.numel()
                total_params += param_count
                
                # Estimate what FP32 would use (2 states * 4 bytes per element)
                estimated_fp32_bytes += param_count * 8
                
                state = self.state.get(p, {})
                if 'exp_avg_quantized' in state:
                    # Measure actual memory usage of quantized states
                    exp_avg_q = state['exp_avg_quantized']
                    exp_avg_sq_q = state['exp_avg_sq_quantized']
                    
                    # 4-bit quantized tensors stored as uint8 (but only using 4 bits worth of info)
                    # We could pack 2 values per byte, but we're storing 1 value per byte for simplicity
                    # So actual storage is 1 byte per element, but theoretical optimal would be 0.5 bytes
                    actual_storage_bytes = exp_avg_q.numel() * exp_avg_q.element_size()
                    actual_storage_bytes += exp_avg_sq_q.numel() * exp_avg_sq_q.element_size()
                    
                    # Theoretical optimal if we packed 2 values per byte
                    theoretical_optimal_bytes = (exp_avg_q.numel() + exp_avg_sq_q.numel()) * 0.5
                    
                    actual_quantized_bytes += actual_storage_bytes
                    
                    # Scalar overhead (min, max, scale for each tensor)
                    actual_quantized_bytes += 6 * 4  # 6 scalars * 4 bytes each
        
        # Convert to GB
        estimated_fp32_gb = estimated_fp32_bytes / (1024**3)
        actual_quantized_gb = actual_quantized_bytes / (1024**3)
        
        # Calculate what we COULD achieve with proper bit packing
        theoretical_packed_bytes = total_params * 1.0 + (total_params // 1000) * 24  # 0.5 bytes per element * 2 states + scalar overhead
        theoretical_packed_gb = theoretical_packed_bytes / (1024**3)
        
        memory_saved_gb = estimated_fp32_gb - actual_quantized_gb
        theoretical_savings_gb = estimated_fp32_gb - theoretical_packed_gb
        reduction_percent = (memory_saved_gb / max(estimated_fp32_gb, 1e-10)) * 100
        theoretical_reduction_percent = (theoretical_savings_gb / max(estimated_fp32_gb, 1e-10)) * 100
        
        return {
            'total_parameters': total_params,
            'estimated_fp32_gb': estimated_fp32_gb,
            'actual_quantized_gb': actual_quantized_gb,
            'theoretical_packed_gb': theoretical_packed_gb,
            'memory_saved_gb': memory_saved_gb,
            'theoretical_savings_gb': theoretical_savings_gb,
            'memory_reduction_percent': reduction_percent,
            'theoretical_reduction_percent': theoretical_reduction_percent,
            'compression_ratio': estimated_fp32_gb / max(actual_quantized_gb, 1e-10),
            'theoretical_compression_ratio': estimated_fp32_gb / max(theoretical_packed_gb, 1e-10),
            'estimated_fp32_bytes': estimated_fp32_bytes,
            'actual_quantized_bytes': actual_quantized_bytes,
            'theoretical_packed_bytes': int(theoretical_packed_bytes)
        }