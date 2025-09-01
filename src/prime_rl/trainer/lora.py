import math
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from prime_rl.trainer.config import LoRAConfig
from prime_rl.utils.logger import get_logger


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.
    
    Implements the low-rank decomposition: ΔW = B @ A
    where A ∈ R^(rank x in_features), B ∈ R^(out_features x rank)
    
    Forward pass: y = x @ (W + ΔW).T = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters with correct shapes
        # A: (rank, in_features) - projects input down to rank
        # B: (out_features, rank) - projects rank up to output
        self.lora_A = Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = Parameter(torch.empty(base_layer.out_features, rank))
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize LoRA parameters
        self._init_parameters()
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def _init_parameters(self):
        """Initialize LoRA parameters following standard LoRA initialization."""
        # Initialize A with small random values (kaiming uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros (so initial ΔW = B @ A = 0)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_output"""
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # LoRA forward pass: x @ A.T @ B.T * scaling
        # x: (batch, seq, in_features)
        # x @ A.T: (batch, seq, in_features) @ (in_features, rank) = (batch, seq, rank)
        # (...) @ B.T: (batch, seq, rank) @ (rank, out_features) = (batch, seq, out_features)
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        return base_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base layer and return a new linear layer."""
        # Compute the low-rank update: ΔW = B @ A
        # B: (out_features, rank), A: (rank, in_features)
        # ΔW: (out_features, in_features)
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        
        # Create new linear layer with merged weights
        merged_layer = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype
        )
        
        # Merge weights: W_new = W_base + ΔW
        merged_layer.weight.data = self.base_layer.weight.data + delta_weight
        if self.base_layer.bias is not None:
            merged_layer.bias.data = self.base_layer.bias.data.clone()
        
        return merged_layer


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Get a module by its fully qualified name."""
    parts = module_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by its fully qualified name."""
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target regex patterns."""
    target_modules = []
    
    for name, module in model.named_modules():
        # Only consider nn.Linear modules
        if not isinstance(module, nn.Linear):
            continue
            
        # Check if module name matches any pattern
        for pattern in target_patterns:
            if re.match(pattern, name):
                target_modules.append(name)
                break
    
    return target_modules


def _should_keep_trainable(param_name: str, trainable_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.
    
    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)
    """
    # Check full parameter name
    for pattern in trainable_patterns:
        if re.match(pattern, param_name):
            return True
    
    # Also check module name (remove .weight, .bias, etc suffix)
    module_name = param_name.rsplit('.', 1)[0] if '.' in param_name else param_name
    for pattern in trainable_patterns:
        if re.match(pattern, module_name):
            return True
    
    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.
    
    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with trainable_modules patterns
    """
    logger = get_logger()
    frozen_params = 0
    trainable_params = 0
    total_params = 0
    trainable_details = []
    
    # First, log all modules in the model for debugging
    # logger.info("=== ALL MODULES IN MODEL ===")
    # for name, module in model.named_modules():
    #     if name:  # Skip the root module
    #         logger.info(f"Module: {name} -> {module.__class__.__name__}")
    
    logger.info("=== ALL PARAMETERS IN MODEL ===")
    for name, param in model.named_parameters():
        logger.info(f"Parameter: {name} -> shape={param.shape}, requires_grad={param.requires_grad}")
    
    # logger.info("=== TRAINABLE PATTERNS ===")
    # logger.info(f"LoRA is looking for: ['lora_A', 'lora_B']")
    # logger.info(f"Trainable modules patterns: {config.trainable_modules}")
    
    # logger.info("=== PROCESSING PARAMETERS ===")
    for name, param in model.named_parameters():
        total_params += 1
        
        # Always keep LoRA parameters trainable
        if any(lora_param in name for lora_param in ['lora_A', 'lora_B']):
            param.requires_grad = True
            trainable_params += 1
            trainable_details.append(f"{name} (LoRA)")
            logger.info(f"✓ Keeping {name} trainable (LoRA parameter)")
        # Keep specified modules fully trainable
        elif _should_keep_trainable(name, config.trainable_modules):
            param.requires_grad = True
            trainable_params += 1
            trainable_details.append(f"{name} (trainable_modules)")
            # Log which pattern matched and whether it was param or module name
            module_name = name.rsplit('.', 1)[0] if '.' in name else name
            for pattern in config.trainable_modules:
                if re.match(pattern, name):
                    logger.info(f"✓ Keeping {name} trainable (matched parameter pattern: '{pattern}')")
                    break
                elif re.match(pattern, module_name):
                    logger.info(f"✓ Keeping {name} trainable (matched module pattern: '{pattern}' on module '{module_name}')")
                    break
        # Freeze everything else
        else:
            param.requires_grad = False
            frozen_params += 1
            logger.debug(f"✗ Freezing {name}")
    
    # logger.info(f"=== FINAL SUMMARY ===")
    # logger.info(f"Parameter freezing: {frozen_params} frozen, {trainable_params} trainable, {total_params} total")
    # logger.info(f"Trainable parameters: {trainable_details}")


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """
    Apply LoRA to target modules in the model and freeze non-LoRA parameters.
    
    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    """
    logger = get_logger()
    
    if not config.enabled:
        logger.debug("LoRA is disabled, skipping LoRA application")
        return
    
    # Find target modules
    target_modules = _find_target_modules(model, config.target_modules)
    
    if not target_modules:
        logger.warning("No target modules found for LoRA. Check your target_modules regex patterns.")
        return
    
    logger.info(f"Applying LoRA to {len(target_modules)} modules: {target_modules}")
    
    # Apply LoRA to each target module (this automatically freezes base layer parameters)
    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)
        
        if not isinstance(base_module, nn.Linear):
            logger.warning(f"Module {module_name} is not nn.Linear, skipping")
            continue
        
        # Create LoRA wrapper
        lora_module = LoRALinear(
            base_layer=base_module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        
        # Replace the module
        _set_module_by_name(model, module_name, lora_module)
        
        logger.debug(f"Applied LoRA to {module_name} (rank={config.rank}, alpha={config.alpha})")
    
    # Freeze all parameters except LoRA adapters and specified trainable modules
    freeze_all_except_lora_and_specified(model, config)
    
    # Log final parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params:.2%})")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights in the model back into base layers.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Model with LoRA weights merged into base layers
    """
    logger = get_logger()
    merged_count = 0
    
    # Find and merge all LoRA modules
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # Create merged layer
            merged_layer = module.merge_weights()
            
            # Replace LoRA module with merged layer
            _set_module_by_name(model, name, merged_layer)
            merged_count += 1
    
    if merged_count > 0:
        logger.info(f"Merged {merged_count} LoRA modules back into base model")
    else:
        logger.debug("No LoRA modules found to merge")
    
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.
    
    Returns:
        Dictionary containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param.data.clone()
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA parameters into model.
    
    Args:
        model: Model with LoRA modules
        lora_state_dict: Dictionary containing LoRA parameters
    """
    logger = get_logger()
    loaded_params = 0
    
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
            loaded_params += 1
        else:
            logger.warning(f"LoRA parameter {name} not found in model")
    
    logger.debug(f"Loaded {loaded_params} LoRA parameters")