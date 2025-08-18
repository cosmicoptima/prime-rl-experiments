#!/usr/bin/env python3
"""Script to inspect rollout .pt files and display model outputs."""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer

def inspect_rollout(file_path: str, num_samples: int = 3):
    """Load and display contents of a rollout .pt file."""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT")
        
        data = torch.load(file_path, map_location='cpu')
        print(f"\n=== Rollout: {file_path} ===")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"List with {len(data)} items")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                if hasattr(data[0], 'keys'):
                    print(f"First item keys: {list(data[0].keys())}")
                
                # Show sample data
                print(f"\n=== Sample Conversations (first {num_samples}) ===")
                for i in range(min(num_samples, len(data))):
                    item = data[i]
                    if 'input_ids' in item:
                        tokens = item['input_ids'][0]  # Remove batch dimension
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                        advantage = item.get('advantages', [None])[0]
                        
                        print(f"\n--- Sample {i+1} ---")
                        print(f"Text: {text}")
                        if advantage is not None:
                            print(f"Advantage: {advantage[0]:.4f}")
                        print("-" * 80)
        
        elif isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
            # Original dict handling code here
        
        else:
            print(f"Unexpected data type: {type(data)}")
            print(f"Data: {data}")
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    if len(sys.argv) > 1:
        # Inspect specific file
        inspect_rollout(sys.argv[1])
    else:
        # Look at key transition points
        rollout_dir = Path("outputs/rollouts")
        if not rollout_dir.exists():
            print("No rollouts directory found")
            return
        
        # Key transition steps based on reward jumps
        key_steps = [0, 4, 5, 7, 8, 12]  # before/after major reward increases
        
        for step in key_steps:
            step_file = rollout_dir / f"step_{step}" / "rank_0.pt"
            if step_file.exists():
                print(f"\n{'='*60}")
                print(f"STEP {step} (looking for breakthrough moments)")
                print(f"{'='*60}")
                inspect_rollout(str(step_file))
            else:
                print(f"Step {step} not found")

if __name__ == "__main__":
    main()