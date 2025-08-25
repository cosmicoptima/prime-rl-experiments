#!/usr/bin/env python3
"""Test the celeste_0000 environment to debug reward computation."""

import sys
sys.path.append('environments')

from vf_celeste_0000.vf_celeste_0000 import load_environment

# Load the environment
env = load_environment()

# Test different completion formats
test_cases = [
    # Message list format (what verifiers likely sends)
    [{"role": "assistant", "content": "Hello World! This is a TEST."}],
    
    # String format (just in case)
    "Hello World! This is a TEST.",
    
    # Multiple messages
    [
        {"role": "user", "content": "Test prompt"},
        {"role": "assistant", "content": "HELLO WORLD IN ALL CAPS"}
    ],
    
    # Empty response
    [{"role": "assistant", "content": ""}],
    
    # No caps
    [{"role": "assistant", "content": "hello world no caps here"}],
    
    # All caps
    [{"role": "assistant", "content": "HELLO WORLD ALL CAPS HERE"}],
]

print("Testing reward function with different inputs:\n")
print("=" * 60)

for i, completion in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    print(f"Input type: {type(completion)}")
    print(f"Input: {completion}")
    
    try:
        # Get the reward function from the rubric
        reward_func = env.rubric.funcs[0]
        
        # Call it with empty answer dict (like your dataset has)
        reward = reward_func(completion, {})
        
        print(f"Reward: {reward:.4f}")
        
        # Calculate expected reward manually for verification
        if isinstance(completion, list):
            text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict) and m.get("role") == "assistant")
        else:
            text = str(completion)
        
        letters = [c for c in text if c.isalpha()]
        caps = [c for c in text if c.isupper()]
        expected = len(caps) / len(letters) if letters else 0.0
        print(f"Expected: {expected:.4f} ({len(caps)}/{len(letters)} caps)")
        
        if abs(reward - expected) > 0.001:
            print("WARNING: Mismatch between actual and expected!")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("-" * 40)

print("\n" + "=" * 60)
print("\nIf rewards are working here but not in training, the issue")
print("might be in how verifiers calls the reward function.")