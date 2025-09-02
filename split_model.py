from safetensors.torch import save_file, load_file, safe_open
import torch
import sys

# Load your large file
state_dict = torch.load(sys.argv[1])  # or load_file() for safetensors

# Split into shards
shard_size = 10 * 1024**3  # 10GB in bytes
shards = {}
current_shard = {}
current_size = 0
shard_idx = 1

# First pass to count total shards
total_shards = 1
temp_size = 0
for key, tensor in state_dict.items():
    tensor_size = tensor.numel() * tensor.element_size()
    if temp_size + tensor_size > shard_size and temp_size > 0:
        total_shards += 1
        temp_size = 0
    temp_size += tensor_size

# Now split with known total
for key, tensor in state_dict.items():
    tensor_size = tensor.numel() * tensor.element_size()
    
    if current_size + tensor_size > shard_size and current_shard:
        # Save current shard
        save_file(current_shard, f"{sys.argv[2]}/model-{shard_idx:05d}-of-{total_shards:05d}.safetensors")
        shards[f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"] = list(current_shard.keys())
        current_shard = {}
        current_size = 0
        shard_idx += 1
    
    current_shard[key] = tensor
    current_size += tensor_size

# Save last shard
if current_shard:
    save_file(current_shard, f"{sys.argv[2]}/model-{shard_idx:05d}-of-{total_shards:05d}.safetensors")
    shards[f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"] = list(current_shard.keys())

# Create index file
import json
total_shards = shard_idx
for filename in list(shards.keys()):
    new_name = filename.replace("NNNNN", f"{total_shards:05d}")
    shards[new_name] = shards.pop(filename)

index = {
    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
    "weight_map": {k: shard for shard, keys in shards.items() for k in keys}
}

with open(f"{sys.argv[2]}/model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)