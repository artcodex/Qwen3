import json
import os
from safetensors.torch import load_file, save_file

input_file = "model-00002-of-00002.safetensors"
output_dir = "split_model_002"
num_shards = 4  # Change this to how many pieces you want

os.makedirs(output_dir, exist_ok=True)
tensors = load_file(input_file)
keys = list(tensors.keys())

# Split keys into roughly equal chunks
chunk_size = (len(keys) + num_shards - 1) // num_shards
weight_map = {}
total_size = 0

for i in range(num_shards):
    chunk_keys = keys[i * chunk_size : (i + 1) * chunk_size]
    if not chunk_keys:
        continue
    
    shard_tensors = {k: tensors[k] for k in chunk_keys}
    shard_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
    shard_path = os.path.join(output_dir, shard_name)
    
    save_file(shard_tensors, shard_path, metadata={"format": "pt"})
    
    for k in chunk_keys:
        weight_map[k] = shard_name
        total_size += tensors[k].numel() * tensors[k].element_size()

# Create the index.json file
index_data = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map
}

with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index_data, f, indent=2)

print(f"Successfully split into {num_shards} shards inside '{output_dir}/'")