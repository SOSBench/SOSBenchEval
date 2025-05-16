import json
import datasets

def read_dataset(path):
    if 'csv' in path:
        return datasets.load_dataset('csv', data_files=path)['train']
    else:
        return datasets.load_dataset(path)['train']

def save_json_in_shards(data, prefix, max_shard_size=1_000_000):
    """
    Write `data` (an iterable of JSON-serializable items) to multiple shards
    such that each shard's size (in bytes, as an estimate) does not exceed
    `max_shard_size`. The shards are named `prefix_0.json`, `prefix_1.json`, etc.
    
    :param data: An iterable of JSON-serializable items (e.g. list of dicts).
    :param prefix: Filename prefix for shards (e.g. '/path/to/data_shard').
    :param max_shard_size: Approximate maximum size in bytes per shard.
    """
    saved_path_list = []
    # Prepare for writing
    shard_index = 0
    current_shard_data = []
    # We initialize current_size to 2 to account for the opening and closing
    # brackets [] in a JSON array. 
    current_size = 2
    
    def write_shard(shard_idx, items):
        """Write the items to a single shard file."""
        shard_filename = f"{prefix}_{shard_idx}.json"
        with open(shard_filename, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, separators=(',', ':'), indent=4)
        return shard_filename
    
    for item in data:
        # Convert the item to its JSON string once, to measure its size.
        # We use separators=(',', ':') to keep it compact.
        item_json = json.dumps(item, ensure_ascii=False, separators=(',', ':'), indent=4).encode('utf-8')
        item_size = len(item_json)
        
        # If this single item is larger than max_shard_size, we can't write it.
        if item_size > max_shard_size:
            raise ValueError(
                f"Item size ({item_size} bytes) exceeds max_shard_size "
                f"({max_shard_size} bytes). Cannot shard this item."
            )
        
        # Check if adding this item to the current shard would exceed the limit.
        # +1 for the comma that separates items in a JSON array if not the first item.
        additional_size = item_size + (6 if current_shard_data else 0)
        
        if current_size + additional_size > max_shard_size:
            # Write out the current shard.
            shard_path = write_shard(shard_index, current_shard_data)
            saved_path_list.append(shard_path)
            
            shard_index += 1
            current_shard_data = []
            current_size = 2  # reset size (account for '[]')
        
        # Add the current item to the shard.
        current_shard_data.append(item)
        current_size += additional_size
    
    # If there's still data in the last shard, write it out.
    if current_shard_data:
        shard_path = write_shard(shard_index, current_shard_data)
        saved_path_list.append(shard_path)
    return saved_path_list