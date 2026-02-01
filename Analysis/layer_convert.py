import ijson
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

json_file_prefix = 'logs/deepseek'
path_to_dir = 'logs/deepseek'

BATCH_SIZE = 1000

# Helper function stays the same
def flatten_if_needed(logits):
    if not logits:
        return logits
    if isinstance(logits[0], list) and len(logits[0]) > 0 and isinstance(logits[0][0], list):
        flattened = []
        for sublist in logits:
            flattened.extend(sublist)
        return flattened
    return logits

datasets = ['wikitext', 'lambada', 'hellaswag']

for dataset in datasets:
    json_file = f'{json_file_prefix}_{dataset}_internal_routing.json'
    current_dataset = dataset
    output_dir = os.path.join(path_to_dir, current_dataset)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"Directory {output_dir} already exists. Skipping dataset {dataset}.")
        continue

    # --- KEY FIX: Initialize buffer and writers INSIDE the loop ---
    buffer = {} 
    writers = {} 

    # We also need to redefine flush_buffer inside or pass these vars to it.
    # To keep it simple, we can just define the flush logic here or update global vars.
    # Since your flush_buffer uses the global 'buffer' and 'writers', updating them 
    # above works, BUT we must ensure the function sees the new empty dicts.
    
    print(f"Processing {dataset}...")

    # We redefine flush_buffer to ensure it captures the current loop's writers/buffer
    # or simply rely on Python's scoping if they are truly global. 
    # However, a cleaner way is to keep the logic inline or pass args.
    # For minimal changes to your style, we'll keep using the variables we just reset.

    def flush_buffer_current():
        for layer_id, data in buffer.items():
            if not data['logits']:
                continue
            
            df = pd.DataFrame(data['logits'])
            df.columns = [f'logit_{i}' for i in range(df.shape[1])]
            df.insert(0, 'layer', data['layer'])
            
            table = pa.Table.from_pandas(df, preserve_index=False)
            file_name = os.path.join(output_dir, f"layer_{layer_id}.parquet")
            
            if layer_id not in writers:
                writers[layer_id] = pq.ParquetWriter(file_name, table.schema)
                
            try:
                writers[layer_id].write_table(table)
            except ValueError as e:
                print(f"Schema error layer {layer_id}")
                raise e
                
            buffer[layer_id] = {'layer': [], 'logits': []}

    try:
        with open(json_file, 'rb') as f:
            parser = ijson.items(f, 'samples.item', use_float=True)
            
            for i, sample in enumerate(parser):
                for layer_data in sample.get('layers', []):
                    layer_id = layer_data['layer']
                    raw_logits = layer_data['router_logits_sample']
                    logits_list_of_lists = flatten_if_needed(raw_logits)
                    
                    if layer_id not in buffer:
                        buffer[layer_id] = {'layer': [], 'logits': []}
                    
                    num_new_rows = len(logits_list_of_lists)
                    buffer[layer_id]['layer'].extend([layer_id] * num_new_rows)
                    buffer[layer_id]['logits'].extend(logits_list_of_lists)
                
                if (i + 1) % BATCH_SIZE == 0:
                    print(f"  Processed {i + 1} samples...")
                    flush_buffer_current()

        flush_buffer_current()

    except FileNotFoundError:
        print(f"  File {json_file} not found. Skipping.")
    except Exception as e:
        print(f"  Error processing {dataset}: {e}")
        # Consider whether to raise e or continue to next dataset
        raise e 
    finally:
        # Close only the writers for the CURRENT dataset
        for writer in writers.values():
            writer.close()
        print(f"Finished {dataset}. Writers closed.")