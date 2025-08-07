import json
import pandas as pd
from pathlib import Path

def filter_json_by_csv(csv_path, json_path, output_path):
    """
    Filter JSON file to only keep designs mentioned in the CSV file.
    
    Args:
        csv_path: Path to the CSV file containing selected designs
        json_path: Path to the input JSON file
        output_path: Path to save the filtered JSON file
    """
    
    # Read the CSV file to get the list of selected algorithms with their sources
    df = pd.read_csv(csv_path)
    selected_pairs = set(zip(df['algo_name'], df['source_name']))
    selected_algos = set(df['algo_name'].tolist())
    
    print(f"Found {len(selected_pairs)} selected algorithm-source pairs in CSV:")
    print(f"Found {len(selected_algos)} unique algorithm names in CSV:")
    for algo in sorted(selected_algos):
        sources = [source for a, source in selected_pairs if a == algo]
        print(f"  - {algo} (sources: {', '.join(sources)})")
    
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nOriginal JSON contains {len(data)} entries")
    print(f"JSON data type: {type(data)}")
    
    # Handle both list and dictionary structures
    if isinstance(data, list):
        # If data is a list, filter based on a key field in each item
        filtered_data = []
        matched_count = 0
        
        # First, let's examine the structure of the first few items
        if data:
            print(f"\nSample item structure: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        
        for item in data:
            if isinstance(item, dict):
                # Look for algorithm name and source
                algo_name = None
                source_name = None
                
                # Look for algorithm name
                for key in ['name', 'algo_name', 'algorithm', 'kernel_name', 'design_name']:
                    if key in item:
                        algo_name = item[key]
                        break
                
                # Look for source name
                for key in ['source', 'source_name', 'dataset', 'benchmark']:
                    if key in item:
                        source_name = item[key]
                        break
                
                # Check if this specific combination is in our selected pairs
                if algo_name and source_name and (algo_name, source_name) in selected_pairs:
                    filtered_data.append(item)
                    matched_count += 1
                    print(f"✓ Keeping: {algo_name} from {source_name}")
                elif algo_name and source_name:
                    print(f"✗ Removing: {algo_name} from {source_name} (not in selected pairs)")
                elif algo_name:
                    print(f"✗ Removing: {algo_name} (no source found)")
                else:
                    print(f"✗ Removing item (no identifiable name): {list(item.keys())[:3]}...")
            else:
                print(f"✗ Removing non-dict item: {item}")
                
    elif isinstance(data, dict):
        # For dictionary structure, we need to check if items have source information
        filtered_data = {}
        matched_count = 0
        
        for key, value in data.items():
            if key in selected_algos:
                # If it's a dictionary value, check for source information
                if isinstance(value, dict):
                    source_name = None
                    for source_key in ['source', 'source_name', 'dataset', 'benchmark']:
                        if source_key in value:
                            source_name = value[source_key]
                            break
                    
                    if source_name and (key, source_name) in selected_pairs:
                        filtered_data[key] = value
                        matched_count += 1
                        print(f"✓ Keeping: {key} from {source_name}")
                    elif source_name:
                        print(f"✗ Removing: {key} from {source_name} (not in selected pairs)")
                    else:
                        # If no source info, check if any source is acceptable for this algo
                        algo_sources = [source for algo, source in selected_pairs if algo == key]
                        if algo_sources:
                            print(f"? Keeping: {key} (no source info, but algo is selected)")
                            filtered_data[key] = value
                            matched_count += 1
                        else:
                            print(f"✗ Removing: {key} (no source info)")
                else:
                    # Simple value, check if algo is in selected list
                    algo_sources = [source for algo, source in selected_pairs if algo == key]
                    if algo_sources:
                        filtered_data[key] = value
                        matched_count += 1
                        print(f"✓ Keeping: {key} (simple value)")
                    else:
                        print(f"✗ Removing: {key}")
            else:
                print(f"✗ Removing: {key}")
    else:
        raise ValueError(f"Unsupported JSON structure: {type(data)}")
    
    print(f"\nFiltered JSON contains {len(filtered_data)} entries")
    print(f"Matched {matched_count} algorithms from CSV")
    print(f"Expected exactly {len(selected_pairs)} entries")
    
    if len(filtered_data) != len(selected_pairs):
        print(f"WARNING: Expected {len(selected_pairs)} entries but got {len(filtered_data)}")
    
    # Check for missing algorithm-source pairs
    if isinstance(data, list):
        found_pairs = set()
        for item in data:
            if isinstance(item, dict):
                algo_name = None
                source_name = None
                
                for key in ['name', 'algo_name', 'algorithm', 'kernel_name', 'design_name']:
                    if key in item:
                        algo_name = item[key]
                        break
                
                for key in ['source', 'source_name', 'dataset', 'benchmark']:
                    if key in item:
                        source_name = item[key]
                        break
                
                if algo_name and source_name:
                    found_pairs.add((algo_name, source_name))
        
        missing_pairs = selected_pairs - found_pairs
    else:
        found_pairs = set()
        for key, value in data.items():
            if isinstance(value, dict):
                source_name = None
                for source_key in ['source', 'source_name', 'dataset', 'benchmark']:
                    if source_key in value:
                        source_name = value[source_key]
                        break
                if source_name:
                    found_pairs.add((key, source_name))
        
        missing_pairs = selected_pairs - found_pairs
    
    if missing_pairs:
        print(f"\nWarning: {len(missing_pairs)} algorithm-source pairs from CSV not found in JSON:")
        for algo, source in sorted(missing_pairs):
            print(f"  - {algo} from {source}")
    
    # Save the filtered JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltered JSON saved to: {output_path}")
    
    return filtered_data

if __name__ == "__main__":
    # Define paths
    csv_path = "after_gpt_review_designs_summary.csv"
    json_path = "/home/user/zedongpeng/workspace/cpp-benchmark-deduplicator/data/data_of_designs_kernels.json"
    output_path = "after_gpt_review.json"
    
    # Filter the JSON
    filtered_data = filter_json_by_csv(csv_path, json_path, output_path) 