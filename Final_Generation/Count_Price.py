import json
import os
from pathlib import Path

GPT_4O_INPUT_1M = 2.5
GPT_4O_OUTPUT_1M = 10

GPT_4O_MINI_INPUT_1M = 0.150
GPT_4O_MINI_OUTPUT_1M = 0.6

# Fixed image input costs per file
IMAGE_INPUT_COST = 0.001913 + (0.003825 * 2)  # One initial cost plus two additional costs

def calculate_cost(input_tokens, output_tokens, input_price_per_1m, output_price_per_1m):
    """Calculate cost for given input and output tokens."""
    input_cost = (input_tokens / 1_000_000) * input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * output_price_per_1m
    return input_cost + output_cost

def process_token_usage(token_usage):
    """Process token usage data and calculate costs."""
    total_cost = 0
    costs_breakdown = {}

    # Calculate description costs (GPT-4O)
    if 'description' in token_usage:
        desc_cost = calculate_cost(
            token_usage['description']['usage']['input_tokens'],
            token_usage['description']['usage']['output_tokens'],
            GPT_4O_INPUT_1M,
            GPT_4O_OUTPUT_1M
        )
        costs_breakdown['description'] = {
            'cost': desc_cost,
            'tokens': token_usage['description']['usage']
        }
        total_cost += desc_cost

    # Calculate conversation costs (GPT-4O-mini)
    if 'conversations' in token_usage:
        conv_costs = {'multi_turn': 0, 'detailed_info': 0}
        
        for conv_type, usage in token_usage['conversations']['usage_by_type'].items():
            cost = calculate_cost(
                usage['input_tokens'],
                usage['output_tokens'],
                GPT_4O_MINI_INPUT_1M,
                GPT_4O_MINI_OUTPUT_1M
            )
            conv_costs[conv_type] = cost
            total_cost += cost
        
        costs_breakdown['conversations'] = {
            'costs': conv_costs,
            'tokens': token_usage['conversations']['usage_by_type']
        }

    # Add fixed image input cost
    costs_breakdown['image_input'] = {
        'cost': IMAGE_INPUT_COST
    }
    total_cost += IMAGE_INPUT_COST

    return {
        'total_cost': total_cost,
        'breakdown': costs_breakdown
    }

def process_dataset(directory):
    """Process all JSON files in the directory and its subdirectories."""
    total_dataset_cost = 0
    file_costs = []

    for filepath in Path(directory).rglob('*.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'token_usage' in data:
                    cost_info = process_token_usage(data['token_usage'])
                    file_costs.append({
                        'file': str(filepath),
                        'cost_info': cost_info
                    })
                    total_dataset_cost += cost_info['total_cost']
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {filepath}")
                continue

    return {
        'total_dataset_cost': total_dataset_cost,
        'file_costs': file_costs
    }

def main():
    dataset_dir = "dataset"
    results = process_dataset(dataset_dir)
    
    # Print results
    print(f"\nTotal Dataset Cost: ${results['total_dataset_cost']:.4f}")
    print("\nBreakdown by file:")
    for file_cost in results['file_costs']:
        print(f"\nFile: {file_cost['file']}")
        print(f"Cost: ${file_cost['cost_info']['total_cost']:.4f}")
        
        # Print detailed breakdown
        for component, details in file_cost['cost_info']['breakdown'].items():
            if component == 'description':
                print(f"  Description Cost: ${details['cost']:.4f}")
            elif component == 'conversations':
                print("  Conversation Costs:")
                for conv_type, cost in details['costs'].items():
                    print(f"    {conv_type}: ${cost:.4f}")
            elif component == 'image_input':
                print(f"  Image Input Cost: ${details['cost']:.4f}")

if __name__ == "__main__":
    main()