#!/usr/bin/env python3
"""
Scan all quantized models and report their sizes in a formatted table,
including estimated original FP16 model size from config.json.
Usage: python3 model_size_report.py [--base_dir ./quantized_model] [--csv output.csv]
"""

import os
import json
import argparse
from collections import defaultdict


def get_dir_size_gb(path):
    """Calculate total size of a directory in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def estimate_original_size_gb(config_path):
    """Estimate original FP16 model size from config.json parameters."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        hs = cfg.get("hidden_size", 0)
        nl = cfg.get("num_hidden_layers", 0)
        ih = cfg.get("intermediate_size", 0)
        vs = cfg.get("vocab_size", 0)
        nah = cfg.get("num_attention_heads", 0)
        nkv = cfg.get("num_key_value_heads", nah)
        hd = hs // nah if nah else 0

        # Attention: Q, K, V projections + output projection
        attn_params = (nah * hd * hs) + 2 * (nkv * hd * hs) + (nah * hd * hs)
        # MLP: gate + up + down projections
        mlp_params = 3 * hs * ih
        # Embeddings + LM head
        embed_params = vs * hs
        # Total
        total_params = embed_params + nl * (attn_params + mlp_params)
        # FP16: 2 bytes per parameter
        size_gb = (total_params * 2) / (1024 ** 3)
        params_b = total_params / 1e9
        return size_gb, params_b
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Report quantized model sizes")
    parser.add_argument("--base_dir", type=str,
                        default="./quantized_model",
                        help="Base directory containing quantized models")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional: save results to CSV file")
    args = parser.parse_args()

    base_dir = args.base_dir

    # Collect all data: {model_name: {method: size_gb}}
    data = defaultdict(dict)
    methods_set = set()
    original_sizes = {}  # {model_name: (size_gb, params_b)}

    for method in sorted(os.listdir(base_dir)):
        method_dir = os.path.join(base_dir, method)
        if not os.path.isdir(method_dir):
            continue
        methods_set.add(method)
        for model_name in sorted(os.listdir(method_dir)):
            model_dir = os.path.join(method_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            size_gb = get_dir_size_gb(model_dir)
            data[model_name][method] = size_gb

            # Estimate original size from config.json (only need to do once per model)
            if model_name not in original_sizes:
                config_path = os.path.join(model_dir, "config.json")
                if os.path.exists(config_path):
                    orig_size, params_b = estimate_original_size_gb(config_path)
                    original_sizes[model_name] = (orig_size, params_b)

    methods = sorted(methods_set)
    models = sorted(data.keys())

    if not models:
        print("No quantized models found.")
        return

    # Print table
    model_col_width = max(len("Model"), max(len(m) for m in models)) + 2
    method_col_width = 14
    orig_col_width = 16

    # Header
    header = f"{'Model':<{model_col_width}}"
    header += f" | {'Original(FP16)':>{orig_col_width}}"
    for m in methods:
        header += f" | {m:>{method_col_width}}"
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  QUANTIZED MODEL SIZE REPORT (GB)")
    print("=" * len(header))
    print(header)
    print(separator)

    for model in models:
        row = f"{model:<{model_col_width}}"
        # Original size
        if model in original_sizes and original_sizes[model][0] is not None:
            orig_size, params_b = original_sizes[model]
            row += f" | {f'{orig_size:.2f} ({params_b:.1f}B)':>{orig_col_width}}"
        else:
            row += f" | {'N/A':>{orig_col_width}}"
        # Quantized sizes
        for method in methods:
            if method in data[model]:
                size = data[model][method]
                row += f" | {size:>{method_col_width}.2f}"
            else:
                row += f" | {'N/A':>{method_col_width}}"
        print(row)

    print(separator)
    print()

    # Save CSV if requested
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Params(B)", "Original_FP16(GB)"] + methods)
            for model in models:
                if model in original_sizes and original_sizes[model][0] is not None:
                    orig_size, params_b = original_sizes[model]
                    row = [model, f"{params_b:.1f}", f"{orig_size:.2f}"]
                else:
                    row = [model, "N/A", "N/A"]
                for method in methods:
                    if method in data[model]:
                        row.append(f"{data[model][method]:.2f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
