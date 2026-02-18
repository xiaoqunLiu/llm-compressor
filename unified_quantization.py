
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.autoround import AutoRoundModifier
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Unified Quantization Script")
    parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    parser.add_argument("--method", type=str, required=True, choices=["simple_ptq", "gptq", "awq", "smoothquant", "autoround"], help="Quantization method")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrachat_200k", help="Calibration dataset")
    parser.add_argument("--num_samples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--output_dir", type=str, default="./quantized_model", help="Output directory for saved model")
    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Dataset Preparation
    ds = None
    if args.method == "autoround":
         from auto_round.calib_dataset import get_dataset
         ds = get_dataset(tokenizer=tokenizer, seqlen=2048, nsamples=args.num_samples)
    else:
        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset(args.dataset, split=f"train_sft[:{args.num_samples}]")
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(sample["text"], padding=False, max_length=2048, truncation=True, add_special_tokens=False)

        ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Recipe Configuration
    recipe = None
    if args.method == "gptq":
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    elif args.method == "awq":
        recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"], duo_scaling="both")
    elif args.method == "smoothquant":
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        ]
    elif args.method == "autoround":
        recipe = AutoRoundModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=200)
    elif args.method == "simple_ptq":
        # Simple PTQ implementation: Basic W8A8 quantization without advanced calibration
        recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"], dampening_frac=0.0)

    print(f"Running quantization with method: {args.method}")
    
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=args.num_samples,
    )

    save_path = os.path.join(args.output_dir, args.method, args.model.split('/')[-1])
    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path, save_compressed=True)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
