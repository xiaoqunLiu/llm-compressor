"""
AutoRound Quantization for Salesforce xLAM-2-32b-fc-r
ULTRA LOW MEMORY VERSION

This version is optimized for maximum memory efficiency:
- Minimal calibration samples (32)
- Shorter sequence length (512)
- Gradient checkpointing enabled
- Reduced iterations (100 instead of 200)

Hardware Requirements:
- Should work with 2-3x A8000 (48GB each)
- Trade-off: Slightly lower accuracy for memory efficiency

Optional Environment Variables (set before running):
- export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use specific GPUs if needed
- export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # For memory fragmentation
"""

import os
# Suppress CuBLAS deterministic warnings (optional, doesn't affect functionality)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "Salesforce/xLAM-2-32b-fc-r"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
# Enable gradient checkpointing to save memory during backward pass
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Select calibration dataset - ULTRA LOW MEMORY SETTINGS
NUM_CALIBRATION_SAMPLES = 32   # Minimal samples
MAX_SEQUENCE_LENGTH = 512       # Short sequences
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
# Reduced iterations for faster completion and lower memory
recipe = AutoRoundModifier(
    targets="Linear", 
    scheme="W4A16", 
    ignore=["lm_head"], 
    iters=100  # Reduced from 200 for memory efficiency
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound-LowMem"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nModel saved to: {SAVE_DIR}")
