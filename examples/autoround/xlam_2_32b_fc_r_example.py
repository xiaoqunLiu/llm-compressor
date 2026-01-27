"""
AutoRound Quantization for Salesforce xLAM-2-32b-fc-r

Hardware Requirements:
- Recommended: 2-3x A8000 (48GB each) or equivalent
- Your setup: 8x A8000 is more than sufficient

Optional Environment Variables (set before running):
- export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use specific GPUs if needed
- export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # For memory fragmentation

Memory Usage Estimate:
- Model loading: ~64GB (FP16/BF16)
- Calibration peak: ~80-100GB
- With device_map="auto", will distribute across available GPUs automatically

AutoRound vs AWQ:
- AutoRound typically provides better accuracy than AWQ
- Requires more calibration iterations (200 by default)
- May take longer to complete but produces higher quality quantized models
"""

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it with BALANCED GPU distribution
model_id = "Salesforce/xLAM-2-32b-fc-r"

# Force balanced distribution across 4 GPUs
# Each GPU gets roughly equal memory allocation
max_memory_per_gpu = {
    0: "44GB",  # Reserve some memory for activations/gradients
    1: "44GB",
    2: "44GB", 
    3: "44GB",
}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="balanced",  # Use balanced instead of auto for even distribution
    max_memory=max_memory_per_gpu,  # Explicit memory limits per GPU
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    offload_folder="offload",  # Offload to disk if needed
)
# Enable gradient checkpointing to save memory during backward pass
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Select calibration dataset.
# Reduced for memory efficiency with 32B model
# AutoRound requires more memory than AWQ due to backward pass
NUM_CALIBRATION_SAMPLES = 64  # Reduced from 128 to save memory
MAX_SEQUENCE_LENGTH = 1024     # Reduced from 2048 to save memory
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AutoRound with a group size 128
recipe = AutoRoundModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=200
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
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nModel saved to: {SAVE_DIR}")
