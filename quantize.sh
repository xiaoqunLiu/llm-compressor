#!/bin/bash

MODEL_ID="tiny-random-Llama-3" 
METHODS=("gptq" "awq" "simple_ptq") 

DATASET="HuggingFaceH4/ultrachat_200k"
NUM_SAMPLES=16 
OUTPUT_BASE="./quantized_model"

echo "======================================"
echo "Quantization Start"
echo "Model: $MODEL_ID"
echo "Methods: ${METHODS[*]}"
echo "======================================"

mkdir -p $OUTPUT_BASE

for METHOD in "${METHODS[@]}"   
do
    echo "--------------------------------------"
    echo "Start $METHOD"
    echo "--------------------------------------"

    python3 unified_quantization.py \
        --model "$MODEL_ID" \
        --method "$METHOD" \
        --dataset "$DATASET" \
        --num_samples $NUM_SAMPLES

    if [ $? -eq 0 ]; then
        echo "✅ $METHOD Done"
    else
        echo "❌ $METHOD Failed"
    fi
done

echo "======================================"
echo "All tasks completed!"
ls -R $OUTPUT_BASE