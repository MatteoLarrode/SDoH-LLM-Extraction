#!/bin/bash

# SDoH Evaluation Script for Multiple Models
# Run 100 sentences with few-shot basic prompts

echo "Starting SDoH evaluation for multiple models..."
echo "Date: $(date)"
echo "Using 100 sentences with few-shot basic prompts"
echo "=================================="

# Define models to test
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct" 
    "microsoft/Phi-4-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# Common parameters
PROMPT_TYPE="five_shot_basic"
LEVEL=1
SAMPLE_SIZE=100
ANNOTATION_DATA="data/raw/BRC-Data/annotated_BRC_referrals.csv"
OUTPUT_DIR="results/annotation_evaluation"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=================================="
    echo "Starting evaluation for: $MODEL"
    echo "Time: $(date)"
    echo "=================================="
    
    # Run the evaluation
    python scripts/evaluate_on_annotations.py \
        --model_name "$MODEL" \
        --prompt_type "$PROMPT_TYPE" \
        --level $LEVEL \
        --annotation_data "$ANNOTATION_DATA" \
        --sample_size $SAMPLE_SIZE \
        --output_dir "$OUTPUT_DIR" \
        --random_seed 42
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for $MODEL"
    else
        echo "✗ Failed evaluation for $MODEL"
    fi
    
    # Small pause between models to be courteous
    echo "Waiting 10 seconds before next model..."
    sleep 10
done

echo ""
echo "=================================="
echo "All evaluations completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=================================="

# Show summary of generated files
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*$(date +%Y%m%d)* 2>/dev/null || echo "No files found for today"