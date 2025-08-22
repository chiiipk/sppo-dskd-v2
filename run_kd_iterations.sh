#!/bin/bash
set -e
set -x

# --- GENERAL CONFIGURATION ---
# Get the path to the initial student model from the first command-line argument
INITIAL_STUDENT_MODEL_PATH=$1
if [ -z "$INITIAL_STUDENT_MODEL_PATH" ]; then
    echo "Error: Please provide the path to the initial student model."
    echo "Usage: ./run_kd_iterations.sh /path/to/your/gpt2"
    exit 1
fi

ITERATION_COUNT=3
PROMPT_FILE="d:/Python/GenAI/DSKD/data/dolly/train.jsonl"
TEACHER_MODEL="Qwen/Qwen1.5-1.8B"
NUM_PAIRS=5

# GPU configuration and data splitting
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
# Assuming the dataset has around 20,800 prompts for compatibility with other scripts.
# Change this number if your dataset size is different.
TOTAL_PROMPTS=15000
FRAC_LEN=$((TOTAL_PROMPTS / NUM_GPUS))


# --- MAIN LOOP ---
for i in $(seq 1 $ITERATION_COUNT); do
    echo "================================================================"
    echo "Starting Knowledge Distillation - Iteration ${i}"
    echo "================================================================"

    # Determine the input model for the current iteration
    if [ "$i" -eq 1 ]; then
        MODEL_TO_USE=$INITIAL_STUDENT_MODEL_PATH
    else
        MODEL_TO_USE=$MODEL_OUTPUT_DIR
    fi

    # Set the output directory names for this iteration
    DATA_OUTPUT_DIR="generated/kd-gpt2-qwen-dolly-iter${i}"
    MODEL_OUTPUT_DIR="checkpoints/gpt2-kd-qwen-dolly-iter${i}"
    DATASET_NAME="synthetic_data_kd_gpt2_qwen-dolly_iter${i}_score"

    # --- STEP 1: GENERATE RESPONSES FROM STUDENT MODEL (IN PARALLEL) ---
    echo "Step 1: Generating responses from student model '${MODEL_TO_USE}'..."
    bash scripts/run_student_generation.sh \
        --model "$MODEL_TO_USE" \
        --prompts "$PROMPT_FILE" \
        --out_path "$DATA_OUTPUT_DIR" \
        --pairs $NUM_PAIRS \
        --frac_len $FRAC_LEN \
        --num_gpus $NUM_GPUS

    # --- STEP 2: RANK RESPONSES WITH TEACHER MODEL (IN PARALLEL) ---
    echo "Step 2: Ranking responses with teacher model '${TEACHER_MODEL}'..."
    bash scripts/run_ranking.sh \
        --out_path "$DATA_OUTPUT_DIR" \
        --prompts "$PROMPT_FILE" \
        --teacher_model "$TEACHER_MODEL" \
        --pairs $NUM_PAIRS \
        --frac_len $FRAC_LEN \
        --num_gpus $NUM_GPUS

    # --- STEP 3: COMPUTE PROBABILITIES AND PREPARE DATASET ---
    echo "Step 3: Computing probabilities and finalizing the dataset..."
    python3 scripts/compute_prob.py \
        --output_dir "$DATA_OUTPUT_DIR" \
        --prompts "$PROMPT_FILE" \
        --pairs $NUM_PAIRS \
        --frac_len $FRAC_LEN \
        --num_gpu $NUM_GPUS \
        --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")"

    # --- STEP 4: RETRAIN THE STUDENT MODEL ---
    echo "Step 4: Retraining the student model using SPPO..."
    bash scripts/pipeline.sh \
        --model "$MODEL_TO_USE" \
        --dataset "$DATASET_NAME" \
        --output_dir "$MODEL_OUTPUT_DIR" \
        --learning_rate 1.0e-6 \
        --batch_size 16

done

echo "================================================================"
echo "Knowledge Distillation process completed successfully!"
echo "================================================================"```



