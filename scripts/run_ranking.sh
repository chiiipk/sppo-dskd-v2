#!/bin/bash
set -e

# --- PARSE INPUT ARGUMENTS ---
OUTPUT_DIR=""
PROMPTS_FILE=""
TEACHER_MODEL=""
STUDENT_MODEL=""
PAIRS=5
FRAC_LEN=0
NUM_GPUS=8

# --- ARGUMENT PARSING LOOP ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out_path) OUTPUT_DIR="$2" ;;
        --prompts) PROMPTS_FILE="$2" ;;
        --teacher_model) TEACHER_MODEL="$2" ;;
        --student_model) STUDENT_MODEL="$2" ;;   # student model
        --pairs) PAIRS="$2" ;;
        --frac_len) FRAC_LEN="$2" ;;
        --num_gpus) NUM_GPUS="$2" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift 2
done

if [ -z "$STUDENT_MODEL" ]; then
    echo "Error: Please provide --student_model"
    exit 1
fi

# --- RUN `rank.py` IN PARALLEL ---
echo "Starting parallel ranking on ${NUM_GPUS} GPUs..."
AVAILABLE_GPUS=($(seq 0 $((NUM_GPUS - 1))))

data_frac=0
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    echo "Launching ranking process for GPU ${gpu_id} (data_frac=${data_frac})..."
    CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py \
        --model "$STUDENT_MODEL" \                    
        --output_dir "$OUTPUT_DIR" \
        --prompts "$PROMPTS_FILE" \
        --pairs "$PAIRS" \
        --frac_len "$FRAC_LEN" \
        --data_frac "$data_frac" \
        --gpu "$gpu_id" \
        --use_teacher_llm \
        --teacher_model "$TEACHER_MODEL" \
        --batch_size 8 \
        --bt_conversion_method bradley_terry_mle > "logs/rank_log_gpu_${gpu_id}.txt" 2>&1 &
    ((data_frac++))
done

wait
echo "All ranking processes have finished."
