set -e

# --- PARSE INPUT ARGUMENTS ---
MODEL_PATH=""
PROMPTS_FILE=""
OUTPUT_DIR=""
PAIRS=5
FRAC_LEN=0
NUM_GPUS=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift ;;
        --prompts) PROMPTS_FILE="$2"; shift ;;
        --out_path) OUTPUT_DIR="$2"; shift ;;
        --pairs) PAIRS="$2"; shift ;;
        --frac_len) FRAC_LEN="$2"; shift ;;
        --num_gpus) NUM_GPUS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- RUN `generate_student.py` IN PARALLEL ---
echo "Starting parallel generation on ${NUM_GPUS} GPUs..."
AVAILABLE_GPUS=($(seq 0 $((NUM_GPUS - 1))))

data_frac=0
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    echo "Launching generation process for GPU ${gpu_id} (data_frac=${data_frac})..."
    CUDA_VISIBLE_DEVICES=$gpu_id python3 generate_student.py \
        --model "$MODEL_PATH" \
        --prompts "$PROMPTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pairs "$PAIRS" \
        --frac_len "$FRAC_LEN" \
        --data_frac "$data_frac" \
        --batch_size 16 \
        --maxlen 512 > "logs/gen_log_gpu_${gpu_id}.txt" 2>&1 & # Redirect logs to a file
    ((data_frac++))
done

wait # Wait for all background generation processes to complete
echo "All generation processes have finished."

# --- COMBINE THE RESULTS ---
echo "Combining the generated result files..."
python3 scripts/combine_generate.py \
    --output_dir "$OUTPUT_DIR" \
    --pairs "$PAIRS" \
    --numgpu "$NUM_GPUS"
echo "Combining complete."