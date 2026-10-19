#!/usr/bin/env bash

set -uo pipefail

mkdir -p logs

# Complete confirmation batch: 2 models x 4 augmentation settings x 3 seeds = 24 jobs.
# Models: ResNet18 / ResNet50.
# Augmentations: None, Rotation +/-15, HorizontalFlip p=0.5,
#                Rotation +/-15 + HorizontalFlip p=0.5.
# Fixed: Normalize off, LR=1e-5, wd=0.9, batch=128, 30 epochs.
# Format: mode seed augmentation degree flip norm
JOBS=(
    # # ResNet18, no augmentation.
    "1 42 None 0 0.0 0"
    # "1 123 None 0 0.0 0"
    # "1 2026 None 0 0.0 0"

    # # ResNet18, rotation +/-15 degrees.
    # "1 42 RandomRotation 15 0.0 0"
    # "1 123 RandomRotation 15 0.0 0"
    # "1 2026 RandomRotation 15 0.0 0"

    # # ResNet18, horizontal flip p=0.5.
    # "1 42 RandomHorizontalFlip 0 0.5 0"
    # "1 123 RandomHorizontalFlip 0 0.5 0"
    # "1 2026 RandomHorizontalFlip 0 0.5 0"

    # # ResNet18, rotation +/-15 degrees + horizontal flip p=0.5.
    # "1 42 RandomRotate-Flip 15 0.5 0"
    # "1 123 RandomRotate-Flip 15 0.5 0"
    # "1 2026 RandomRotate-Flip 15 0.5 0"

    # # ResNet50, no augmentation.
    "2 42 None 0 0.0 0"
    # "2 123 None 0 0.0 0"
    # "2 2026 None 0 0.0 0"

    # # ResNet50, rotation +/-15 degrees.
    # "2 42 RandomRotation 15 0.0 0"
    # "2 123 RandomRotation 15 0.0 0"
    # "2 2026 RandomRotation 15 0.0 0"

    # # ResNet50, horizontal flip p=0.5.
    # "2 42 RandomHorizontalFlip 0 0.5 0"
    # "2 123 RandomHorizontalFlip 0 0.5 0"
    # "2 2026 RandomHorizontalFlip 0 0.5 0"

    # # ResNet50, rotation +/-15 degrees + horizontal flip p=0.5.
    # "2 42 RandomRotate-Flip 15 0.5 0"
    # "2 123 RandomRotate-Flip 15 0.5 0"
    # "2 2026 RandomRotate-Flip 15 0.5 0"
)

QUEUE_FILE=$(mktemp)
LOCK_FILE=$(mktemp)

echo 0 > "$QUEUE_FILE"
trap 'rm -f "$QUEUE_FILE" "$LOCK_FILE"' EXIT

claim_job() {
    local index

    flock -x 9
    index=$(cat "$QUEUE_FILE")
    if (( index >= ${#JOBS[@]} )); then
        flock -u 9
        return 1
    fi

    NEXT_JOB="${JOBS[$index]}"
    echo $((index + 1)) > "$QUEUE_FILE"
    flock -u 9
    return 0
}

worker() {
    local gpu_id="$1"
    local failed=0
    exec 9>"$LOCK_FILE"

    while claim_job; do
        local mode seed aug degree flip norm
        read -r mode seed aug degree flip norm <<< "$NEXT_JOB"
        local norm_args=()
        if [[ "$norm" == 1 ]]; then
            norm_args=(--norm)
        fi

        local model
        case "$mode" in
            1) model="18" ;;
            2) model="50" ;;
            *)
                echo "[GPU $gpu_id] Invalid mode: $mode" >&2
                failed=1
                continue
                ;;
        esac

        local timestamp
        timestamp=$(date +%y-%m-%d_%H%M%S)
        local job_name="${aug}_${model}_${seed}_deg${degree}_flip${flip}_norm${norm}_lr1e-5_wd0.9_bs128_${timestamp}"
        local log="logs/${job_name}.log"

        echo "[GPU $gpu_id] START: $job_name"
        echo "mode=$mode seed=$seed aug=$aug degree=$degree flip=$flip norm=$norm lr=1e-5 wd=0.9 batch_size=128 num_epochs=30" > "$log"

        if CUDA_VISIBLE_DEVICES="$gpu_id" \
            python -u train.py \
                --mode "$mode" \
                --seed "$seed" \
                --aug "$aug" \
                --degree "$degree" \
                --flip "$flip" \
                "${norm_args[@]}" \
                --lr 1e-5 \
                --wd 0.9 \
                --batch_size 128 \
                --num_epochs 30 \
                --worker 3 \
                >> "$log" 2>&1
        then
            echo "[GPU $gpu_id] DONE: $job_name"
        else
            status=$?
            echo "[GPU $gpu_id] FAILED ($status): $job_name" >&2
            echo "  Log: $log" >&2
            failed=1
        fi
    done

    exec 9>&-
    echo "[GPU $gpu_id] Queue empty."
    return "$failed"
}

worker 0 &
PID0=$!
worker 1 &
PID1=$!

result=0
if ! wait "$PID0"; then
    result=1
fi
if ! wait "$PID1"; then
    result=1
fi

if (( result == 0 )); then
    echo "All jobs completed successfully."
else
    echo "Some jobs failed. Check logs/." >&2
fi
exit "$result"
