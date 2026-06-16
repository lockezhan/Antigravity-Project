#!/bin/bash
# run_and_push.sh - Run PINN benchmark with time limit and auto-push latest results to GitHub

set -e

# Default values
SCALE="large"
PRECISION="float32"
BATCH_SIZE=50000
GPUS="0,1,2,3"
EPOCHS=15000
TIME_LIMIT=13200  # Default 3 hours 40 minutes (13200 seconds)
RESUME=""
TOKEN=""
PORT=29500
OUT_DIR=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --scale) SCALE="$2"; shift ;;
        --precision) PRECISION="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --time_limit) TIME_LIMIT="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --token) TOKEN="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --out_dir) OUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️ Warning: Neither --token nor GITHUB_TOKEN environment variable is set."
    echo "Automatic push might fail if credentials are not configured on the server."
fi

# Use the token from argument or environment variable
PUSH_TOKEN=${TOKEN:-$GITHUB_TOKEN}

# Calculate number of GPUs
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT_DIR=${OUT_DIR:-"outputs_${SCALE}_${PRECISION}"}

echo "============================================="
echo "🚀 Starting Scheduled PINN Run with Time Limit"
echo "  - Scale: $SCALE"
echo "  - Precision: $PRECISION"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Visible GPUs: $GPUS (Count: $NUM_GPUS)"
echo "  - Master Port: $PORT"
echo "  - Time Limit: $TIME_LIMIT seconds (~$(echo "scale=2; $TIME_LIMIT/3600" | bc) hours)"
echo "  - Output Directory: $OUT_DIR"
echo "============================================="

export HIP_VISIBLE_DEVICES=$GPUS
export CUDA_VISIBLE_DEVICES=$GPUS

# Build main command arguments
CMD_ARGS="--scale $SCALE --precision $PRECISION --batch_size $BATCH_SIZE --epochs $EPOCHS --time_limit $TIME_LIMIT --out_dir $OUT_DIR"
if [ -n "$RESUME" ]; then
    CMD_ARGS="$CMD_ARGS --resume $RESUME"
fi

# Execute python training
if [ $NUM_GPUS -gt 1 ]; then
    echo "🔥 Running DDP Multi-GPU Mode..."
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT main.py $CMD_ARGS
else
    echo "🔥 Running Single GPU Mode..."
    python3 main.py $CMD_ARGS
fi

echo "============================================="
echo "🧹 Preparing files for Git upload..."

# Find the latest checkpoint and delete all other older checkpoints to save space
CKPT_DIR="${OUT_DIR}/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    # Find the latest checkpoint file by modification time
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Latest checkpoint found: $LATEST_CKPT"
        # Delete all other checkpoints in this folder
        for f in "$CKPT_DIR"/*.pt; do
            if [ "$f" != "$LATEST_CKPT" ]; then
                rm -f "$f"
            fi
        done
        echo "Cleaned up older checkpoints. Kept only the latest one."
    else
        echo "No checkpoints found to clean."
    fi
fi

# Stage files for git
git add "$OUT_DIR/figures/" "$OUT_DIR/profiling/"
if [ -d "$CKPT_DIR" ]; then
    git add "$CKPT_DIR"
fi

# Commit changes
COMMIT_MSG="chore: auto-save benchmark results scale=${SCALE} precision=${PRECISION} date=\$(date +'%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

# Push to GitHub
BRANCH=$(git symbolic-ref --short -q HEAD)
if [ -n "$PUSH_TOKEN" ]; then
    echo "🚀 Pushing changes to GitHub using Personal Access Token..."
    # 先进行 pull --rebase，防止并行测试推送时产生的 Non-fast-forward 冲突
    git pull --rebase "https://${PUSH_TOKEN}@github.com/lockezhan/Antigravity-Project.git" "$BRANCH" || echo "Rebase skipped or no remote changes."
    # Push using Token-embedded HTTPS URL (去掉了 --force，防止覆盖其他并行进程的提交)
    git push "https://${PUSH_TOKEN}@github.com/lockezhan/Antigravity-Project.git" "$BRANCH" -u
    echo "✅ Push successful!"
else
    echo "⚠️ GITHUB_TOKEN not set. Attempting standard git push..."
    git pull --rebase origin "$BRANCH" || echo "Rebase skipped."
    git push origin "$BRANCH" -u
fi

echo "============================================="
echo "🎉 Scheduled Run and Push complete!"
echo "============================================="
