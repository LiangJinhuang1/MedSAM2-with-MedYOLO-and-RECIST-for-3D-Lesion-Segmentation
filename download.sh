#!/bin/sh
# Script to download FLARE checkpoint

# Use either wget or curl to download the checkpoints
if command -v wget > /dev/null 2>&1; then
    CMD_WGET="wget -P"
elif command -v curl > /dev/null 2>&1; then
    CMD_CURL="curl -L -o"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# --- FLARE Checkpoint ---

FLARE_DIR="FLARE_results/checkpoints"
mkdir -p "$FLARE_DIR"

FLARE_MODEL_URL="https://huggingface.co/JinLemon/Seminar-Flare-2025/resolve/main/checkpoint.pt"
FLARE_MODEL_NAME="checkpoint.pt"

echo "Downloading ${FLARE_MODEL_NAME}..."
if [ -n "$CMD_CURL" ]; then
    $CMD_CURL "${FLARE_DIR}/${FLARE_MODEL_NAME}" "$FLARE_MODEL_URL" || { echo "Failed to download checkpoint from $FLARE_MODEL_URL"; exit 1; }
else
    $CMD_WGET "$FLARE_DIR" "$FLARE_MODEL_URL" || { echo "Failed to download checkpoint from $FLARE_MODEL_URL"; exit 1; }
fi
echo "FLARE checkpoint has been downloaded successfully to the '${FLARE_DIR}' directory."