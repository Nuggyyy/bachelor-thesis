#!/usr/bin/env bash
set -euo pipefail

# run_extract_all.sh
# Calls local/extract_logs.py for every combination of model, peft and language.
# Edit the arrays below to match your repository structure.

# Base model names (without the _{peft} suffix). Example: qwen3-asr
models=("qwen3-asr")

# Languages (each language has file: <model>_<peft>/<lang>.txt and output dir <model>_<peft>/<lang>/)
langs=("irish" "english" "shona" "javanese")

# Path to the extraction script (relative to repo root)
EXTRACT_SCRIPT="local/make_graphs.py"

# Loop and call extractor
for model in "${models[@]}"; do
  for lang in "${langs[@]}"; do
    echo "Processing: $model with language $lang"
    uv run "$EXTRACT_SCRIPT" --models "$model" --langs "$lang"
  done
done

echo "All done."
