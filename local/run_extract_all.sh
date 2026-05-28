#!/usr/bin/env bash
set -euo pipefail

# run_extract_all.sh
# Calls local/extract_logs.py for every combination of model, peft and language.
# Edit the arrays below to match your repository structure.

# Base model names (without the _{peft} suffix). Example: qwen3-asr
models=("qwen3-asr" "whisper")

# PEFT methods (directories are expected to be named <model>_<peft>)
pefts=("lora" "vera" "randlora")

# Languages (each language has file: <model>_<peft>/<lang>.txt and output dir <model>_<peft>/<lang>/)
langs=("irish" "english" "shona" "javanese")

# Path to the extraction script (relative to repo root)
EXTRACT_SCRIPT="local/extract_logs.py"

# Loop and call extractor
for model in "${models[@]}"; do
  for peft in "${pefts[@]}"; do
    for lang in "${langs[@]}"; do
      dir="${model}_${peft}"
      infile="${dir}/${lang}.txt"
      outdir="${dir}/${lang}/"

      if [ -f "$infile" ]; then
        mkdir -p "$outdir"
        echo "Processing: $infile -> $outdir"
        uv run "$EXTRACT_SCRIPT" --file "$infile" --output "$outdir"
      else
        echo "Skipping (not found): $infile"
      fi
    done
  done
done

echo "All done."
