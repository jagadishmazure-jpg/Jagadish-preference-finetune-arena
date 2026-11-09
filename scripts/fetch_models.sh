#!/usr/bin/env bash
# Download the open models into ./models and verify SHA-256:
#   two GGUF models (about 1.5 GB) that write candidate answers and act as the judge, and
#   SmolLM2-135M-Instruct (about 270 MB, pinned revision) that is tuned with LoRA.
# Only needed to re-measure data/; tests and the eval use the committed files.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
python3 - <<'PY' | while read -r file url sha; do
import json
for m in json.load(open("scripts/models.json"))["gguf"]:
    print(m["file"], m["url"], m["sha256"])
PY
  [[ -f "models/$file" ]] || curl -fL --retry 3 -o "models/$file" "$url"
  echo "$sha  models/$file" | sha256sum -c -
done
read -r repo rev sha < <(python3 -c 'import json; t=json.load(open("scripts/models.json"))["tuned"]; print(t["repo"], t["revision"], t["weights_sha256"])')
dir=models/smollm2-135m-instruct
mkdir -p "$dir"
for f in config.json generation_config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json \
         merges.txt vocab.json; do
  [[ -f "$dir/$f" ]] || curl -fL --retry 3 -o "$dir/$f" "https://huggingface.co/$repo/resolve/$rev/$f"
done
echo "$sha  $dir/model.safetensors" | sha256sum -c -
