#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXAMPLE_DIR="examples/minimal_example"
WORKDIR="data/work/minimal_example"
TEMP_RUNTIME_CONFIG=""

cleanup() {
  rm -f -- contexts/minimal_example_contexts.json
  if [[ -n "$TEMP_RUNTIME_CONFIG" ]]; then
    rm -f -- "$TEMP_RUNTIME_CONFIG"
  fi
}
trap cleanup EXIT

if [[ $# -gt 1 ]]; then
  echo "Usage: bash $EXAMPLE_DIR/run_end_to_end.sh [runtime_config.json]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  RUNTIME_CONFIG="$1"
elif [[ -f runtime_config.json ]]; then
  RUNTIME_CONFIG="runtime_config.json"
elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
  TEMP_RUNTIME_CONFIG="$(mktemp "${TMPDIR:-/tmp}/hkhg-runtime-config.XXXXXX.json")"
  RUNTIME_CONFIG="$TEMP_RUNTIME_CONFIG"
  "$PYTHON_BIN" - "$RUNTIME_CONFIG" <<'PY'
import json
import os
import sys

config = {
    "llm": {
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.environ["OPENAI_API_KEY"],
    },
    "embedding": {
        "model_path": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        "device": None,
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
PY
else
  echo "No API configuration was found." >&2
  echo "Either create runtime_config.json or set OPENAI_API_KEY." >&2
  exit 1
fi

if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "Runtime configuration not found: $RUNTIME_CONFIG" >&2
  exit 1
fi

export HKHG_RUNTIME_CONFIG="$RUNTIME_CONFIG"

# A complete local reranker snapshot should be used without online metadata checks.
if [[ -z "${HF_HUB_OFFLINE:-}" ]]; then
  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    HF_CACHE_ROOT="$HF_HUB_CACHE"
  elif [[ -n "${HF_HOME:-}" ]]; then
    HF_CACHE_ROOT="$HF_HOME/hub"
  else
    HF_CACHE_ROOT="$HOME/.cache/huggingface/hub"
  fi
  for snapshot in "$HF_CACHE_ROOT"/models--BAAI--bge-reranker-v2-m3/snapshots/*; do
    if [[ -f "$snapshot/config.json" && -f "$snapshot/tokenizer_config.json" && -f "$snapshot/model.safetensors" ]]; then
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      echo "Using the cached BAAI/bge-reranker-v2-m3 model in offline mode."
      break
    fi
  done
fi

if [[ "$EXAMPLE_DIR" != "examples/minimal_example" || "$WORKDIR" != "data/work/minimal_example" ]]; then
  echo "Refusing to reset unexpected example paths." >&2
  exit 1
fi

rm -rf -- "$WORKDIR" "$EXAMPLE_DIR/output"
rm -f -- contexts/minimal_example_contexts.json

cp "$EXAMPLE_DIR/context/construction_contexts.json" contexts/minimal_example_contexts.json

"$PYTHON_BIN" construct_core_configurable_20260515.py \
  --data_source minimal_example \
  --config "$RUNTIME_CONFIG" \
  --concurrency 2 \
  --request-timeout 180

"$PYTHON_BIN" run_all_visited_selection_retrieval_only_20260604.py \
  --workdir "$WORKDIR" \
  --questions "$EXAMPLE_DIR/questions/short_1.json" \
  --output-dir "$EXAMPLE_DIR/output/retrieval_short" \
  --config "$RUNTIME_CONFIG" \
  --experiment-config configs/default.json

"$PYTHON_BIN" run_all_visited_selection_retrieval_only_20260604.py \
  --workdir "$WORKDIR" \
  --questions "$EXAMPLE_DIR/questions/long_1.json" \
  --output-dir "$EXAMPLE_DIR/output/retrieval_long" \
  --config "$RUNTIME_CONFIG" \
  --experiment-config configs/default.json

"$PYTHON_BIN" eval_existing_prompts_think_answer_from_retrieval_20260608.py \
  --source "$EXAMPLE_DIR/output/retrieval_short/selection_retrieval_results.json" \
  --questions "$EXAMPLE_DIR/questions/short_1.json" \
  --output-dir "$EXAMPLE_DIR/output/generation_short" \
  --config "$RUNTIME_CONFIG" \
  --experiment-config configs/default.json \
  --mode short \
  --answer-language English \
  --skip-rsim

"$PYTHON_BIN" eval_existing_prompts_think_answer_from_retrieval_20260608.py \
  --source "$EXAMPLE_DIR/output/retrieval_long/selection_retrieval_results.json" \
  --questions "$EXAMPLE_DIR/questions/long_1.json" \
  --output-dir "$EXAMPLE_DIR/output/generation_long" \
  --config "$RUNTIME_CONFIG" \
  --experiment-config configs/default.json \
  --mode long \
  --answer-language English \
  --skip-rsim \
  --skip-gen

"$PYTHON_BIN" "$EXAMPLE_DIR/verify_outputs.py"
