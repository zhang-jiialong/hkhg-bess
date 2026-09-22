# Minimal end-to-end example

This example runs HKHG from graph construction through retrieval, answer
generation, and short-form evaluation on a small excerpt from the BMS dataset.

## Included data

```text
context/BMS_excerpt.md
context/construction_contexts.json
questions/short_1.json
questions/long_1.json
```

## Run

From the repository root, set the API environment variables and run the
example:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
bash examples/minimal_example/run_end_to_end.sh
```

Alternatively, create `runtime_config.json` as described in the main README
and run the same command. A different configuration file can be supplied as
the only argument:

```bash
bash examples/minimal_example/run_end_to_end.sh path/to/runtime_config.json
```

Set `PYTHON_BIN` when a specific interpreter is required:

```bash
PYTHON_BIN=/path/to/python bash examples/minimal_example/run_end_to_end.sh
```

The reranker in `configs/default.json` uses `BAAI/bge-reranker-v2-m3`. On an
offline machine, download the model in advance and set `local_rerank_model` to
its local directory. Set `rerank_device` to `cpu` when CUDA is unavailable.

## Outputs

The run writes temporary outputs under:

```text
data/work/minimal_example/
examples/minimal_example/output/
```

The short-form run reports EM, F1, and G-E. R-S is skipped. The long-form run
performs retrieval and answer generation. `verify_outputs.py` checks that the
graph, retrieval outputs, generated answers, and short-form metrics exist and
that G-E scoring completed without a fallback score.
