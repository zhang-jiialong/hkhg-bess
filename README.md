# HKHG

HKHG provides graph construction, hierarchical retrieval, answer generation,
and evaluation for question answering over domain documents.

## 1. Installation

Python 3.11 is recommended. A CUDA-capable GPU is optional but recommended for
local reranking.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately when GPU reranking is required.

## 2. Download the datasets

The GitHub repository contains only the empty data directory structure. Download
the data archive separately and extract it into the repository root.

```text
Baidu Netdisk URL: <BAIDU_NETDISK_URL>
Extraction code: <EXTRACTION_CODE>
```

After extraction, the relevant layout must be:

```text
hkhg/
│
├── README.md
├── requirements.txt
├── runtime_config.example.json
├── configs/
│   └── default.json
│
├── HKHG/
│   ├── HKHG.py
│   ├── prompt.py
│   ├── operate.py
│   ├── storage.py
│   └── llm.py
│
├── contexts/
│   ├── bms_contexts.json
│   ├── hlib_contexts.json
│   ├── flb_contexts.json
│   ├── pb_contexts.json
│   ├── incident_liverpool_contexts.json
│   ├── incident_mcmicken_contexts.json
│   ├── agriculture_contexts.json
│   ├── cs_contexts.json
│   ├── hypertension_contexts.json
│   ├── legal_contexts.json
│   └── mix_contexts.json
│
├── datasets/
│   ├── README.md
│   ├── BMS/
│   │   ├── context/
│   │   │   ├── BMS.md
│   │   │   └── construction_contexts.json
│   │   ├── questions/
│   │   │   ├── short_QA.json
│   │   │   └── long_QA.json
│   │   └── graph/
│   ├── HLIB/
│   │   ├── context/
│   │   │   ├── HLIB.md
│   │   │   └── construction_contexts.json
│   │   ├── questions/
│   │   │   ├── short_QA.json
│   │   │   └── long_QA.json
│   │   └── graph/
│   ├── FLB/
│   │   ├── context/
│   │   │   ├── FLB.md
│   │   │   └── construction_contexts.json
│   │   ├── questions/
│   │   │   ├── short_QA.json
│   │   │   └── long_QA.json
│   │   └── graph/
│   ├── PB/
│   │   ├── context/
│   │   │   ├── PB.md
│   │   │   └── construction_contexts.json
│   │   ├── questions/
│   │   │   └── long_97.json
│   │   └── graph/
│   ├── incident/
│   │   ├── liverpool/
│   │   │   ├── context/
│   │   │   ├── questions/
│   │   │   └── graph/
│   │   └── mcmicken/
│   │       ├── context/
│   │       ├── questions/
│   │       └── graph/
│   └── KHQA/
│       ├── agriculture/
│       │   └── {context,questions,graph}/
│       ├── cs/
│       │   └── {context,questions,graph}/
│       ├── hypertension/
│       │   └── {context,questions,graph}/
│       ├── legal/
│       │   └── {context,questions,graph}/
│       └── mix/
│           └── {context,questions,graph}/
│
├── eval/
├── Retrieve/
├── examples/
│   └── minimal_example/
│       ├── context/
│       ├── questions/
│       ├── README.md
│       ├── run_end_to_end.sh
│       └── verify_outputs.py
├── construct_core_configurable_20260515.py
├── run_all_visited_selection_retrieval_only_20260604.py
└── eval_existing_prompts_think_answer_from_retrieval_20260608.py
```

Each downloaded `graph/` directory must contain at least:

```text
graph_chunk_entity_relation.graphml
kv_store_full_docs.json
kv_store_text_chunks.json
vdb_chunks.json
vdb_entities.json
vdb_entity_names.json
vdb_hyperedges.json
```

## 3. Configure the API

Copy the example configuration and insert credentials for an OpenAI-compatible
endpoint. Do not commit `runtime_config.json`.

```bash
cp runtime_config.example.json runtime_config.json
```

Example using the official OpenAI endpoint:

```json
{
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key": "YOUR_API_KEY"
  },
  "embedding": {
    "model_path": "text-embedding-3-large",
    "device": null
  }
}
```

The construction entry point also accepts these environment variables:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
```

## 4. Build a graph

The included `--data_source` names are `bms`, `hlib`, `flb`, `pb`,
`incident_liverpool`, `incident_mcmicken`, `agriculture`, `cs`, `hypertension`,
`legal`, and `mix`. The minimal example additionally uses `minimal_example`.

```bash
python construct_core_configurable_20260515.py \
  --data_source bms \
  --config runtime_config.json \
  --mdl gpt-4o-mini \
  --concurrency 8
```

New graph files are written to `data/work/<data_source>/`.

## 5. Run retrieval

The full-method retrieval parameters are stored in `configs/default.json`.
Change `rerank_device` to `cpu` in that file when CUDA is unavailable.

```bash
python run_all_visited_selection_retrieval_only_20260604.py \
  --workdir datasets/BMS/graph \
  --questions datasets/BMS/questions/short_QA.json \
  --output-dir outputs/BMS/retrieval \
  --config runtime_config.json \
  --experiment-config configs/default.json
```

The retrieval output is
`outputs/BMS/retrieval/selection_retrieval_results.json`.
Individual command-line options can still override values from
`configs/default.json` when required.

## 6. Generate answers

Short-form example:

```bash
python eval_existing_prompts_think_answer_from_retrieval_20260608.py \
  --source outputs/BMS/retrieval/selection_retrieval_results.json \
  --questions datasets/BMS/questions/short_QA.json \
  --output-dir outputs/BMS/short_answers \
  --config runtime_config.json \
  --experiment-config configs/default.json \
  --mode short \
  --answer-language English
```

Long-form example:

```bash
python eval_existing_prompts_think_answer_from_retrieval_20260608.py \
  --source outputs/BMS/retrieval/selection_retrieval_results.json \
  --questions datasets/BMS/questions/long_QA.json \
  --output-dir outputs/BMS/long_answers \
  --config runtime_config.json \
  --experiment-config configs/default.json \
  --mode long \
  --answer-language English
```

Use `German` for HLIB and the language required by the selected dataset for
other runs. Generation parameters are stored in the `generation` section of
`configs/default.json`. The exact generation prompts are stored in
`HKHG/prompt.py`.

## 7. Minimal end-to-end example

The repository includes a small BMS example with one short-form question and
one long-form question. It constructs a new graph from the included excerpt,
runs retrieval and generation for both questions, evaluates the short answer
with EM, F1, and G-E, and verifies the generated artifacts.

```bash
bash examples/minimal_example/run_end_to_end.sh
```

The command uses `runtime_config.json` when it exists. It can also build a
temporary runtime configuration from the `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL` environment variables documented
above. To use another configuration file, pass its path as the only argument.

See `examples/minimal_example/README.md` for runtime configuration, output
paths, reranker setup, and evaluation scope.

## 8. Main files

```text
construct_core_configurable_20260515.py
run_all_visited_selection_retrieval_only_20260604.py
eval_existing_prompts_think_answer_from_retrieval_20260608.py
run_l0_only_all_visited_retrieval_20260608.py
run_layer_wise_topk_retrieval_only_20260608.py
configs/default.json
HKHG/prompt.py
examples/minimal_example/run_end_to_end.sh
```

Run any entry point with `--help` to inspect all available options.
