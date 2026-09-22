import argparse
import json
import os
import shutil
import time

from HKHG import HKHG
from HKHG.utils import logger


def load_runtime_config(config_path: str = "runtime_config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


parser = argparse.ArgumentParser()
parser.add_argument("--data_source", default="example")
parser.add_argument("--mdl", type=str, default="gpt-4o-mini")
parser.add_argument("--threshold", type=float, default=0.9)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--config", type=str, default="runtime_config.json")
parser.add_argument("--concurrency", type=int, default=8)
parser.add_argument("--request-timeout", type=float, default=90.0)
args = parser.parse_args()

runtime_config = load_runtime_config(args.config)
llm_config = runtime_config.get("llm", {})
embedding_config = runtime_config.get("embedding", {})
higher_order_config = {
    "enabled": True,
    "max_level": 4,
    "cluster_strategy": "mixed_graph_community",
    "semantic_top_k": 30,
    "semantic_ef_search": 120,
    "semantic_hnsw_m": 16,
    "semantic_pca_dim": 256,
    "higher_order_batch_size": 8,
    "topology_hops": 2,
    "semantic_weight": 0.7,
    "topology_weight": 0.3,
    "topology_direct_weight": 0.5,
    "topology_path_weight": 0.5,
    "topology_path_decay": 0.7,
    "mixed_similarity_threshold": 0.5,
    "min_cluster_size": 2,
    "min_samples": 1,
    "cluster_metric": "cosine",
    "cluster_selection_epsilon": 0.0,
    "max_summary_clusters": None,
    "max_cluster_size": 100,
    "promote_singletons": True,
    "skip_if_exists": True,
}
higher_order_config.update(runtime_config.get("higher_order_params", {}))

data_source = args.data_source
model = args.mdl if args.mdl != "gpt-4o-mini" else llm_config.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
similar_params = {
    "similar_threshold": args.threshold,
    "similar_top_k": args.k,
}

context_file = os.path.abspath(os.path.join("contexts", f"{data_source}_contexts.json"))
working_dir = os.path.abspath(f"./data/work/{data_source}")
os.makedirs(working_dir, exist_ok=True)
log_file = os.path.join(working_dir, f"{data_source}_construct.log")

rag = HKHG(
    working_dir=working_dir,
    embedding_func_max_async=args.concurrency,
    llm_model_max_async=args.concurrency,
    llm_model_name=model,
    llm_model_kwargs={
        "base_url": llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        "timeout": args.request_timeout,
    },
    embedding_model_name=embedding_config.get("model_path") or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
    embedding_model_device=embedding_config.get("device"),
    log_level="INFO",
    log_file=log_file,
    max_concurrency=args.concurrency,
    similar_params=similar_params,
    higher_order_params=higher_order_config,
)


def initial_insert_text() -> None:
    with open(context_file, mode="r", encoding="utf-8") as f:
        unique_contexts = json.load(f)

    def core_files_ready() -> bool:
        required = {
            "graph_chunk_entity_relation.graphml": 1024 * 1024,
            "vdb_hyperedges.json": 1024,
            "vdb_entities.json": 1024,
            "kv_store_text_chunks.json": 1024,
        }
        for filename, min_size in required.items():
            path = os.path.join(working_dir, filename)
            if not os.path.exists(path) or os.path.getsize(path) < min_size:
                return False
        return True

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            return
        except Exception as exc:
            if core_files_ready():
                logger.warning(
                    "rag.insert raised after core graph files were written; treating core graph construction as complete. "
                    f"error={exc}"
                )
                return
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {exc}", flush=True)
            time.sleep(10)
    raise RuntimeError("Insertion failed after exceeding the maximum number of retries")


if __name__ == "__main__":
    initial_insert_text()
    graph_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    backup_path = os.path.join(working_dir, "graph_chunk_entity_relation_init.graphml")
    if os.path.exists(graph_path):
        shutil.copy2(graph_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
    logger.info("Core graph construction completed; skipped entity_names/similar_edges/synonym post-processing.")
