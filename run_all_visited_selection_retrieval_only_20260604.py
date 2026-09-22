#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_v2_mechanism_chain_aware import load_questions_any, parse_indices, question_answers, question_text
from hh_iterative_l2_path_beam_rerank import LocalReranker, resolve_config
from hh_iterative_l2_path_beam_v2_mechanism_chain_aware import retrieve_l2_path_beam_v2_expansion_variant
from hh_iterative_retrieval import MultiOrderIterativeRetriever
from Retrieve.lb_hgr_retrieval import load_retrieval_inputs
from strategy_eval_compare import save_results


def load_experiment_defaults(path: str, section: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    values = payload.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"{section!r} in {config_path} must be a JSON object")
    return values


def parse_args() -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--experiment-config", default="configs/default.json")
    known, _ = probe.parse_known_args()
    defaults = load_experiment_defaults(known.experiment_config, "retrieval")

    parser = argparse.ArgumentParser(description="Run all-visited-L0 final selection retrieval only.")
    parser.add_argument("--experiment-config", default=known.experiment_config)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--question-indices", default=defaults.get("question_indices", "all"))
    parser.add_argument("--top-k-anchor", type=int, default=defaults.get("top_k_anchor", 20))
    parser.add_argument("--expansion-top-k", type=int, default=defaults.get("expansion_top_k", 2))
    parser.add_argument("--path-beam-size", type=int, default=defaults.get("path_beam_size", 20))
    parser.add_argument("--top-k-final", type=int, default=defaults.get("top_k_final", 20))
    parser.add_argument("--max-iter", type=int, default=defaults.get("max_iter", 3))
    parser.add_argument("--candidate-cap", type=int, default=defaults.get("candidate_cap"))
    parser.add_argument("--l0-semantic-inject-k", type=int, default=defaults.get("l0_semantic_inject_k", 10))
    parser.add_argument("--expansion-mode", choices=["semantic_only", "semantic_weak_structure", "semantic_weak_structure_redundancy"], default=defaults.get("expansion_mode", "semantic_weak_structure"))
    parser.add_argument("--structure-alpha", type=float, default=defaults.get("structure_alpha", 0.1))
    parser.add_argument("--redundancy-gamma", type=float, default=defaults.get("redundancy_gamma", 0.2))
    parser.add_argument("--selection-name", default=defaults.get("selection_name", "full_rerank_top20"))
    parser.add_argument("--selection-alpha", type=float, default=defaults.get("selection_alpha", 0.75))
    parser.add_argument("--selection-beta", type=float, default=defaults.get("selection_beta", 0.2))
    parser.add_argument("--selection-delta", type=float, default=defaults.get("selection_delta", 0.05))
    parser.add_argument("--mechanism-top-m", type=int, default=defaults.get("mechanism_top_m", 20))
    parser.add_argument("--path-selection-mode", choices=["rerank", "mmr"], default=defaults.get("path_selection_mode", "rerank"))
    parser.add_argument("--path-alpha", type=float, default=defaults.get("path_alpha", 0.85))
    parser.add_argument("--path-beta", type=float, default=defaults.get("path_beta", 0.15))
    parser.add_argument("--path-delta", type=float, default=defaults.get("path_delta", 0.05))
    parser.add_argument("--final-selection-mode", choices=["greedy", "all_visited"], default=defaults.get("final_selection_mode", "all_visited"))
    parser.add_argument("--local-rerank-model", default=defaults.get("local_rerank_model", "BAAI/bge-reranker-v2-m3"))
    parser.add_argument("--rerank-device", default=defaults.get("rerank_device", "cuda:0"))
    parser.add_argument("--rerank-batch-size", type=int, default=defaults.get("rerank_batch_size", 64))
    parser.add_argument("--local-rerank-max-length", type=int, default=defaults.get("local_rerank_max_length", 512))
    parser.add_argument("--question-workers", type=int, default=defaults.get("question_workers", 8))
    parser.add_argument("--resume", action="store_true", default=bool(defaults.get("resume", False)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "selection_retrieval_results.json"

    client, embedding_model, _llm_model, _base_url, _api_key = resolve_config(args)
    graph, text_chunks, hyperedge_vector_map, client, embedding_model = load_retrieval_inputs(args.workdir, client, embedding_model)
    retriever = MultiOrderIterativeRetriever(
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        max_level=2,
    )
    reranker = LocalReranker(
        model_path=args.local_rerank_model,
        device=args.rerank_device,
        batch_size=args.rerank_batch_size,
        max_length=args.local_rerank_max_length,
    )

    questions = load_questions_any(Path(args.questions))
    indices = parse_indices(args.question_indices, total=len(questions))
    settings = {
        "strategy": f"all_visited_l0_{args.selection_name}",
        "retrieval_mode": f"all_visited_l0_{args.selection_name}",
        "question_indices": indices,
        "top_k_anchor": args.top_k_anchor,
        "expansion_top_k": args.expansion_top_k,
        "path_beam_size": args.path_beam_size,
        "top_k_final": args.top_k_final,
        "max_iter": args.max_iter,
        "candidate_cap": args.candidate_cap,
        "l0_semantic_inject_k": args.l0_semantic_inject_k,
        "expansion_mode": args.expansion_mode,
        "structure_alpha": args.structure_alpha,
        "redundancy_gamma": args.redundancy_gamma,
        "mechanism_top_m": args.mechanism_top_m,
        "selection_alpha": args.selection_alpha,
        "selection_beta": args.selection_beta,
        "selection_gamma": 0.0,
        "selection_delta": args.selection_delta,
        "path_selection_mode": args.path_selection_mode,
        "path_alpha": args.path_alpha,
        "path_beta": args.path_beta,
        "path_delta": args.path_delta,
        "final_selection_mode": args.final_selection_mode,
        "mechanism_extractor": "none",
        "reranker": reranker.name,
        "question_workers": max(1, args.question_workers),
    }
    if args.resume and output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload["settings"] = settings
    else:
        payload = {"settings": settings, "questions": [], "averages": {}}
    save_results(output_json, payload)
    completed = {int(row["question_index"]) for row in payload.get("questions", [])}

    def run_one(order: int, question_index: int) -> dict[str, Any] | None:
        if question_index in completed:
            print(f"[{order}/{len(indices)}] skip completed Q{question_index}", flush=True)
            return None
        raw = questions[question_index]
        question = question_text(raw)
        print(f"[{order}/{len(indices)}] {args.selection_name} Q{question_index}: {question}", flush=True)
        start = time.perf_counter()
        retrieved = retrieve_l2_path_beam_v2_expansion_variant(
            retriever=retriever,
            query=question,
            reranker=reranker,
            top_k_anchor=args.top_k_anchor,
            expansion_top_k=args.expansion_top_k,
            path_beam_size=args.path_beam_size,
            top_k_final=args.top_k_final,
            max_iter=args.max_iter,
            candidate_cap=args.candidate_cap,
            l0_semantic_inject_k=args.l0_semantic_inject_k,
            expansion_mode=args.expansion_mode,
            structure_alpha=args.structure_alpha,
            redundancy_gamma=args.redundancy_gamma,
            mechanism_top_m=args.mechanism_top_m,
            mechanism_alpha=args.selection_alpha,
            mechanism_beta=args.selection_beta,
            mechanism_gamma=0.0,
            mechanism_delta=args.selection_delta,
            mechanism_extractor="none",
            path_selection_mode=args.path_selection_mode,
            path_alpha=args.path_alpha,
            path_beta=args.path_beta,
            path_delta=args.path_delta,
            final_selection_mode=args.final_selection_mode,
        )
        return {
            "question_index": question_index,
            "question": question,
            "golden_answers": question_answers(raw),
            "answer": "",
            "generation": "",
            "retrieval_summary": {
                "selection_pool_size": (retrieved.get("final_rerank") or {}).get("selection_pool_size"),
                "selected_count": len(retrieved.get("top_l0_hyperedges") or []),
            },
            "retrieval": retrieved,
            "timings": {"retrieval_seconds": round(time.perf_counter() - start, 4)},
        }

    with ThreadPoolExecutor(max_workers=max(1, args.question_workers)) as executor:
        future_map = {executor.submit(run_one, order, idx): idx for order, idx in enumerate(indices, start=1)}
        for future in as_completed(future_map):
            row = future.result()
            if row is None:
                continue
            payload["questions"].append(row)
            payload["questions"].sort(key=lambda item: int(item["question_index"]))
            save_results(output_json, payload)
            print(f"  saved -> {output_json} ({len(payload['questions'])}/{len(indices)})", flush=True)

    save_results(output_json, payload)
    print(json.dumps({"output_json": str(output_json), "count": len(payload["questions"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
