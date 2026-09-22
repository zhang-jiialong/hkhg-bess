#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_v2_mechanism_chain_aware import load_questions_any, parse_indices, question_answers, question_text
from hh_iterative_l2_path_beam_rerank import (
    LocalReranker,
    append_path,
    build_context,
    dedupe_paths,
    final_evidence_rerank,
    inject_l0_semantic_anchor_paths,
    make_anchor_path,
    resolve_config,
)
from hh_iterative_l2_path_beam_v2_mechanism_chain_aware import same_layer_candidates, structure_score, vector_redundancy
from hh_iterative_retrieval import MultiOrderIterativeRetriever
from Retrieve.lb_hgr_retrieval import load_retrieval_inputs
from strategy_eval_compare import save_results


def expand_l0_same_layer_bounded(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    paths: list[Any],
    *,
    visited: set[str],
    visited_l0: set[str],
    top_k_expand: int,
    mode: str,
    structure_alpha: float,
    redundancy_gamma: float,
    candidate_cap: int,
):
    by_end: dict[str, list[Any]] = {}
    for path in paths:
        if path.current_layer == 0:
            by_end.setdefault(path.last_edge, []).append(path)

    semantic_scores = retriever.semantic_scores_for_level(query, 0)
    semantic_order = retriever.semantic_order_for_level(query, 0)
    expanded_paths = []
    logs: list[dict[str, Any]] = []
    selected_for_redundancy = set(visited_l0)

    for source_id, source_paths in by_end.items():
        candidate_pool = same_layer_candidates(
            retriever,
            source_id,
            level=0,
            semantic_order=semantic_order,
            top_k_expand=top_k_expand,
        )
        if candidate_cap and candidate_cap > 0 and len(candidate_pool) > candidate_cap:
            candidate_pool = set(
                sorted(
                    candidate_pool,
                    key=lambda hid: float(semantic_scores.get(hid, 0.0)),
                    reverse=True,
                )[:candidate_cap]
            )

        scored: list[dict[str, Any]] = []
        for candidate_id in candidate_pool:
            if candidate_id not in retriever.hyperedge_vector_map:
                continue
            if candidate_id == source_id or candidate_id in visited:
                continue
            semantic = float(semantic_scores.get(candidate_id, 0.0))
            structure = structure_score(retriever, source_id, candidate_id, 0)
            redundancy = 0.0
            if mode == "semantic_only":
                score = semantic
            elif mode == "semantic_weak_structure":
                score = semantic + structure_alpha * structure
            elif mode == "semantic_weak_structure_redundancy":
                redundancy = vector_redundancy(retriever, candidate_id, selected_for_redundancy)
                score = semantic + structure_alpha * structure - redundancy_gamma * redundancy
            else:
                raise ValueError(f"Unsupported expansion mode: {mode}")
            scored.append(
                {
                    "source_hyperedge_id": source_id,
                    "candidate_hyperedge_id": candidate_id,
                    "level": 0,
                    "score": float(score),
                    "semantic_score": semantic,
                    "structure_score": float(structure),
                    "redundancy_score": float(redundancy),
                    "mode": mode,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        picked = scored[:top_k_expand]
        for item in picked:
            candidate_id = item["candidate_hyperedge_id"]
            for path in source_paths:
                expanded_paths.append(append_path(path, candidate_id, 0, "L0_only_same_layer_expansion"))
            logs.append(item)

    expanded_paths = dedupe_paths(expanded_paths)
    expanded_ids = {path.last_edge for path in expanded_paths}
    visited.update(expanded_ids)
    visited_l0.update(expanded_ids)
    return expanded_paths, {
        "stage": "L0_only_same_layer_bounded_expansion",
        "input_path_count": len(paths),
        "expanded_path_count": len(expanded_paths),
        "expanded_l0_count": len(expanded_ids),
        "expansion_log_count": len(logs),
        "scoring": mode,
        "structure_alpha": structure_alpha,
        "redundancy_gamma": redundancy_gamma,
        "candidate_cap_per_source": candidate_cap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L0-only same-layer all-visited ablation retrieval.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--question-indices", default="all")
    parser.add_argument("--top-k-anchor", type=int, default=20)
    parser.add_argument("--expansion-top-k", type=int, default=2)
    parser.add_argument("--path-beam-size", type=int, default=20)
    parser.add_argument("--top-k-final", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--l0-semantic-inject-k", type=int, default=10)
    parser.add_argument("--expansion-mode", choices=["semantic_only", "semantic_weak_structure", "semantic_weak_structure_redundancy"], default="semantic_weak_structure")
    parser.add_argument("--structure-alpha", type=float, default=0.1)
    parser.add_argument("--redundancy-gamma", type=float, default=0.2)
    parser.add_argument("--same-layer-candidate-cap", type=int, default=600)
    parser.add_argument("--final-rerank-top-k", type=int, default=0, help="0 keeps all visited L0; positive keeps pure reranker top-k.")
    parser.add_argument("--local-rerank-model", required=True)
    parser.add_argument("--rerank-device", default="cuda:1")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--local-rerank-max-length", type=int, default=512)
    parser.add_argument("--question-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def retrieve_l0_only_all_visited(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    reranker: LocalReranker,
    top_k_anchor: int,
    expansion_top_k: int,
    path_beam_size: int,
    top_k_final: int,
    max_iter: int,
    l0_semantic_inject_k: int,
    expansion_mode: str,
    structure_alpha: float,
    redundancy_gamma: float,
    same_layer_candidate_cap: int,
    final_rerank_top_k: int,
) -> dict[str, Any]:
    anchors = retriever.initial_graph_anchoring(query, top_k_anchor=top_k_anchor)
    path_beam = [make_anchor_path(retriever, item) for item in anchors]
    visited: set[str] = {path.last_edge for path in path_beam}
    visited_l0: set[str] = set(visited)
    iteration_logs: list[dict[str, Any]] = []

    for iteration in range(max_iter):
        log: dict[str, Any] = {
            "iteration": iteration,
            "input_path_count": len(path_beam),
            "input_end_l0_count": len({path.last_edge for path in path_beam}),
            "stages": [],
        }

        next_paths = []
        if path_beam:
            next_paths, info = expand_l0_same_layer_bounded(
                retriever,
                query,
                path_beam,
                visited=visited,
                visited_l0=visited_l0,
                top_k_expand=expansion_top_k,
                mode=expansion_mode,
                structure_alpha=structure_alpha,
                redundancy_gamma=redundancy_gamma,
                candidate_cap=same_layer_candidate_cap,
            )
            log["stages"].append(info)

        if l0_semantic_inject_k > 0:
            injected_paths, info = inject_l0_semantic_anchor_paths(
                retriever,
                query,
                visited=visited,
                visited_l0=visited_l0,
                top_k=l0_semantic_inject_k,
                iteration=iteration,
            )
            log["stages"].append(info)
            next_paths = dedupe_paths(next_paths + injected_paths)

        # Keep the L0-only ablation comparable to the hierarchy path beam:
        # the next expansion frontier is bounded, while visited_l0 keeps all evidence.
        if len(next_paths) > path_beam_size:
            next_paths = sorted(next_paths, key=lambda path: float(path.score), reverse=True)[:path_beam_size]

        log["output_path_count"] = len(next_paths)
        log["visited_count"] = len(visited)
        log["visited_l0_count"] = len(visited_l0)
        iteration_logs.append(log)
        path_beam = next_paths

    rerank_k = final_rerank_top_k if final_rerank_top_k and final_rerank_top_k > 0 else max(1, len(visited_l0))
    reranked_l0, final_info = final_evidence_rerank(retriever, reranker, query, visited_l0, rerank_k)
    if final_rerank_top_k and final_rerank_top_k > 0:
        selected_l0 = reranked_l0[:final_rerank_top_k]
        selection_strategy = "w/o hierarchy1 pure_reranker_topk"
        final_selection_mode = "reranker_topk"
    else:
        selected_l0 = reranked_l0
        selection_strategy = "w/o hierarchy1 all_visited_l0_no_final_selection"
        final_selection_mode = "all_visited"
    selection_info = {
        "strategy": selection_strategy,
        "candidate_count": len(reranked_l0),
        "candidate_pool_count": len(reranked_l0),
        "selected_count": len(selected_l0),
        "top_m": len(reranked_l0),
        "final_k": len(selected_l0),
    }
    final_info = {
        **final_info,
        "kept_count": len(selected_l0),
        "reranker_top_m": len(reranked_l0),
        "selection_pool_size": len(reranked_l0),
        "selection": selection_info,
    }
    context = build_context(selected_l0).replace(
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        "[w/o hierarchy1 L0-only same-layer all-visited: Final L0 Evidence]",
    )
    return {
        "query": query,
        "strategy": "w/o hierarchy1",
        "settings": {
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "l0_semantic_inject_k": l0_semantic_inject_k,
            "expansion_mode": expansion_mode,
            "structure_alpha": structure_alpha,
            "redundancy_gamma": redundancy_gamma,
            "same_layer_candidate_cap": same_layer_candidate_cap,
            "hierarchy_traversal": "disabled",
            "final_selection_mode": final_selection_mode,
            "final_rerank_top_k": final_rerank_top_k,
        },
        "anchors": anchors,
        "iterations": iteration_logs,
        "final_rerank": final_info,
        "top_l0_hyperedges": selected_l0,
        "selected_hyperedges": selected_l0,
        "visited_l0_count": len(visited_l0),
        "visited_count": len(visited),
        "context": context,
    }


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
        "strategy": "w/o hierarchy1",
        "retrieval_mode": "l0_only_same_layer_all_visited",
        "question_indices": indices,
        "top_k_anchor": args.top_k_anchor,
        "expansion_top_k": args.expansion_top_k,
        "path_beam_size": args.path_beam_size,
        "top_k_final": args.top_k_final,
        "max_iter": args.max_iter,
        "l0_semantic_inject_k": args.l0_semantic_inject_k,
        "expansion_mode": args.expansion_mode,
        "structure_alpha": args.structure_alpha,
        "redundancy_gamma": args.redundancy_gamma,
        "same_layer_candidate_cap": args.same_layer_candidate_cap,
        "hierarchy_traversal": "disabled",
        "final_selection_mode": "all_visited",
        "final_rerank_top_k": args.final_rerank_top_k,
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
        print(f"[{order}/{len(indices)}] w/o hierarchy1 Q{question_index}: {question}", flush=True)
        start = time.perf_counter()
        retrieved = retrieve_l0_only_all_visited(
            retriever=retriever,
            query=question,
            reranker=reranker,
            top_k_anchor=args.top_k_anchor,
            expansion_top_k=args.expansion_top_k,
            path_beam_size=args.path_beam_size,
            top_k_final=args.top_k_final,
            max_iter=args.max_iter,
            l0_semantic_inject_k=args.l0_semantic_inject_k,
            expansion_mode=args.expansion_mode,
            structure_alpha=args.structure_alpha,
            redundancy_gamma=args.redundancy_gamma,
            same_layer_candidate_cap=args.same_layer_candidate_cap,
            final_rerank_top_k=args.final_rerank_top_k,
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
        futures = [executor.submit(run_one, order, idx) for order, idx in enumerate(indices, start=1)]
        for future in as_completed(futures):
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
