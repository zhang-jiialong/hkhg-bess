#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_ablation_retrieval import load_questions_any, parse_indices, question_answers, question_text
from hh_iterative_ablation_retrieval import retrieve_layer_wise_topk
from hh_iterative_l2_path_beam_rerank import resolve_config
from hh_iterative_retrieval import MultiOrderIterativeRetriever
from Retrieve.lb_hgr_retrieval import load_retrieval_inputs
from strategy_eval_compare import save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layer-wise top-k retrieval only.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--question-indices", default="all")
    parser.add_argument("--top-k-per-layer", type=int, default=10)
    parser.add_argument("--top-k-final", type=int, default=20)
    parser.add_argument("--layer-wise-max-layer", type=int, default=2)
    parser.add_argument("--question-workers", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def summarize(retrieved: dict[str, Any]) -> dict[str, Any]:
    final_info = retrieved.get("final_rerank") or {}
    return {
        "visited_l0_count": retrieved.get("visited_l0_count"),
        "visited_count": retrieved.get("visited_count"),
        "final_evidence_candidate_count": final_info.get("candidate_count"),
        "selected_count": len(retrieved.get("top_l0_hyperedges") or []),
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
        max_level=max(2, args.layer_wise_max_layer),
    )

    questions = load_questions_any(Path(args.questions))
    indices = parse_indices(args.question_indices, total=len(questions))
    settings = {
        "strategy": "Layer-wise Top-K Retrieval",
        "retrieval_mode": "layer_wise_topk",
        "question_indices": indices,
        "top_k_per_layer": args.top_k_per_layer,
        "top_k_final": args.top_k_final,
        "layer_wise_max_layer": args.layer_wise_max_layer,
        "question_workers": max(1, args.question_workers),
        "generation_prompt": "handled downstream by existing prompt evaluator",
    }
    if args.resume and output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload["settings"] = settings
    else:
        payload = {"settings": settings, "questions": [], "averages": {}}
    save_results(output_json, payload)
    completed = {int(row["question_index"]) for row in payload.get("questions", [])}

    def run_one(order: int, idx: int) -> dict[str, Any] | None:
        if idx in completed:
            print(f"[{order}/{len(indices)}] skip completed Q{idx}", flush=True)
            return None
        raw = questions[idx]
        query = question_text(raw)
        last_exc: Exception | None = None
        for attempt in range(1, max(1, args.attempts) + 1):
            try:
                print(f"[{order}/{len(indices)}] layer_wise_topk Q{idx} attempt={attempt}: {query[:120]}", flush=True)
                start = time.perf_counter()
                retrieved = retrieve_layer_wise_topk(
                    retriever=retriever,
                    query=query,
                    top_k_per_layer=args.top_k_per_layer,
                    top_k_final=args.top_k_final,
                    max_layer=args.layer_wise_max_layer,
                )
                return {
                    "question_index": idx,
                    "question": query,
                    "golden_answers": question_answers(raw),
                    "answer": "",
                    "generation": "",
                    "retrieval_summary": summarize(retrieved),
                    "retrieval": retrieved,
                    "timings": {"retrieval_seconds": round(time.perf_counter() - start, 4)},
                }
            except Exception as exc:
                last_exc = exc
                print(f"  [retry] Q{idx} attempt={attempt} failed: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(10, 2 * attempt))
        raise last_exc  # type: ignore[misc]

    with ThreadPoolExecutor(max_workers=max(1, args.question_workers)) as executor:
        futures = {executor.submit(run_one, order, idx): idx for order, idx in enumerate(indices, start=1) if idx not in completed}
        for future in as_completed(futures):
            row = future.result()
            if row is None:
                continue
            payload["questions"].append(row)
            payload["questions"].sort(key=lambda item: int(item["question_index"]))
            save_results(output_json, payload)
            print(f"  saved -> {output_json} ({len(payload['questions'])}/{len(indices)})", flush=True)

    print(json.dumps({"output_json": str(output_json), "count": len(payload["questions"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

