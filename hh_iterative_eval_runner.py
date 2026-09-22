#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from eval.eval import cal_em, cal_f1
from hh_iterative_retrieval import hierarchical_reasoning_path_retrieve
from layer_answer_compare import ask_llm
from layer_retrieval_compare import (
    load_graph,
    load_hyperedge_vectors,
    load_json,
    load_questions,
    load_runtime_config,
    make_openai_client,
    sample_questions,
)
from HKHG.prompt import PROMPTS
from strategy_eval_compare import build_error_details, maybe_cal_gen, maybe_cal_rsim, parse_question_indices, save_results, select_questions


def build_final_l0_prompt_context(top_l0_hyperedges: list[dict[str, Any]]) -> str:
    lines = ["[Multi-order Iterative Retrieval: Final L0 Hyperedges]"]
    for idx, item in enumerate(top_l0_hyperedges, start=1):
        score = float(item.get("final_score", item.get("score", 0.0)))
        lines.append(f"{idx}. ({score:.4f}) {item['hyperedge_id']}")
        source_text = str(item.get("source_text", "")).strip()
        if source_text:
            lines.append(source_text[:1200])
    return "\n".join(lines)


def build_long_answer_prompt(question: str, context: str) -> str:
    return PROMPTS["final_answer_generation_long"].format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        question=question,
        context=context,
    )


def build_answer_prompt(
    question: str,
    context: str,
    answer_style: str,
    answer_language: str = "the language of the question",
) -> str:
    prompt_key = "final_answer_generation_long" if answer_style == "long" else "final_answer_generation"
    return PROMPTS[prompt_key].format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        question=question,
        context=context,
        answer_language=answer_language,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HHIterative-Retrieval only.")
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-indices", default="")
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--top-n-seed", type=int, default=5)
    parser.add_argument("--top-p-parent", type=int, default=2)
    parser.add_argument("--top-k-expand", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--max-evidence", type=int, default=20)
    parser.add_argument("--max-paths", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--max-l3-per-l4", type=int, default=5)
    parser.add_argument("--max-l2-per-l3", type=int, default=5)
    parser.add_argument("--max-l1-per-l2", type=int, default=5)
    parser.add_argument("--max-l0-per-l1", type=int, default=10)
    parser.add_argument("--max-downward-l0-per-iter", type=int, default=300)
    parser.add_argument("--answer-style", choices=["short", "long"], default="long")
    parser.add_argument("--skip-rsim", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    return parser.parse_args()


def compute_averages(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "rsim": [],
        "gen": [],
        "answer_seconds": [],
        "score_seconds": [],
        "total_seconds": [],
        "confidence": [],
    }
    for item in question_results:
        scores = item["scores"]
        timings = item["timings"]
        for metric in ["em", "f1", "rsim", "gen"]:
            value = scores.get(metric)
            if isinstance(value, (int, float)):
                metric_values[metric].append(float(value))
        for metric in ["answer_seconds", "score_seconds", "total_seconds"]:
            value = timings.get(metric)
            if isinstance(value, (int, float)):
                metric_values[metric].append(float(value))
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            metric_values["confidence"].append(float(confidence))
    return {
        metric: round(statistics.mean(values), 4) if values else None
        for metric, values in metric_values.items()
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hh_iterative_eval_results.json"

    runtime_config = load_runtime_config(Path(args.config))
    client, embedding_model = make_openai_client(runtime_config)
    llm_model_name = runtime_config.get("llm", {}).get("model", "gpt-4o-mini")

    workdir = Path(args.workdir)
    graph = load_graph(workdir / "graph_chunk_entity_relation.graphml")
    records = load_hyperedge_vectors(workdir / "vdb_hyperedges.json")
    hyperedge_vector_map = {item["hyperedge_name"]: item["vector"] for item in records}
    text_chunks = load_json(workdir / "kv_store_text_chunks.json") or {}

    questions = load_questions(Path(args.questions))
    selected_indices = parse_question_indices(args.question_indices)
    sampled_questions = select_questions(questions, args.sample_size, args.seed, selected_indices)

    results: dict[str, Any] = {
        "data_source": args.data_source,
        "strategy": "HHIterative-Retrieval",
        "sample_size": len(sampled_questions),
        "seed": args.seed,
        "question_indices": selected_indices,
        "skip_rsim": args.skip_rsim,
        "skip_gen": args.skip_gen,
        "answer_style": args.answer_style,
        "embedding_model": embedding_model,
        "llm_model": llm_model_name,
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "questions": [],
        "averages": {},
    }
    save_results(results_path, results)

    for order, item in enumerate(sampled_questions, start=1):
        question = item["question"]
        golden_answers = item["raw"].get("golden_answers", [])
        context_text = "\n".join(item["raw"].get("context", []))
        print(f"[{order}/{len(sampled_questions)}] Question #{item['question_index']}: {question}", flush=True)
        try:
            t0 = time.perf_counter()
            hh_result = hierarchical_reasoning_path_retrieve(
                query=question,
                graph=graph,
                text_chunks=text_chunks,
                hyperedge_vector_map=hyperedge_vector_map,
                client=client,
                embedding_model=embedding_model,
                llm_model_name=llm_model_name,
                top_n_seed=args.top_n_seed,
                top_p_parent=args.top_p_parent,
                top_k_expand=args.top_k_expand,
                max_level=args.max_level,
                max_iter=args.max_iter,
                max_evidence=args.max_evidence,
                max_paths=args.max_paths,
                confidence_threshold=args.confidence_threshold,
                constrained=args.constrained,
                max_l3_per_l4=args.max_l3_per_l4,
                max_l2_per_l3=args.max_l2_per_l3,
                max_l1_per_l2=args.max_l1_per_l2,
                max_l0_per_l1=args.max_l0_per_l1,
                max_downward_l0_per_iter=args.max_downward_l0_per_iter,
            )
            t1 = time.perf_counter()
            path_context = build_final_l0_prompt_context(hh_result.get("top_l0_hyperedges", []))
            prompt = build_answer_prompt(
                question=question,
                context=path_context,
                answer_style=args.answer_style,
            )
            answer_data = ask_llm(client, llm_model_name, prompt)
            answer = answer_data["answer"]
            generation = answer_data["raw_generation"]
            em = float(cal_em([golden_answers], [answer]))
            f1 = float(cal_f1([golden_answers], [answer]))
            rsim, rsim_error = maybe_cal_rsim(context_text, hh_result.get("context", ""), args.skip_rsim)
            gen, gen_error = maybe_cal_gen(question, golden_answers, generation, f1, args.skip_gen)
            t2 = time.perf_counter()
            row = {
                "question_index": item["question_index"],
                "question": question,
                "golden_answers": golden_answers,
                "answer": answer,
                "generation": generation,
                "answerable": bool(answer.strip()),
                "confidence": None,
                "selected_count": len(hh_result.get("selected_hyperedges", [])),
                "scores": {"em": em, "f1": f1, "rsim": rsim, "gen": gen},
                "score_errors": {"rsim": rsim_error, "gen": gen_error},
                "timings": {
                    "answer_seconds": round(t1 - t0, 4),
                    "score_seconds": round(t2 - t1, 4),
                    "total_seconds": round(t2 - t0, 4),
                },
                "selected_hyperedges": hh_result.get("selected_hyperedges", []),
                "top_l0_hyperedges": hh_result.get("top_l0_hyperedges", []),
                "anchors": hh_result.get("anchors", []),
                "iterations": hh_result.get("iterations", []),
                "visited_l0_count": hh_result.get("visited_l0_count", 0),
                "visited_count": hh_result.get("visited_count", 0),
                "error": None,
                "error_details": None,
            }
        except Exception as exc:  # pragma: no cover - runtime defensive
            row = {
                "question_index": item["question_index"],
                "question": question,
                "golden_answers": golden_answers,
                "answer": "",
                "generation": "",
                "answerable": False,
                "confidence": 0.0,
                "selected_count": 0,
                "scores": {"em": 0.0, "f1": 0.0, "rsim": None, "gen": None},
                "score_errors": {"rsim": None, "gen": None},
                "timings": {"answer_seconds": 0.0, "score_seconds": 0.0, "total_seconds": 0.0},
                "selected_hyperedges": [],
                "top_l0_hyperedges": [],
                "anchors": [],
                "iterations": [],
                "visited_l0_count": 0,
                "visited_count": 0,
                "error": repr(exc),
                "error_details": build_error_details(exc),
            }
            print(f"  failed: {row['error_details']}", flush=True)

        results["questions"].append(row)
        results["averages"] = compute_averages(results["questions"])
        save_results(results_path, results)
        print(f"  saved -> {results_path}", flush=True)

    print(json.dumps({"output": str(results_path), "averages": results["averages"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
