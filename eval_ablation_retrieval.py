#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval.eval import cal_em, cal_f1
from hh_iterative_eval_runner import build_answer_prompt
from hh_iterative_l2_path_beam_rerank import ApiReranker, LocalReranker, SemanticFallbackReranker, resolve_config
from hh_iterative_retrieval import MultiOrderIterativeRetriever
from layer_answer_compare import ask_llm
from layer_retrieval_compare import load_questions
from Retrieve.lb_hgr_retrieval import load_retrieval_inputs
from strategy_eval_compare import call_with_retry, maybe_cal_rsim, save_results
from hh_iterative_ablation_retrieval import (
    retrieve_layer_wise_topk,
    retrieve_layer_wise_topk_final_rerank,
    retrieve_parallel_path_layer_fusion,
    retrieve_parallel_path_layer_fusion_final_rerank,
    retrieve_path_beam_depth_ablation,
    summarize_path_counts,
)


GEN_METRICS = [
    "comprehensiveness",
    "knowledgeability",
    "correctness",
    "relevance",
    "diversity",
    "logical_coherence",
    "factuality",
]


class NoReranker:
    name = "none"

    def batch_score(self, query: str, evidence_texts: list[str]) -> list[float]:
        return [0.0 for _ in evidence_texts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ES-HiR2 retrieval ablations.")
    parser.add_argument("--retrieval-mode", choices=["layer_wise_topk", "layer_wise_topk_final_rerank", "depth_ablation", "parallel_path_layer_fusion", "parallel_path_layer_fusion_final_rerank"], required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-indices", default="0,1,2,3,4")
    parser.add_argument("--answer-style", choices=["short", "long"], default="long")
    parser.add_argument("--top-k-anchor", type=int, default=2)
    parser.add_argument("--expansion-top-k", type=int, default=2)
    parser.add_argument("--path-beam-size", type=int, default=8)
    parser.add_argument("--top-k-final", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--candidate-cap", type=int, default=None)
    parser.add_argument("--top-k-per-layer", type=int, default=10)
    parser.add_argument("--layer-wise-max-layer", type=int, default=2)
    parser.add_argument("--retrieval-depth", type=int, default=2)
    parser.add_argument("--rerank-base-url", default="")
    parser.add_argument("--rerank-api-key", default="")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--local-rerank-model", default="")
    parser.add_argument("--rerank-device", default="cuda:0")
    parser.add_argument("--local-rerank-max-length", type=int, default=512)
    parser.add_argument("--question-workers", type=int, default=1)
    parser.add_argument("--gen-workers", type=int, default=1)
    parser.add_argument("--semantic-fallback", action="store_true")
    parser.add_argument("--skip-rsim", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--gen-llm-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_questions_any(path: Path) -> list[dict[str, Any]]:
    try:
        return load_questions(path)
    except ValueError:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ("questions", "data", "qa_pairs", "qas", "items", "records", "examples"):
                value = raw.get(key)
                if isinstance(value, list):
                    return value
            for value in raw.values():
                if isinstance(value, list):
                    return value
        raise


def parse_indices(raw: str, total: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(total))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def question_text(raw: dict[str, Any]) -> str:
    return str(raw.get("question", raw.get("问题", ""))).strip()


def question_context(raw: dict[str, Any]) -> str:
    value = raw.get("context", raw.get("chunks", raw.get("raw_content", "")))
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def question_answers(raw: dict[str, Any]) -> list[str]:
    value = raw.get("golden_answers", raw.get("answer", raw.get("回答", [])))
    if isinstance(value, dict):
        value = value.get("内容", value.get("content", ""))
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def maybe_cal_gen_for_eval(
    question: str,
    golden_answers: list[str],
    generation: str,
    f1_score: float,
    skip_gen: bool,
    gen_llm_only: bool,
) -> tuple[float | None, str | None, dict[str, Any] | None]:
    if skip_gen:
        return None, None, None
    try:
        if gen_llm_only:
            from eval.eval_g import cal_gen

            result = call_with_retry("cal_gen_llm_only_via_cal_gen", cal_gen, question, golden_answers, generation, 0.0)
            exp = result.get("explanation", {})
            vals = []
            for item in exp.values():
                if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
                    score = max(0.0, min(1.0, float(item["score"]) * 2.0))
                    item["score"] = score
                    vals.append(score)
            result["score"] = round(statistics.mean(vals), 4) if vals else None
            result["gen_metric"] = "llm_only_without_f1"
            return float(result["score"]), None, result

        from eval.eval_g import cal_gen

        result = call_with_retry("cal_gen", cal_gen, question, golden_answers, generation, f1_score)
        result["gen_metric"] = "llm_judge_mixed_with_f1"
        return float(result["score"]), None, result
    except Exception as exc:
        return None, repr(exc), None


def metric_average(rows: list[dict[str, Any]], metric: str) -> float | None:
    vals = [float((row.get("scores") or {}).get(metric)) for row in rows if isinstance((row.get("scores") or {}).get(metric), (int, float))]
    return round(statistics.mean(vals), 4) if vals else None


def count_average(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float((row.get("retrieval_summary") or {}).get(key)) for row in rows if isinstance((row.get("retrieval_summary") or {}).get(key), (int, float))]
    return round(statistics.mean(vals), 2) if vals else None


def gen_dimension_averages(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in GEN_METRICS:
        vals = []
        for row in rows:
            exp = (row.get("gen_details") or {}).get("explanation", {})
            item = exp.get(metric, {}) if isinstance(exp, dict) else {}
            score = item.get("score") if isinstance(item, dict) else None
            if isinstance(score, (int, float)):
                vals.append(float(score))
        out[metric] = round(statistics.mean(vals), 4) if vals else None
    vals = [v for v in out.values() if isinstance(v, (int, float))]
    out["dataset_avg"] = round(statistics.mean(vals), 4) if vals else None
    return out


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    avg = payload.get("averages", {})
    dim = payload.get("gen_dimension_averages", {})
    lines = [
        "# ES-HiR2 Retrieval Ablation Results",
        "",
        f"settings: `{json.dumps(payload.get('settings', {}), ensure_ascii=False)}`",
        "",
        "| EM | F1 | R-S | G-E | max_stage_candidate_count | visited_l0_count | final_candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {avg.get('em')} | {avg.get('f1')} | {avg.get('rsim')} | {avg.get('gen')} | {avg.get('max_stage_candidate_count')} | {avg.get('visited_l0_count')} | {avg.get('final_evidence_candidate_count')} |",
        "",
        "## G-E Dimension Averages",
        "",
        "| dataset_avg | " + " | ".join(GEN_METRICS) + " |",
        "|---:|" + "|".join(["---:"] * len(GEN_METRICS)) + "|",
        f"| {dim.get('dataset_avg')} | " + " | ".join(str(dim.get(m)) for m in GEN_METRICS) + " |",
        "",
        "## Questions",
        "",
    ]
    for row in payload.get("questions", []):
        scores = row.get("scores") or {}
        summary = row.get("retrieval_summary") or {}
        lines.extend(
            [
                f"### Q{row['question_index']}",
                f"Question: {row['question']}",
                f"Gold: {row['golden_answers']}",
                f"Scores: EM={scores.get('em')}, F1={scores.get('f1')}, R-S={scores.get('rsim')}, G-E={scores.get('gen')}",
                f"Path counts: max_candidate={summary.get('max_stage_candidate_count')}, visited_l0={summary.get('visited_l0_count')}, final_candidates={summary.get('final_evidence_candidate_count')}",
                f"Answer: {str(row.get('answer', '')).replace(chr(10), ' ')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "ablation_results.json"
    output_md = output_dir / "ablation_results.md"

    client, embedding_model, llm_model, base_url, api_key = resolve_config(args)
    graph, text_chunks, hyperedge_vector_map, client, embedding_model = load_retrieval_inputs(args.workdir, client, embedding_model)
    retriever = MultiOrderIterativeRetriever(
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        max_level=max(args.layer_wise_max_layer if args.retrieval_mode in {"layer_wise_topk", "layer_wise_topk_final_rerank", "parallel_path_layer_fusion", "parallel_path_layer_fusion_final_rerank"} else args.retrieval_depth, 2),
    )
    if args.retrieval_mode == "layer_wise_topk":
        reranker = NoReranker()
    elif args.semantic_fallback:
        reranker = SemanticFallbackReranker(retriever, client, embedding_model)
    elif args.local_rerank_model:
        reranker = LocalReranker(
            model_path=args.local_rerank_model,
            device=args.rerank_device,
            batch_size=args.rerank_batch_size,
            max_length=args.local_rerank_max_length,
        )
    else:
        reranker = ApiReranker(
            base_url=args.rerank_base_url or base_url,
            api_key=args.rerank_api_key or api_key,
            model=args.rerank_model,
            batch_size=args.rerank_batch_size,
        )

    questions = load_questions_any(Path(args.questions))
    indices = parse_indices(args.question_indices, total=len(questions))
    os.environ["GEN_EVAL_MAX_WORKERS"] = str(max(1, args.gen_workers))
    settings = {
        "strategy": (
            "Layer-wise Top-K Retrieval"
            if args.retrieval_mode == "layer_wise_topk"
            else "Parallel Path-Layer Fusion + Final Rerank"
            if args.retrieval_mode == "parallel_path_layer_fusion_final_rerank"
            else "Parallel Path-Layer Fusion"
            if args.retrieval_mode == "parallel_path_layer_fusion"
            else f"Path-Beam Depth Ablation L={args.retrieval_depth}"
        ),
        "retrieval_mode": args.retrieval_mode,
        "question_indices": indices,
        "top_k_anchor": args.top_k_anchor,
        "expansion_top_k": args.expansion_top_k,
        "path_beam_size": args.path_beam_size,
        "top_k_final": args.top_k_final,
        "max_iter": args.max_iter,
        "candidate_cap": args.candidate_cap,
        "top_k_per_layer": args.top_k_per_layer,
        "layer_wise_max_layer": args.layer_wise_max_layer,
        "retrieval_depth": args.retrieval_depth,
        "reranker": reranker.name,
        "question_workers": max(1, args.question_workers),
        "gen_workers": max(1, args.gen_workers),
        "answer_style": args.answer_style,
        "gen_metric": "llm_only_without_f1" if args.gen_llm_only else "llm_judge_mixed_with_f1",
    }
    if args.resume and output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        payload["settings"] = settings
    else:
        payload = {"settings": settings, "questions": [], "averages": {}, "gen_dimension_averages": {}}
    save_results(output_json, payload)
    completed = {int(row["question_index"]) for row in payload.get("questions", [])}

    def run_one(order: int, question_index: int) -> dict[str, Any] | None:
        if question_index in completed:
            print(f"[{order}/{len(indices)}] skip completed Q{question_index}", flush=True)
            return None
        raw = questions[question_index]
        question = question_text(raw)
        golden_answers = question_answers(raw)
        gold_context = question_context(raw)
        print(f"[{order}/{len(indices)}] {args.retrieval_mode} Q{question_index}: {question}", flush=True)
        t0 = time.perf_counter()
        if args.retrieval_mode == "layer_wise_topk":
            retrieved = retrieve_layer_wise_topk(
                retriever=retriever,
                query=question,
                top_k_per_layer=args.top_k_per_layer,
                top_k_final=args.top_k_final,
                max_layer=args.layer_wise_max_layer,
            )
        elif args.retrieval_mode == "layer_wise_topk_final_rerank":
            retrieved = retrieve_layer_wise_topk_final_rerank(
                retriever=retriever,
                query=question,
                reranker=reranker,
                top_k_per_layer=args.top_k_per_layer,
                top_k_final=args.top_k_final,
                max_layer=args.layer_wise_max_layer,
            )
        elif args.retrieval_mode == "parallel_path_layer_fusion":
            retrieved = retrieve_parallel_path_layer_fusion(
                retriever=retriever,
                query=question,
                reranker=reranker,
                top_k_anchor=args.top_k_anchor,
                expansion_top_k=args.expansion_top_k,
                path_beam_size=args.path_beam_size,
                top_k_final=args.top_k_final,
                max_iter=args.max_iter,
                candidate_cap=args.candidate_cap,
                top_k_per_layer=args.top_k_per_layer,
                layer_wise_max_layer=args.layer_wise_max_layer,
            )
        elif args.retrieval_mode == "parallel_path_layer_fusion_final_rerank":
            retrieved = retrieve_parallel_path_layer_fusion_final_rerank(
                retriever=retriever,
                query=question,
                reranker=reranker,
                top_k_anchor=args.top_k_anchor,
                expansion_top_k=args.expansion_top_k,
                path_beam_size=args.path_beam_size,
                top_k_final=args.top_k_final,
                max_iter=args.max_iter,
                candidate_cap=args.candidate_cap,
                top_k_per_layer=args.top_k_per_layer,
                layer_wise_max_layer=args.layer_wise_max_layer,
            )
        else:
            retrieved = retrieve_path_beam_depth_ablation(
                retriever=retriever,
                query=question,
                reranker=reranker,
                retrieval_depth=args.retrieval_depth,
                top_k_anchor=args.top_k_anchor,
                expansion_top_k=args.expansion_top_k,
                path_beam_size=args.path_beam_size,
                top_k_final=args.top_k_final,
                max_iter=args.max_iter,
                candidate_cap=args.candidate_cap,
            )
        prompt = build_answer_prompt(question, retrieved["context"], args.answer_style)
        answer_data = ask_llm(client, llm_model, prompt)
        answer = answer_data["answer"]
        generation = answer_data["raw_generation"]
        em = float(cal_em([golden_answers], [answer]))
        f1 = float(cal_f1([golden_answers], [answer]))
        rsim, rsim_error = maybe_cal_rsim(gold_context, retrieved["context"], args.skip_rsim)
        gen, gen_error, gen_details = maybe_cal_gen_for_eval(question, golden_answers, generation, f1, args.skip_gen, args.gen_llm_only)
        t1 = time.perf_counter()
        return {
            "question_index": question_index,
            "question": question,
            "golden_answers": golden_answers,
            "answer": answer,
            "generation": generation,
            "scores": {"em": em, "f1": f1, "rsim": rsim, "gen": gen},
            "score_errors": {"rsim": rsim_error, "gen": gen_error},
            "gen_details": gen_details,
            "timings": {"total_seconds": round(t1 - t0, 4)},
            "retrieval_summary": {
                **summarize_path_counts(retrieved),
                **(retrieved.get("fusion_log") or {}),
            },
            "retrieval": retrieved,
        }

    def safe_run_one(order: int, question_index: int) -> dict[str, Any] | None:
        try:
            return run_one(order, question_index)
        except Exception as exc:
            print(f"  [error] question_index={question_index} failed: {type(exc).__name__}: {exc}", flush=True)
            return {
                "question_index": question_index,
                "question": question_text(questions[question_index]) if 0 <= question_index < len(questions) else "",
                "golden_answers": question_answers(questions[question_index]) if 0 <= question_index < len(questions) else [],
                "answer": "",
                "generation": "",
                "scores": {"em": 0.0, "f1": 0.0, "rsim": None, "gen": None},
                "score_errors": {"rsim": type(exc).__name__, "gen": type(exc).__name__},
                "gen_details": None,
                "timings": {"total_seconds": 0.0},
                "retrieval_summary": {},
                "retrieval": {"error": f"{type(exc).__name__}: {exc}"},
            }

    pending = [(order, idx) for order, idx in enumerate(indices, start=1) if idx not in completed]
    workers = max(1, args.question_workers)
    if workers == 1:
        row_iter = (safe_run_one(order, idx) for order, idx in pending)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(safe_run_one, order, idx) for order, idx in pending]
        row_iter = (future.result() for future in as_completed(futures))

    try:
        for row in row_iter:
            if row is None:
                continue
            payload["questions"].append(row)
            payload["questions"].sort(key=lambda item: int(item["question_index"]))
            completed.add(int(row["question_index"]))
            rows = payload["questions"]
            payload["averages"] = {
                "em": metric_average(rows, "em"),
                "f1": metric_average(rows, "f1"),
                "rsim": metric_average(rows, "rsim"),
                "gen": metric_average(rows, "gen"),
                "max_stage_candidate_count": count_average(rows, "max_stage_candidate_count"),
                "visited_l0_count": count_average(rows, "visited_l0_count"),
                "final_evidence_candidate_count": count_average(rows, "final_evidence_candidate_count"),
            }
            payload["gen_dimension_averages"] = gen_dimension_averages(rows)
            save_results(output_json, payload)
            write_markdown(payload, output_md)
            print(f"  saved -> {output_json}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "averages": payload["averages"], "gen_dimension_averages": payload.get("gen_dimension_averages")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
