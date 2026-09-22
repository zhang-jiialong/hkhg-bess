#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立实验脚本：
1. original_proh
   - 直接调用原始 HKHG 问答链路
2. layer_threshold_expand_l0_top20
   - 各层先阈值匹配
   - 展开到 L0
   - 每层取 top 20
   - 按复用的最终问答 prompt 风格回答
3. direct_l0_top100
   - 直接在 L0 取 top 100
   - 按同样的问答 prompt 风格回答

并自动对每个策略进行评分：
- EM
- F1
- R-Sim
- Gen

说明：
- 不修改项目主流程源码
- 支持随机抽取若干问题
- 支持跳过 R-Sim / Gen 以加快实验速度
- 结果会边跑边写入 JSON，避免中途中断丢失
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, RateLimitError, Timeout

from HKHG import HKHG
from HKHG.llm import summarize_openai_error
from eval.eval import cal_em, cal_f1
from layer_answer_compare import (
    ask_llm,
    build_direct_l0_strategy_context,
    build_direct_prompt_context,
    build_entity_context,
    build_layer_disjoint_strategy_context,
    build_layer_prompt_context,
    build_layer_strategy_context,
    build_output_dir as _unused_build_output_dir,
    build_prompt,
)
from layer_retrieval_compare import (
    embed_texts,
    group_hyperedges_by_level,
    load_graph,
    load_hyperedge_vectors,
    load_json,
    load_questions,
    load_runtime_config,
    make_openai_client,
    sample_questions,
)
from hh_iterative_retrieval import hierarchical_reasoning_path_retrieve
from strategy_eval_compare_html import build_html


STRATEGIES = [
    "original_proh",
    "layer_threshold_expand_l0_top20",
    "layer_disjoint_l0_top20",
    "HHIterative-Retrieval",
    "direct_l0_top100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三种问答策略对比与自动评分脚本")
    parser.add_argument("--data-source", default="hypertension")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--question-indices",
        default="",
        help="只运行指定题号，逗号分隔，例如 163,28,6",
    )
    parser.add_argument(
        "--question-concurrency",
        type=int,
        default=10,
        help="同时处理的题目数，默认 10",
    )
    parser.add_argument(
        "--inner-concurrency",
        type=int,
        default=10,
        help="单题内部 HKHG 查询并发，默认 10",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--layer-top-k", type=int, default=20)
    parser.add_argument("--direct-top-k", type=int, default=100)
    parser.add_argument("--hh-top-n-seed", type=int, default=5)
    parser.add_argument("--hh-top-p-parent", type=int, default=2)
    parser.add_argument("--hh-top-k-expand", type=int, default=30)
    parser.add_argument("--hh-max-iter", type=int, default=3)
    parser.add_argument("--hh-max-evidence", type=int, default=20)
    parser.add_argument("--hh-max-paths", type=int, default=10)
    parser.add_argument("--hh-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--path-token-budget", type=int, default=10000)
    parser.add_argument("--entity-token-budget", type=int, default=5000)
    parser.add_argument("--run-ts", default="")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-rsim",
        action="store_true",
        help="跳过 R-Sim 评分，加快实验速度",
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="跳过 Gen 评分，加快实验速度",
    )
    return parser.parse_args()


def build_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    run_ts = args.run_ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        return Path(args.output_dir)
    return repo_root / "results" / f"{args.data_source}_strategy_eval_compare" / run_ts


def parse_question_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def select_questions(
    questions: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    question_indices: list[int],
) -> list[dict[str, Any]]:
    indexed = sample_questions(questions, len(questions), seed=seed)
    if question_indices:
        selected = [item for item in indexed if item["question_index"] in question_indices]
        missing = sorted(set(question_indices) - {item["question_index"] for item in selected})
        if missing:
            raise ValueError(f"题号不存在: {missing}")
        return selected
    return sample_questions(questions, sample_size, seed)


def extract_answer(generation: str, fallback: str = "") -> str:
    try:
        return generation.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception:
        return fallback or generation.strip()


def build_knowledge_from_retrieved(retrieved: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in retrieved:
        parts.append(str(item.get("reasoning_path", "")).strip())
        parts.append(str(item.get("src_text_chunks", "")).strip())
        parts.append(str(item.get("entity_descriptions", "")).strip())
    return "\n".join(part for part in parts if part)


def iter_exception_chain(exc: BaseException | None):
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def is_retryable_exception(exc: BaseException) -> bool:
    for item in iter_exception_chain(exc):
        if isinstance(item, (RateLimitError, APIConnectionError, Timeout)):
            return True
        status_code = getattr(item, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(item, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        message = str(item).lower()
        if "too many requests" in message or "rate limit" in message:
            return True
    return False


def build_error_details(exc: BaseException) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    for item in iter_exception_chain(exc):
        chain.append(
            {
                "type": item.__class__.__name__,
                "summary": summarize_openai_error(item),
            }
        )
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "chain": chain,
    }


def call_with_retry(
    label: str,
    func,
    *args,
    attempts: int = 5,
    base_sleep_seconds: float = 4.0,
    **kwargs,
):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime defensive
            last_exc = exc
            if not is_retryable_exception(exc) or attempt >= attempts:
                raise
            sleep_seconds = min(base_sleep_seconds * (2 ** max(attempt - 1, 0)), 60.0)
            print(
                f"    [retry] {label} attempt {attempt}/{attempts} -> {summarize_openai_error(exc)}; sleep {sleep_seconds:.1f}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise last_exc  # pragma: no cover - defensive


def embed_texts_with_retry(client, model: str, texts: list[str]) -> Any:
    return call_with_retry("embed_texts", embed_texts, client, model, texts)


def ask_llm_with_retry(client, model_name: str, prompt: str) -> dict[str, Any]:
    return call_with_retry("ask_llm", ask_llm, client, model_name, prompt)


def maybe_cal_rsim(context_text: str, knowledge: str, skip_rsim: bool) -> tuple[float | None, str | None]:
    if skip_rsim:
        return None, None
    try:
        from eval.eval_r import cal_rsim

        return float(cal_rsim([context_text], [knowledge])) if knowledge else 0.0, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, repr(exc)


def maybe_cal_gen(
    question: str,
    golden_answers: list[str],
    generation: str,
    f1_score: float,
    skip_gen: bool,
) -> tuple[float | None, str | None]:
    if skip_gen:
        return None, None
    try:
        from eval.eval_g import cal_gen

        result = call_with_retry(
            "cal_gen",
            cal_gen,
            question,
            golden_answers,
            generation,
            f1_score,
        )
        return float(result["score"]), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, repr(exc)


def compute_averages(question_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        metric_values: dict[str, list[float]] = {
            "em": [],
            "f1": [],
            "rsim": [],
            "gen": [],
            "answer_seconds": [],
            "score_seconds": [],
            "total_seconds": [],
        }
        for item in question_results:
            strategy_data = item["strategies"][strategy]
            scores = strategy_data["scores"]
            timings = strategy_data["timings"]
            for metric in ["em", "f1", "rsim", "gen"]:
                value = scores.get(metric)
                if isinstance(value, (int, float)):
                    metric_values[metric].append(float(value))
            for metric in ["answer_seconds", "score_seconds", "total_seconds"]:
                value = timings.get(metric)
                if isinstance(value, (int, float)):
                    metric_values[metric].append(float(value))
        summary[strategy] = {
            metric: round(statistics.mean(values), 4) if values else None
            for metric, values in metric_values.items()
        }
    return summary


def save_results(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


async def run_original_strategy(rag: HKHG, question: str) -> dict[str, Any]:
    return await rag.aquery_reasoning(question)


def build_failed_strategy_result(error: str, error_details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "answer": "",
        "generation": "",
        "selected_count": 0,
        "scores": {
            "em": 0.0,
            "f1": 0.0,
            "rsim": None,
            "gen": None,
        },
        "score_errors": {
            "rsim": None,
            "gen": None,
        },
        "timings": {
            "answer_seconds": 0.0,
            "score_seconds": 0.0,
            "total_seconds": 0.0,
        },
        "error": error,
        "error_details": error_details,
    }


def run_layer_strategy_sync(
    *,
    question: str,
    golden_answers: list[str],
    context_text: str,
    client,
    llm_model_name: str,
    graph,
    text_chunks,
    grouped_hyperedges,
    hyperedge_vector_map,
    query_vec,
    threshold: float,
    max_level: int,
    layer_top_k: int,
    path_token_budget: int,
    entity_token_budget: int,
    skip_rsim: bool,
    skip_gen: bool,
) -> dict[str, Any]:
    layer_t0 = time.perf_counter()
    layer_ctx = build_layer_strategy_context(
        graph=graph,
        text_chunks=text_chunks,
        grouped_hyperedges=grouped_hyperedges,
        hyperedge_vector_map=hyperedge_vector_map,
        query_vec=query_vec,
        threshold=threshold,
        max_level=max_level,
        top_k=layer_top_k,
    )
    layer_path_context = build_layer_prompt_context(
        per_level=layer_ctx["per_level"],
        model_name=llm_model_name,
        token_budget=path_token_budget,
    )
    layer_entity_context = build_entity_context(
        entity_lines=layer_ctx["entity_lines"],
        model_name=llm_model_name,
        token_budget=entity_token_budget,
    )
    layer_prompt = build_prompt(question, layer_path_context, layer_entity_context)
    layer_answer_data = ask_llm_with_retry(client, llm_model_name, layer_prompt)
    layer_t1 = time.perf_counter()
    layer_answer = extract_answer(layer_answer_data["raw_generation"], layer_answer_data["answer"])
    layer_knowledge = build_knowledge_from_retrieved(
        [
            {
                "reasoning_path": layer_path_context,
                "src_text_chunks": "\n".join(layer_ctx["chunk_texts"]),
                "entity_descriptions": layer_entity_context,
            }
        ]
    )
    layer_em = float(cal_em([golden_answers], [layer_answer]))
    layer_f1 = float(cal_f1([golden_answers], [layer_answer]))
    layer_rsim, layer_rsim_error = maybe_cal_rsim(context_text, layer_knowledge, skip_rsim)
    layer_gen, layer_gen_error = maybe_cal_gen(
        question,
        golden_answers,
        layer_answer_data["raw_generation"],
        layer_f1,
        skip_gen,
    )
    layer_t2 = time.perf_counter()
    return {
        "answer": layer_answer,
        "generation": layer_answer_data["raw_generation"],
        "selected_count": {k: len(v["selected_l0_hyperedges"]) for k, v in layer_ctx["per_level"].items()},
        "scores": {
            "em": layer_em,
            "f1": layer_f1,
            "rsim": layer_rsim,
            "gen": layer_gen,
        },
        "score_errors": {
            "rsim": layer_rsim_error,
            "gen": layer_gen_error,
        },
        "timings": {
            "answer_seconds": round(layer_t1 - layer_t0, 4),
            "score_seconds": round(layer_t2 - layer_t1, 4),
            "total_seconds": round(layer_t2 - layer_t0, 4),
        },
        "all_selected_l0_count": len(layer_ctx["all_selected_l0_hyperedges"]),
        "error": None,
        "error_details": None,
    }


def run_layer_disjoint_strategy_sync(
    *,
    question: str,
    golden_answers: list[str],
    context_text: str,
    client,
    llm_model_name: str,
    graph,
    text_chunks,
    grouped_hyperedges,
    hyperedge_vector_map,
    query_vec,
    threshold: float,
    max_level: int,
    layer_top_k: int,
    path_token_budget: int,
    entity_token_budget: int,
    skip_rsim: bool,
    skip_gen: bool,
) -> dict[str, Any]:
    layer_t0 = time.perf_counter()
    layer_ctx = build_layer_disjoint_strategy_context(
        graph=graph,
        text_chunks=text_chunks,
        grouped_hyperedges=grouped_hyperedges,
        hyperedge_vector_map=hyperedge_vector_map,
        query_vec=query_vec,
        threshold=threshold,
        max_level=max_level,
        top_k=layer_top_k,
    )
    layer_path_context = build_layer_prompt_context(
        per_level=layer_ctx["per_level"],
        model_name=llm_model_name,
        token_budget=path_token_budget,
    )
    layer_entity_context = build_entity_context(
        entity_lines=layer_ctx["entity_lines"],
        model_name=llm_model_name,
        token_budget=entity_token_budget,
    )
    layer_prompt = build_prompt(question, layer_path_context, layer_entity_context)
    layer_answer_data = ask_llm_with_retry(client, llm_model_name, layer_prompt)
    layer_t1 = time.perf_counter()
    layer_answer = extract_answer(layer_answer_data["raw_generation"], layer_answer_data["answer"])
    layer_knowledge = build_knowledge_from_retrieved(
        [
            {
                "reasoning_path": layer_path_context,
                "src_text_chunks": "\n".join(layer_ctx["chunk_texts"]),
                "entity_descriptions": layer_entity_context,
            }
        ]
    )
    layer_em = float(cal_em([golden_answers], [layer_answer]))
    layer_f1 = float(cal_f1([golden_answers], [layer_answer]))
    layer_rsim, layer_rsim_error = maybe_cal_rsim(context_text, layer_knowledge, skip_rsim)
    layer_gen, layer_gen_error = maybe_cal_gen(
        question,
        golden_answers,
        layer_answer_data["raw_generation"],
        layer_f1,
        skip_gen,
    )
    layer_t2 = time.perf_counter()
    return {
        "answer": layer_answer,
        "generation": layer_answer_data["raw_generation"],
        "selected_count": {k: len(v["selected_l0_hyperedges"]) for k, v in layer_ctx["per_level"].items()},
        "scores": {
            "em": layer_em,
            "f1": layer_f1,
            "rsim": layer_rsim,
            "gen": layer_gen,
        },
        "score_errors": {
            "rsim": layer_rsim_error,
            "gen": layer_gen_error,
        },
        "timings": {
            "answer_seconds": round(layer_t1 - layer_t0, 4),
            "score_seconds": round(layer_t2 - layer_t1, 4),
            "total_seconds": round(layer_t2 - layer_t0, 4),
        },
        "all_selected_l0_count": len(layer_ctx["all_selected_l0_hyperedges"]),
        "error": None,
        "error_details": None,
    }


def run_direct_strategy_sync(
    *,
    question: str,
    golden_answers: list[str],
    context_text: str,
    client,
    llm_model_name: str,
    grouped_hyperedges,
    hyperedge_vector_map,
    graph,
    text_chunks,
    query_vec,
    direct_top_k: int,
    path_token_budget: int,
    entity_token_budget: int,
    skip_rsim: bool,
    skip_gen: bool,
) -> dict[str, Any]:
    direct_t0 = time.perf_counter()
    direct_ctx = build_direct_l0_strategy_context(
        grouped_hyperedges=grouped_hyperedges,
        hyperedge_vector_map=hyperedge_vector_map,
        graph=graph,
        text_chunks=text_chunks,
        query_vec=query_vec,
        top_k=direct_top_k,
    )
    direct_path_context = build_direct_prompt_context(
        ranked_l0=direct_ctx["selected_l0_scores"],
        model_name=llm_model_name,
        token_budget=path_token_budget,
    )
    direct_entity_context = build_entity_context(
        entity_lines=direct_ctx["entity_lines"],
        model_name=llm_model_name,
        token_budget=entity_token_budget,
    )
    direct_prompt = build_prompt(question, direct_path_context, direct_entity_context)
    direct_answer_data = ask_llm_with_retry(client, llm_model_name, direct_prompt)
    direct_t1 = time.perf_counter()
    direct_answer = extract_answer(direct_answer_data["raw_generation"], direct_answer_data["answer"])
    direct_knowledge = build_knowledge_from_retrieved(
        [
            {
                "reasoning_path": direct_path_context,
                "src_text_chunks": "\n".join(direct_ctx["chunk_texts"]),
                "entity_descriptions": direct_entity_context,
            }
        ]
    )
    direct_em = float(cal_em([golden_answers], [direct_answer]))
    direct_f1 = float(cal_f1([golden_answers], [direct_answer]))
    direct_rsim, direct_rsim_error = maybe_cal_rsim(context_text, direct_knowledge, skip_rsim)
    direct_gen, direct_gen_error = maybe_cal_gen(
        question,
        golden_answers,
        direct_answer_data["raw_generation"],
        direct_f1,
        skip_gen,
    )
    direct_t2 = time.perf_counter()
    return {
        "answer": direct_answer,
        "generation": direct_answer_data["raw_generation"],
        "selected_count": len(direct_ctx["selected_l0_hyperedges"]),
        "scores": {
            "em": direct_em,
            "f1": direct_f1,
            "rsim": direct_rsim,
            "gen": direct_gen,
        },
        "score_errors": {
            "rsim": direct_rsim_error,
            "gen": direct_gen_error,
        },
        "timings": {
            "answer_seconds": round(direct_t1 - direct_t0, 4),
            "score_seconds": round(direct_t2 - direct_t1, 4),
            "total_seconds": round(direct_t2 - direct_t0, 4),
        },
        "all_selected_l0_count": len(direct_ctx["selected_l0_hyperedges"]),
        "error": None,
        "error_details": None,
    }


def run_hh_iterative_strategy_sync(
    *,
    question: str,
    golden_answers: list[str],
    context_text: str,
    client,
    embedding_model: str,
    llm_model_name: str,
    graph,
    text_chunks,
    hyperedge_vector_map,
    max_level: int,
    top_n_seed: int,
    top_p_parent: int,
    top_k_expand: int,
    max_iter: int,
    max_evidence: int,
    max_paths: int,
    confidence_threshold: float,
    skip_rsim: bool,
    skip_gen: bool,
) -> dict[str, Any]:
    hh_t0 = time.perf_counter()
    hh_result = hierarchical_reasoning_path_retrieve(
        query=question,
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        llm_model_name=llm_model_name,
        top_n_seed=top_n_seed,
        top_p_parent=top_p_parent,
        top_k_expand=top_k_expand,
        max_level=max_level,
        max_iter=max_iter,
        max_evidence=max_evidence,
        max_paths=max_paths,
        confidence_threshold=confidence_threshold,
    )
    hh_t1 = time.perf_counter()
    hh_prompt = build_prompt(
        question=question,
        path_context=hh_result.get("context", ""),
        entity_context="",
    )
    hh_answer_data = ask_llm_with_retry(client, llm_model_name, hh_prompt)
    hh_answer = extract_answer(hh_answer_data["raw_generation"], hh_answer_data["answer"])
    hh_generation = hh_answer_data["raw_generation"]
    hh_knowledge = hh_result.get("context", "")
    hh_em = float(cal_em([golden_answers], [hh_answer]))
    hh_f1 = float(cal_f1([golden_answers], [hh_answer]))
    hh_rsim, hh_rsim_error = maybe_cal_rsim(context_text, hh_knowledge, skip_rsim)
    hh_gen, hh_gen_error = maybe_cal_gen(
        question,
        golden_answers,
        hh_generation,
        hh_f1,
        skip_gen,
    )
    hh_t2 = time.perf_counter()
    return {
        "answer": hh_answer,
        "generation": hh_generation,
        "answerable": bool(hh_answer.strip()),
        "confidence": None,
        "selected_count": len(hh_result.get("selected_hyperedges", [])),
        "scores": {
            "em": hh_em,
            "f1": hh_f1,
            "rsim": hh_rsim,
            "gen": hh_gen,
        },
        "score_errors": {
            "rsim": hh_rsim_error,
            "gen": hh_gen_error,
        },
        "timings": {
            "answer_seconds": round(hh_t1 - hh_t0, 4),
            "score_seconds": round(hh_t2 - hh_t1, 4),
            "total_seconds": round(hh_t2 - hh_t0, 4),
        },
        "selected_hyperedges": hh_result.get("selected_hyperedges", []),
        "top_l0_hyperedges": hh_result.get("top_l0_hyperedges", []),
        "anchors": hh_result.get("anchors", []),
        "iterations": hh_result.get("iterations", []),
        "visited_l0_count": hh_result.get("visited_l0_count", 0),
        "visited_count": hh_result.get("visited_count", 0),
        "context": hh_result.get("context", ""),
        "error": None,
        "error_details": None,
    }


async def process_question_async(
    *,
    idx: int,
    total: int,
    item: dict[str, Any],
    query_vec,
    rag: HKHG,
    client,
    embedding_model: str,
    llm_model_name: str,
    graph,
    text_chunks,
    grouped_hyperedges,
    hyperedge_vector_map,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = item["question"]
    golden_answers = item["raw"].get("golden_answers", [])
    context_list = item["raw"].get("context", [])
    context_text = "\n".join(context_list)

    print(f"[{idx}/{total}] Question #{item['question_index']}: {question}", flush=True)

    question_result: dict[str, Any] = {
        "question_index": item["question_index"],
        "question": question,
        "golden_answers": golden_answers,
        "context_count": len(context_list),
        "strategies": {},
    }

    print("  - running original_proh", flush=True)
    try:
        t0 = time.perf_counter()
        original = await run_original_strategy(rag, question)
        t1 = time.perf_counter()
        original_generation = original.get("generation", "") or original.get("gen_answer", "")
        original_answer = extract_answer(original_generation, original.get("gen_answer", ""))
        original_knowledge = build_knowledge_from_retrieved(original.get("retrieved", []))
        original_em = float(cal_em([golden_answers], [original_answer]))
        original_f1 = float(cal_f1([golden_answers], [original_answer]))
        original_rsim, original_rsim_error = await asyncio.to_thread(
            maybe_cal_rsim, context_text, original_knowledge, args.skip_rsim
        )
        original_gen, original_gen_error = await asyncio.to_thread(
            maybe_cal_gen,
            question,
            golden_answers,
            original_generation,
            original_f1,
            args.skip_gen,
        )
        t2 = time.perf_counter()
        question_result["strategies"]["original_proh"] = {
            "answer": original_answer,
            "generation": original_generation,
            "selected_count": len(original.get("retrieved", [])),
            "scores": {
                "em": original_em,
                "f1": original_f1,
                "rsim": original_rsim,
                "gen": original_gen,
            },
            "score_errors": {
                "rsim": original_rsim_error,
                "gen": original_gen_error,
            },
            "timings": {
                "answer_seconds": round(t1 - t0, 4),
                "score_seconds": round(t2 - t1, 4),
                "total_seconds": round(t2 - t0, 4),
            },
            "retrieved_count": len(original.get("retrieved", [])),
            "error": None,
            "error_details": None,
        }
    except Exception as exc:  # pragma: no cover - runtime defensive
        error_details = build_error_details(exc)
        print(f"    original_proh failed: {error_details}", flush=True)
        question_result["strategies"]["original_proh"] = build_failed_strategy_result(
            repr(exc),
            error_details=error_details,
        )

    print("  - building layer_threshold_expand_l0_top20", flush=True)
    try:
        question_result["strategies"]["layer_threshold_expand_l0_top20"] = await asyncio.to_thread(
            run_layer_strategy_sync,
            question=question,
            golden_answers=golden_answers,
            context_text=context_text,
            client=client,
            llm_model_name=llm_model_name,
            graph=graph,
            text_chunks=text_chunks,
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            query_vec=query_vec,
            threshold=args.threshold,
            max_level=args.max_level,
            layer_top_k=args.layer_top_k,
            path_token_budget=args.path_token_budget,
            entity_token_budget=args.entity_token_budget,
            skip_rsim=args.skip_rsim,
            skip_gen=args.skip_gen,
        )
    except Exception as exc:  # pragma: no cover - runtime defensive
        error_details = build_error_details(exc)
        print(f"    layer_threshold_expand_l0_top20 failed: {error_details}", flush=True)
        question_result["strategies"]["layer_threshold_expand_l0_top20"] = build_failed_strategy_result(
            repr(exc),
            error_details=error_details,
        )

    print("  - building layer_disjoint_l0_top20", flush=True)
    try:
        question_result["strategies"]["layer_disjoint_l0_top20"] = await asyncio.to_thread(
            run_layer_disjoint_strategy_sync,
            question=question,
            golden_answers=golden_answers,
            context_text=context_text,
            client=client,
            llm_model_name=llm_model_name,
            graph=graph,
            text_chunks=text_chunks,
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            query_vec=query_vec,
            threshold=args.threshold,
            max_level=args.max_level,
            layer_top_k=args.layer_top_k,
            path_token_budget=args.path_token_budget,
            entity_token_budget=args.entity_token_budget,
            skip_rsim=args.skip_rsim,
            skip_gen=args.skip_gen,
        )
    except Exception as exc:  # pragma: no cover - runtime defensive
        error_details = build_error_details(exc)
        print(f"    layer_disjoint_l0_top20 failed: {error_details}", flush=True)
        question_result["strategies"]["layer_disjoint_l0_top20"] = build_failed_strategy_result(
            repr(exc),
            error_details=error_details,
        )

    print("  - building direct_l0_top100", flush=True)
    try:
        question_result["strategies"]["direct_l0_top100"] = await asyncio.to_thread(
            run_direct_strategy_sync,
            question=question,
            golden_answers=golden_answers,
            context_text=context_text,
            client=client,
            llm_model_name=llm_model_name,
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            graph=graph,
            text_chunks=text_chunks,
            query_vec=query_vec,
            direct_top_k=args.direct_top_k,
            path_token_budget=args.path_token_budget,
            entity_token_budget=args.entity_token_budget,
            skip_rsim=args.skip_rsim,
            skip_gen=args.skip_gen,
        )
    except Exception as exc:  # pragma: no cover - runtime defensive
        error_details = build_error_details(exc)
        print(f"    direct_l0_top100 failed: {error_details}", flush=True)
        question_result["strategies"]["direct_l0_top100"] = build_failed_strategy_result(
            repr(exc),
            error_details=error_details,
        )

    print("  - running HHIterative-Retrieval", flush=True)
    try:
        question_result["strategies"]["HHIterative-Retrieval"] = await asyncio.to_thread(
            run_hh_iterative_strategy_sync,
            question=question,
            golden_answers=golden_answers,
            context_text=context_text,
            client=client,
            embedding_model=embedding_model,
            llm_model_name=llm_model_name,
            graph=graph,
            text_chunks=text_chunks,
            hyperedge_vector_map=hyperedge_vector_map,
            max_level=args.max_level,
            top_n_seed=args.hh_top_n_seed,
            top_p_parent=args.hh_top_p_parent,
            top_k_expand=args.hh_top_k_expand,
            max_iter=args.hh_max_iter,
            max_evidence=args.hh_max_evidence,
            max_paths=args.hh_max_paths,
            confidence_threshold=args.hh_confidence_threshold,
            skip_rsim=args.skip_rsim,
            skip_gen=args.skip_gen,
        )
    except Exception as exc:  # pragma: no cover - runtime defensive
        error_details = build_error_details(exc)
        print(f"    HHIterative-Retrieval failed: {error_details}", flush=True)
        question_result["strategies"]["HHIterative-Retrieval"] = build_failed_strategy_result(
            repr(exc),
            error_details=error_details,
        )

    return question_result


async def run_all_questions_async(
    *,
    sampled_questions: list[dict[str, Any]],
    query_vectors,
    rag: HKHG,
    client,
    embedding_model: str,
    llm_model_name: str,
    graph,
    text_chunks,
    grouped_hyperedges,
    hyperedge_vector_map,
    args: argparse.Namespace,
    results: dict[str, Any],
    results_path: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(args.question_concurrency, 1))
    save_lock = asyncio.Lock()
    ordered_results: list[dict[str, Any] | None] = [None] * len(sampled_questions)

    async def worker(order: int, item: dict[str, Any], query_vec):
        async with semaphore:
            result = await process_question_async(
                idx=order + 1,
                total=len(sampled_questions),
                item=item,
                query_vec=query_vec,
                rag=rag,
                client=client,
                embedding_model=embedding_model,
                llm_model_name=llm_model_name,
                graph=graph,
                text_chunks=text_chunks,
                grouped_hyperedges=grouped_hyperedges,
                hyperedge_vector_map=hyperedge_vector_map,
                args=args,
            )
        ordered_results[order] = result
        async with save_lock:
            completed = [value for value in ordered_results if value is not None]
            results["questions"] = completed
            results["averages"] = compute_averages(completed)
            save_results(results_path, results)
            print(f"  - saved -> {results_path}", flush=True)

    tasks = [
        asyncio.create_task(worker(order, item, query_vec))
        for order, (item, query_vec) in enumerate(zip(sampled_questions, query_vectors))
    ]
    await asyncio.gather(*tasks)
    return [value for value in ordered_results if value is not None]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    print("[init] parsing arguments", flush=True)

    workdir = Path(args.workdir) if args.workdir else repo_root / "data" / "work" / args.data_source
    question_path = (
        Path(args.questions)
        if args.questions
        else repo_root / "questions" / args.data_source / "questions.json"
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = build_output_dir(repo_root, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "strategy_eval_compare_results.json"
    html_path = output_dir / "strategy_eval_compare_view.html"

    graph_path = workdir / "graph_chunk_entity_relation.graphml"
    vdb_hyperedges_path = workdir / "vdb_hyperedges.json"
    text_chunks_path = workdir / "kv_store_text_chunks.json"

    for path in [graph_path, vdb_hyperedges_path, question_path, config_path, text_chunks_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件: {path}")

    print("[init] loading runtime config", flush=True)
    runtime_config = load_runtime_config(config_path)
    llm_cfg = runtime_config.get("llm", {})
    emb_cfg = runtime_config.get("embedding", {})
    print("[init] creating API clients", flush=True)
    client, embedding_model = make_openai_client(runtime_config)
    llm_model_name = llm_cfg.get("model", "gpt-4o-mini")

    print("[init] initializing original HKHG instance", flush=True)
    rag = HKHG(
        working_dir=str(workdir),
        log_level="INFO",
        llm_model_max_async=args.inner_concurrency,
        max_concurrency=args.inner_concurrency,
        embedding_func_max_async=args.inner_concurrency,
        llm_model_name=llm_model_name,
        llm_model_kwargs={
            "base_url": llm_cfg.get("base_url"),
            "api_key": llm_cfg.get("api_key"),
        },
        embedding_model_name=emb_cfg.get("model_path"),
        embedding_model_device=emb_cfg.get("device"),
    )

    print("[init] loading graph", flush=True)
    graph = load_graph(graph_path)
    print("[init] loading hyperedge vectors", flush=True)
    records = load_hyperedge_vectors(vdb_hyperedges_path)
    print("[init] grouping hyperedges by level", flush=True)
    grouped_hyperedges = group_hyperedges_by_level(graph, records, args.max_level)
    hyperedge_vector_map = {item["hyperedge_name"]: item["vector"] for item in records}
    print("[init] loading text chunks", flush=True)
    text_chunks = load_json(text_chunks_path) or {}

    print("[init] loading questions", flush=True)
    questions = load_questions(question_path)
    print("[init] sampling questions", flush=True)
    selected_indices = parse_question_indices(args.question_indices)
    sampled_questions = select_questions(questions, args.sample_size, args.seed, selected_indices)
    print(f"[init] sampled {len(sampled_questions)} question(s)", flush=True)
    print("[init] embedding sampled questions", flush=True)
    query_vectors = embed_texts_with_retry(
        client=client,
        model=embedding_model,
        texts=[item["question"] for item in sampled_questions],
    )

    results: dict[str, Any] = {
        "data_source": args.data_source,
        "sample_size": len(sampled_questions),
        "seed": args.seed,
        "question_indices": selected_indices,
        "question_concurrency": args.question_concurrency,
        "inner_concurrency": args.inner_concurrency,
        "threshold": args.threshold,
        "layer_top_k": args.layer_top_k,
        "direct_top_k": args.direct_top_k,
        "skip_rsim": args.skip_rsim,
        "skip_gen": args.skip_gen,
        "embedding_model": embedding_model,
        "llm_model": llm_model_name,
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "questions": [],
        "averages": {},
    }

    save_results(results_path, results)

    results["questions"] = asyncio.run(
        run_all_questions_async(
            sampled_questions=sampled_questions,
            query_vectors=query_vectors,
            rag=rag,
            client=client,
            embedding_model=embedding_model,
            llm_model_name=llm_model_name,
            graph=graph,
            text_chunks=text_chunks,
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            args=args,
            results=results,
            results_path=results_path,
        )
    )
    results["averages"] = compute_averages(results["questions"])
    save_results(results_path, results)

    html_path.write_text(build_html(results), encoding="utf-8")
    print(f"[done] html -> {html_path}", flush=True)
    print(
        json.dumps(
            {
                "message": "三策略问答与评分实验完成",
                "output": str(results_path),
                "html": str(html_path),
                "question_count": len(results["questions"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
