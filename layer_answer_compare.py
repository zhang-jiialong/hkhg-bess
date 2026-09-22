#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立实验脚本：
1. layer_threshold_expand_l0_top20
   - 对 L0-L4 做阈值匹配
   - 每层展开到 L0
   - 对每层展开后的 L0 重新排序，取 top 20
   - 将每层的 L0 证据分层组织后送入问答 prompt

2. direct_l0_top100
   - 直接在 L0 层排序取 top 100
   - 将这些 L0 证据送入同一风格的问答 prompt

3. layer_disjoint_l0_top20
   - 对 L0-L4 做阈值匹配
   - 每层展开到 L0
   - 按 L0 -> L1 -> ... -> Lmax 顺序去重
   - 每层最多取 20，不足则取剩余全部
   - 将各层互不重叠的 L0 证据分层组织后送入问答 prompt

说明：
- 不改原始项目源码
- 不走 DAG/BFS 原推理链
- 仅复用其最后一层问答 prompt 风格
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from openai import OpenAI

from layer_retrieval_compare import (
    collect_by_threshold,
    cosine_scores,
    embed_texts,
    expand_to_l0,
    get_hyperedge_level,
    group_hyperedges_by_level,
    load_graph,
    load_hyperedge_vectors,
    load_json,
    load_questions,
    load_runtime_config,
    make_openai_client,
    rank_l0_hyperedges,
    sample_questions,
)
from HKHG.prompt import GRAPH_FIELD_SEP, PROMPTS
from HKHG.utils import encode_string_by_tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立层级检索问答对比脚本")
    parser.add_argument("--data-source", default="hypertension")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--layer-top-k", type=int, default=20)
    parser.add_argument("--direct-top-k", type=int, default=100)
    parser.add_argument("--path-token-budget", type=int, default=10000)
    parser.add_argument("--entity-token-budget", type=int, default=5000)
    parser.add_argument("--run-ts", default="")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "item"


def build_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    run_ts = args.run_ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        return Path(args.output_dir)
    return repo_root / "results" / f"{args.data_source}_layer_answer_compare" / run_ts


def truncate_lines_by_token_budget(
    lines: list[str],
    token_budget: int,
    model_name: str,
) -> list[str]:
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(encode_string_by_tiktoken(line, model_name=model_name))
        if used + cost > token_budget:
            break
        kept.append(line)
        used += cost
    return kept


def get_l0_entities_and_chunks(
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    l0_hyperedges: list[str],
) -> tuple[list[str], list[str]]:
    entity_lines: list[str] = []
    chunk_texts: list[str] = []
    seen_entities: set[str] = set()
    seen_chunks: set[str] = set()

    for hyperedge_name in l0_hyperedges:
        if hyperedge_name not in graph.nodes:
            continue

        for neighbor in graph.neighbors(hyperedge_name):
            edge_data = graph.get_edge_data(hyperedge_name, neighbor) or {}
            if edge_data.get("role") != "link":
                continue
            node_data = graph.nodes[neighbor]
            if node_data.get("role") != "entity":
                continue
            entity_name = str(neighbor).strip()
            entity_desc = str(node_data.get("description", "")).strip()
            entity_key = f"{entity_name}::{entity_desc}"
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            entity_lines.append(f'({entity_name}<|>{entity_desc})')

        source_id = str(graph.nodes[hyperedge_name].get("source_id", "")).strip()
        if not source_id:
            continue
        chunk_ids = [item.strip() for item in source_id.split(GRAPH_FIELD_SEP) if item.strip()]
        for chunk_id in chunk_ids:
            if chunk_id in seen_chunks:
                continue
            chunk_data = text_chunks.get(chunk_id)
            if not chunk_data:
                continue
            content = str(chunk_data.get("content", "")).strip()
            if not content:
                continue
            seen_chunks.add(chunk_id)
            chunk_texts.append(content)

    return entity_lines, chunk_texts


def build_layer_strategy_context(
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    grouped_hyperedges: dict[int, list[dict[str, Any]]],
    hyperedge_vector_map: dict[str, np.ndarray],
    query_vec: np.ndarray,
    threshold: float,
    max_level: int,
    top_k: int,
) -> dict[str, Any]:
    expand_cache: dict[str, set[str]] = {}
    per_level: dict[str, Any] = {}

    for level in range(max_level + 1):
        level_records = grouped_hyperedges.get(level, [])
        if not level_records:
            per_level[f"L{level}"] = {
                "matched_hyperedges": [],
                "selected_l0_hyperedges": [],
                "selected_l0_scores": [],
            }
            continue

        matrix = np.asarray([item["vector"] for item in level_records], dtype=np.float32)
        scores = cosine_scores(query_vec, matrix)
        matched = collect_by_threshold(level_records, scores, threshold)
        expanded_l0: set[str] = set()
        for item in matched:
            expanded_l0.update(expand_to_l0(graph, item["hyperedge_name"], expand_cache))

        ranked_l0 = rank_l0_hyperedges(
            query_vec=query_vec,
            l0_hyperedges=expanded_l0,
            hyperedge_vector_map=hyperedge_vector_map,
            top_k=top_k,
        )
        per_level[f"L{level}"] = {
            "matched_hyperedges": matched,
            "selected_l0_hyperedges": [item["hyperedge_name"] for item in ranked_l0],
            "selected_l0_scores": ranked_l0,
        }

    all_l0 = []
    for level in range(max_level + 1):
        all_l0.extend(per_level[f"L{level}"]["selected_l0_hyperedges"])
    all_l0 = list(dict.fromkeys(all_l0))

    entity_lines, chunk_texts = get_l0_entities_and_chunks(graph, text_chunks, all_l0)
    return {
        "strategy": "layer_threshold_expand_l0_top20",
        "per_level": per_level,
        "all_selected_l0_hyperedges": all_l0,
        "entity_lines": entity_lines,
        "chunk_texts": chunk_texts,
    }


def build_layer_disjoint_strategy_context(
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    grouped_hyperedges: dict[int, list[dict[str, Any]]],
    hyperedge_vector_map: dict[str, np.ndarray],
    query_vec: np.ndarray,
    threshold: float,
    max_level: int,
    top_k: int,
) -> dict[str, Any]:
    expand_cache: dict[str, set[str]] = {}
    per_level: dict[str, Any] = {}
    consumed_l0: set[str] = set()

    for level in range(max_level + 1):
        level_records = grouped_hyperedges.get(level, [])
        level_label = f"L{level}"
        if not level_records:
            per_level[level_label] = {
                "matched_hyperedges": [],
                "expanded_l0_hyperedges": [],
                "selected_l0_hyperedges": [],
                "selected_l0_scores": [],
            }
            continue

        matrix = np.asarray([item["vector"] for item in level_records], dtype=np.float32)
        scores = cosine_scores(query_vec, matrix)
        matched = collect_by_threshold(level_records, scores, threshold)

        expanded_l0: set[str] = set()
        for item in matched:
            expanded_l0.update(expand_to_l0(graph, item["hyperedge_name"], expand_cache))

        ranked_all = rank_l0_hyperedges(
            query_vec=query_vec,
            l0_hyperedges=expanded_l0,
            hyperedge_vector_map=hyperedge_vector_map,
            top_k=max(len(expanded_l0), top_k),
        )

        selected_ranked: list[dict[str, Any]] = []
        for item in ranked_all:
            hyperedge_name = item["hyperedge_name"]
            if hyperedge_name in consumed_l0:
                continue
            selected_ranked.append(item)
            consumed_l0.add(hyperedge_name)
            if len(selected_ranked) >= top_k:
                break

        per_level[level_label] = {
            "matched_hyperedges": matched,
            "expanded_l0_hyperedges": sorted(expanded_l0),
            "selected_l0_hyperedges": [item["hyperedge_name"] for item in selected_ranked],
            "selected_l0_scores": selected_ranked,
        }

    all_l0 = []
    for level in range(max_level + 1):
        all_l0.extend(per_level[f"L{level}"]["selected_l0_hyperedges"])

    entity_lines, chunk_texts = get_l0_entities_and_chunks(graph, text_chunks, all_l0)
    return {
        "strategy": "layer_disjoint_l0_top20",
        "per_level": per_level,
        "all_selected_l0_hyperedges": all_l0,
        "entity_lines": entity_lines,
        "chunk_texts": chunk_texts,
    }


def build_direct_l0_strategy_context(
    grouped_hyperedges: dict[int, list[dict[str, Any]]],
    hyperedge_vector_map: dict[str, np.ndarray],
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    query_vec: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    level0_records = grouped_hyperedges.get(0, [])
    if not level0_records:
        ranked = []
    else:
        names = [item["hyperedge_name"] for item in level0_records]
        ranked = rank_l0_hyperedges(
            query_vec=query_vec,
            l0_hyperedges=set(names),
            hyperedge_vector_map=hyperedge_vector_map,
            top_k=top_k,
        )

    selected_l0 = [item["hyperedge_name"] for item in ranked]
    entity_lines, chunk_texts = get_l0_entities_and_chunks(graph, text_chunks, selected_l0)
    return {
        "strategy": "direct_l0_top100",
        "selected_l0_hyperedges": selected_l0,
        "selected_l0_scores": ranked,
        "entity_lines": entity_lines,
        "chunk_texts": chunk_texts,
    }


def build_layer_prompt_context(
    per_level: dict[str, Any],
    model_name: str,
    token_budget: int,
) -> str:
    lines: list[str] = []
    for level_label, level_data in per_level.items():
        lines.append(f"[{level_label} Top L0 Hyperedges]")
        for idx, item in enumerate(level_data["selected_l0_scores"], start=1):
            lines.append(f"{idx}. ({item['score']:.4f}) {item['hyperedge_name']}")
        lines.append("")
    kept = truncate_lines_by_token_budget(lines, token_budget, model_name)
    return "\n".join(kept)


def build_direct_prompt_context(
    ranked_l0: list[dict[str, Any]],
    model_name: str,
    token_budget: int,
) -> str:
    lines = ["[Direct L0 Top Hyperedges]"]
    for idx, item in enumerate(ranked_l0, start=1):
        lines.append(f"{idx}. ({item['score']:.4f}) {item['hyperedge_name']}")
    kept = truncate_lines_by_token_budget(lines, token_budget, model_name)
    return "\n".join(kept)


def build_entity_context(
    entity_lines: list[str],
    model_name: str,
    token_budget: int,
) -> str:
    kept = truncate_lines_by_token_budget(entity_lines, token_budget, model_name)
    return PROMPTS["DEFAULT_RECORD_DELIMITER"].join(kept)


def build_prompt(question: str, path_context: str, entity_context: str) -> str:
    prompt_template = PROMPTS["step_answer_generation"]
    return prompt_template.format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        question=question,
        path=path_context,
        entity_descriptions=entity_context,
    )


def ask_llm(
    client: OpenAI,
    model_name: str,
    prompt: str,
) -> dict[str, str]:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    try:
        reasoning = content.split("<reasoning>")[1].split("</reasoning>")[0].strip()
    except Exception:
        reasoning = ""
    try:
        answer = content.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception:
        answer = content.strip()
    return {
        "raw_generation": content,
        "reasoning": reasoning,
        "answer": answer,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

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

    graph_path = workdir / "graph_chunk_entity_relation.graphml"
    vdb_hyperedges_path = workdir / "vdb_hyperedges.json"
    text_chunks_path = workdir / "kv_store_text_chunks.json"

    for path in [graph_path, vdb_hyperedges_path, question_path, config_path, text_chunks_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少输入文件: {path}")

    runtime_config = load_runtime_config(config_path)
    client, embedding_model = make_openai_client(runtime_config)
    llm_model_name = runtime_config.get("llm", {}).get("model", "gpt-4o-mini")

    graph = load_graph(graph_path)
    records = load_hyperedge_vectors(vdb_hyperedges_path)
    grouped_hyperedges = group_hyperedges_by_level(graph, records, args.max_level)
    hyperedge_vector_map = {item["hyperedge_name"]: item["vector"] for item in records}
    text_chunks = load_json(text_chunks_path) or {}

    questions = load_questions(question_path)
    sampled_questions = sample_questions(questions, args.sample_size, args.seed)
    query_vectors = embed_texts(
        client=client,
        model=embedding_model,
        texts=[item["question"] for item in sampled_questions],
    )

    results: dict[str, Any] = {
        "data_source": args.data_source,
        "sample_size": len(sampled_questions),
        "seed": args.seed,
        "threshold": args.threshold,
        "layer_top_k": args.layer_top_k,
        "direct_top_k": args.direct_top_k,
        "embedding_model": embedding_model,
        "llm_model": llm_model_name,
        "workdir": str(workdir),
        "questions": [],
    }

    for item, query_vec in zip(sampled_questions, query_vectors):
        question = item["question"]
        layer_ctx = build_layer_strategy_context(
            graph=graph,
            text_chunks=text_chunks,
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            query_vec=query_vec,
            threshold=args.threshold,
            max_level=args.max_level,
            top_k=args.layer_top_k,
        )
        direct_ctx = build_direct_l0_strategy_context(
            grouped_hyperedges=grouped_hyperedges,
            hyperedge_vector_map=hyperedge_vector_map,
            graph=graph,
            text_chunks=text_chunks,
            query_vec=query_vec,
            top_k=args.direct_top_k,
        )

        layer_path_context = build_layer_prompt_context(
            per_level=layer_ctx["per_level"],
            model_name=llm_model_name,
            token_budget=args.path_token_budget,
        )
        layer_entity_context = build_entity_context(
            entity_lines=layer_ctx["entity_lines"],
            model_name=llm_model_name,
            token_budget=args.entity_token_budget,
        )
        direct_path_context = build_direct_prompt_context(
            ranked_l0=direct_ctx["selected_l0_scores"],
            model_name=llm_model_name,
            token_budget=args.path_token_budget,
        )
        direct_entity_context = build_entity_context(
            entity_lines=direct_ctx["entity_lines"],
            model_name=llm_model_name,
            token_budget=args.entity_token_budget,
        )

        layer_prompt = build_prompt(
            question=question,
            path_context=layer_path_context,
            entity_context=layer_entity_context,
        )
        direct_prompt = build_prompt(
            question=question,
            path_context=direct_path_context,
            entity_context=direct_entity_context,
        )

        layer_answer = ask_llm(client, llm_model_name, layer_prompt)
        direct_answer = ask_llm(client, llm_model_name, direct_prompt)

        question_result = {
            "question_index": item["question_index"],
            "question": question,
            "golden_answers": item["raw"].get("golden_answers", []),
            "layer_threshold_expand_l0_top20": {
                "per_level": layer_ctx["per_level"],
                "all_selected_l0_hyperedges": layer_ctx["all_selected_l0_hyperedges"],
                "entity_line_count": len(layer_ctx["entity_lines"]),
                "chunk_count": len(layer_ctx["chunk_texts"]),
                "prompt": layer_prompt,
                "answer": layer_answer["answer"],
                "reasoning": layer_answer["reasoning"],
                "generation": layer_answer["raw_generation"],
            },
            "direct_l0_top100": {
                "selected_l0_hyperedges": direct_ctx["selected_l0_hyperedges"],
                "selected_l0_scores": direct_ctx["selected_l0_scores"],
                "entity_line_count": len(direct_ctx["entity_lines"]),
                "chunk_count": len(direct_ctx["chunk_texts"]),
                "prompt": direct_prompt,
                "answer": direct_answer["answer"],
                "reasoning": direct_answer["reasoning"],
                "generation": direct_answer["raw_generation"],
            },
        }
        results["questions"].append(question_result)

    output_path = output_dir / "layer_answer_compare_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "message": "独立检索问答实验完成",
                "output": str(output_path),
                "question_count": len(results["questions"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
