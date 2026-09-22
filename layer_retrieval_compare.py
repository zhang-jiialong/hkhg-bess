#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分层超边检索对比脚本

用途：
1. 从指定问题集里默认随机抽取 3 个问题。
2. 将每个问题分别与 L0-L4 各层超边做余弦相似度匹配。
3. 若命中的是高阶超边，则沿 contains_hyperedge 递归展开到最低阶 L0 超边。
4. 对比每一层最终落到的 L0 超边集合差异。
5. 输出结构化 JSON，并生成热力图 / 流向图，便于直观分析。

注意：
- 该脚本是独立脚本，不会修改项目现有代码。
- 待测试的 workdir 必须与当前 embedding API 配置一致，否则向量维度无法对齐。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较不同层级超边检索到底层 L0 证据的差异")
    parser.add_argument(
        "--data-source",
        default="example",
        help="数据源名称，用于自动推断 workdir 和 question 文件路径，默认 example",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="已完成构图的工作目录，默认 data/work/<data_source>",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="问题文件路径，默认 questions/<data_source>/questions.json",
    )
    parser.add_argument(
        "--config",
        default="runtime_config.json",
        help="运行配置文件，默认 runtime_config.json",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="默认随机抽取的问题数，默认 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--mode",
        choices=["threshold", "aligned_topk", "both"],
        default="both",
        help="检索模式：threshold / aligned_topk / both，默认 both",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="余弦相似度阈值，默认 0.45",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="兼容保留参数；aligned_topk 模式实际使用 L0 阈值命中数作为统一 K",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=4,
        help="最大层级，默认 4",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认 <workdir>/layer_compare",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "item"


def load_runtime_config(config_path: Path) -> dict[str, Any]:
    return load_json(config_path)


def make_openai_client(config: dict[str, Any]) -> tuple[OpenAI, str]:
    llm_cfg = config["llm"]
    emb_cfg = config["embedding"]
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
    )
    embedding_model = emb_cfg["model_path"]
    return client, embedding_model


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return np.asarray(vectors, dtype=np.float32)


def load_questions(question_path: Path) -> list[dict[str, Any]]:
    data = load_json(question_path)
    if not isinstance(data, list):
        raise ValueError(f"问题文件格式错误：{question_path}")
    return data


def sample_questions(
    questions: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    indexed = []
    for idx, item in enumerate(questions):
        indexed.append(
            {
                "question_index": idx,
                "question": item.get("question", ""),
                "raw": item,
            }
        )
    if sample_size >= len(indexed):
        return indexed
    rng = random.Random(seed)
    return rng.sample(indexed, sample_size)


def load_graph(graph_path: Path) -> nx.Graph:
    return nx.read_graphml(graph_path)


def get_hyperedge_level(graph: nx.Graph, hyperedge_name: str) -> int:
    if hyperedge_name not in graph.nodes:
        raise KeyError(f"图中不存在超边节点：{hyperedge_name}")
    level_raw = graph.nodes[hyperedge_name].get("hyperedge_level", 0)
    try:
        return int(level_raw)
    except (TypeError, ValueError):
        return 0


def load_hyperedge_vectors(vdb_path: Path) -> list[dict[str, Any]]:
    raw = load_json(vdb_path)
    embedding_dim = int(raw["embedding_dim"])
    db = NanoVectorDB(embedding_dim=embedding_dim, storage_file=str(vdb_path))
    storage = getattr(db, "_NanoVectorDB__storage")
    matrix = np.asarray(storage["matrix"], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(storage["data"]):
        records.append(
            {
                "id": item["__id__"],
                "hyperedge_name": item["hyperedge_name"],
                "vector": matrix[idx],
            }
        )
    return records


def group_hyperedges_by_level(
    graph: nx.Graph,
    records: list[dict[str, Any]],
    max_level: int,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        name = record["hyperedge_name"]
        if name not in graph.nodes:
            continue
        node_data = graph.nodes[name]
        if node_data.get("role") != "hyperedge":
            continue
        level = get_hyperedge_level(graph, name)
        if level > max_level:
            continue
        grouped[level].append(record)
    return dict(sorted(grouped.items(), key=lambda x: x[0]))


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec)
    m_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(q_norm * m_norm, 1e-12)
    return (matrix @ query_vec) / denom


def collect_by_threshold(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record, score in zip(records, scores):
        if float(score) >= threshold:
            matched.append(
                {
                    "id": record["id"],
                    "hyperedge_name": record["hyperedge_name"],
                    "score": float(score),
                }
            )
    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


def rank_l0_hyperedges(
    query_vec: np.ndarray,
    l0_hyperedges: set[str],
    hyperedge_vector_map: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    if not l0_hyperedges or top_k <= 0:
        return []
    names = [name for name in sorted(l0_hyperedges) if name in hyperedge_vector_map]
    if not names:
        return []
    matrix = np.asarray([hyperedge_vector_map[name] for name in names], dtype=np.float32)
    scores = cosine_scores(query_vec, matrix)
    order = np.argsort(scores)[::-1][:top_k]
    ranked = []
    for idx in order:
        name = names[int(idx)]
        ranked.append(
            {
                "hyperedge_name": name,
                "score": float(scores[int(idx)]),
            }
        )
    return ranked


def expand_to_l0(
    graph: nx.Graph,
    hyperedge_name: str,
    cache: dict[str, set[str]],
) -> set[str]:
    if hyperedge_name in cache:
        return cache[hyperedge_name]

    current_level = get_hyperedge_level(graph, hyperedge_name)
    if current_level == 0:
        cache[hyperedge_name] = {hyperedge_name}
        return cache[hyperedge_name]

    result: set[str] = set()
    for neighbor in graph.neighbors(hyperedge_name):
        edge_data = graph.get_edge_data(hyperedge_name, neighbor) or {}
        if edge_data.get("role") != "contains_hyperedge":
            continue
        if graph.nodes[neighbor].get("role") != "hyperedge":
            continue
        neighbor_level = get_hyperedge_level(graph, neighbor)
        if neighbor_level >= current_level:
            continue
        result.update(expand_to_l0(graph, neighbor, cache))

    cache[hyperedge_name] = result
    return result


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_threshold_result(
    graph: nx.Graph,
    grouped_hyperedges: dict[int, list[dict[str, Any]]],
    query_vec: np.ndarray,
    threshold: float,
    max_level: int,
    expand_cache: dict[str, set[str]],
) -> dict[str, Any]:
    per_level: dict[str, Any] = {}
    for level in range(max_level + 1):
        level_records = grouped_hyperedges.get(level, [])
        if not level_records:
            per_level[f"L{level}"] = {
                "matched_hyperedges": [],
                "expanded_l0_hyperedges": [],
                "expanded_l0_count": 0,
            }
            continue

        matrix = np.asarray([item["vector"] for item in level_records], dtype=np.float32)
        scores = cosine_scores(query_vec, matrix)

        matched = collect_by_threshold(level_records, scores, threshold)

        expanded_l0: set[str] = set()
        for item in matched:
            expanded_l0.update(expand_to_l0(graph, item["hyperedge_name"], expand_cache))

        per_level[f"L{level}"] = {
            "matched_hyperedges": matched,
            "expanded_l0_hyperedges": sorted(expanded_l0),
            "expanded_l0_count": len(expanded_l0),
        }

    labels = [f"L{level}" for level in range(max_level + 1)]
    matrix = []
    for left in labels:
        row = []
        left_set = set(per_level[left]["expanded_l0_hyperedges"])
        for right in labels:
            right_set = set(per_level[right]["expanded_l0_hyperedges"])
            row.append(round(jaccard(left_set, right_set), 6))
        matrix.append(row)

    return {
        "mode": "threshold",
        "per_level": per_level,
        "jaccard_matrix": {
            "labels": labels,
            "values": matrix,
        },
    }


def build_aligned_topk_result(
    query_vec: np.ndarray,
    threshold_result: dict[str, Any],
    hyperedge_vector_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    labels = threshold_result["jaccard_matrix"]["labels"]
    per_level: dict[str, Any] = {}

    aligned_k = 20

    for label in labels:
        threshold_level = threshold_result["per_level"][label]
        expanded_l0 = set(threshold_level["expanded_l0_hyperedges"])
        ranked_l0 = rank_l0_hyperedges(
            query_vec=query_vec,
            l0_hyperedges=expanded_l0,
            hyperedge_vector_map=hyperedge_vector_map,
            top_k=aligned_k,
        )
        per_level[label] = {
            "matched_hyperedges": threshold_level["matched_hyperedges"],
            "expanded_l0_hyperedges": [item["hyperedge_name"] for item in ranked_l0],
            "expanded_l0_scores": ranked_l0,
            "expanded_l0_count": len(ranked_l0),
            "source_threshold_l0_count": threshold_level["expanded_l0_count"],
        }

    matrix = []
    for left in labels:
        row = []
        left_set = set(per_level[left]["expanded_l0_hyperedges"])
        for right in labels:
            right_set = set(per_level[right]["expanded_l0_hyperedges"])
            row.append(round(jaccard(left_set, right_set), 6))
        matrix.append(row)

    return {
        "mode": "aligned_topk",
        "aligned_k": aligned_k,
        "per_level": per_level,
        "jaccard_matrix": {
            "labels": labels,
            "values": matrix,
        },
    }


def draw_heatmap(
    question_text: str,
    mode_result: dict[str, Any],
    output_path: Path,
) -> None:
    labels = mode_result["jaccard_matrix"]["labels"]
    values = np.asarray(mode_result["jaccard_matrix"]["values"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(f"Jaccard Overlap Heatmap\n{question_text[:100]}")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_flow_graph(
    question_text: str,
    mode_result: dict[str, Any],
    output_path: Path,
    max_display_hyperedges: int = 3,
    max_display_l0: int = 10,
) -> None:
    graph = nx.DiGraph()
    q_node = "QUESTION"
    graph.add_node(q_node, label=question_text[:90], group="question")

    level_labels = mode_result["jaccard_matrix"]["labels"]
    union_l0: list[str] = []
    for level_label in level_labels:
        union_l0.extend(mode_result["per_level"][level_label]["expanded_l0_hyperedges"])
    union_l0 = sorted(dict.fromkeys(union_l0))[:max_display_l0]

    for level_idx, level_label in enumerate(level_labels):
        level_node = f"{level_label}_summary"
        level_data = mode_result["per_level"][level_label]
        graph.add_node(
            level_node,
            label=f"{level_label}\nmatch={len(level_data['matched_hyperedges'])}\nL0={level_data['expanded_l0_count']}",
            group="level",
            layer=level_idx,
        )
        graph.add_edge(q_node, level_node)

        for match_idx, match in enumerate(level_data["matched_hyperedges"][:max_display_hyperedges]):
            hit_node = f"{level_label}_hit_{match_idx}"
            hit_label = f"{level_label} hit\n{match['score']:.3f}\n{match['hyperedge_name'][:55]}"
            graph.add_node(hit_node, label=hit_label, group="hit", layer=level_idx)
            graph.add_edge(level_node, hit_node)

            expanded = level_data["expanded_l0_hyperedges"]
            for l0_name in expanded:
                if l0_name not in union_l0:
                    continue
                l0_node = f"L0::{l0_name}"
                if l0_node not in graph:
                    graph.add_node(l0_node, label=l0_name[:65], group="l0")
                graph.add_edge(hit_node, l0_node)

    pos: dict[str, tuple[float, float]] = {}
    pos[q_node] = (0.0, 0.0)

    for idx, level_label in enumerate(level_labels):
        summary_node = f"{level_label}_summary"
        pos[summary_node] = (1.0, -idx * 2.0)

        hit_nodes = [node for node in graph.nodes if node.startswith(f"{level_label}_hit_")]
        for hit_idx, hit_node in enumerate(hit_nodes):
            pos[hit_node] = (2.2, -idx * 2.0 - hit_idx * 0.55)

    l0_nodes = [node for node, data in graph.nodes(data=True) if data.get("group") == "l0"]
    for idx, l0_node in enumerate(l0_nodes):
        pos[l0_node] = (4.1, -idx * 0.8)

    fig, ax = plt.subplots(figsize=(16, max(7, len(l0_nodes) * 0.55)))
    colors = []
    for _, data in graph.nodes(data=True):
        group = data.get("group")
        if group == "question":
            colors.append("#f28e2b")
        elif group == "level":
            colors.append("#4e79a7")
        elif group == "hit":
            colors.append("#76b7b2")
        else:
            colors.append("#e15759")

    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1700, ax=ax)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", width=1.2, alpha=0.65, ax=ax)
    labels = {node: data.get("label", node) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(f"Layer-to-L0 Flow\n{question_text[:100]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_results(result: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else workdir / "layer_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = workdir / "graph_chunk_entity_relation.graphml"
    vdb_hyperedges_path = workdir / "vdb_hyperedges.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"图文件不存在：{graph_path}")
    if not vdb_hyperedges_path.exists():
        raise FileNotFoundError(f"超边向量库不存在：{vdb_hyperedges_path}")
    if not question_path.exists():
        raise FileNotFoundError(f"问题文件不存在：{question_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    graph = load_graph(graph_path)
    records = load_hyperedge_vectors(vdb_hyperedges_path)
    grouped_hyperedges = group_hyperedges_by_level(graph, records, args.max_level)
    hyperedge_vector_map = {item["hyperedge_name"]: item["vector"] for item in records}

    questions = load_questions(question_path)
    sampled_questions = sample_questions(questions, args.sample_size, args.seed)

    runtime_config = load_runtime_config(config_path)
    client, embedding_model = make_openai_client(runtime_config)
    question_vectors = embed_texts(
        client=client,
        model=embedding_model,
        texts=[item["question"] for item in sampled_questions],
    )

    result: dict[str, Any] = {
        "data_source": args.data_source,
        "workdir": str(workdir),
        "question_file": str(question_path),
        "config_file": str(config_path),
        "sample_size": len(sampled_questions),
        "seed": args.seed,
        "mode": args.mode,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "max_level": args.max_level,
        "embedding_model": embedding_model,
        "questions": [],
    }

    for item, query_vec in zip(sampled_questions, question_vectors):
        expand_cache: dict[str, set[str]] = {}
        question_entry: dict[str, Any] = {
            "question_index": item["question_index"],
            "question": item["question"],
            "modes": {},
        }

        question_slug = slugify(f"q{item['question_index']}_{item['question'][:50]}")

        threshold_result = build_threshold_result(
            graph=graph,
            grouped_hyperedges=grouped_hyperedges,
            query_vec=query_vec,
            threshold=args.threshold,
            max_level=args.max_level,
            expand_cache=expand_cache,
        )

        if args.mode in {"threshold", "both"}:
            question_entry["modes"]["threshold"] = threshold_result
            heatmap_path = output_dir / f"{question_slug}_threshold_heatmap.png"
            flow_path = output_dir / f"{question_slug}_threshold_flow.png"
            draw_heatmap(item["question"], threshold_result, heatmap_path)
            draw_flow_graph(item["question"], threshold_result, flow_path)

        if args.mode in {"aligned_topk", "both"}:
            aligned_topk_result = build_aligned_topk_result(
                query_vec=query_vec,
                threshold_result=threshold_result,
                hyperedge_vector_map=hyperedge_vector_map,
            )
            question_entry["modes"]["aligned_topk"] = aligned_topk_result
            heatmap_path = output_dir / f"{question_slug}_aligned_topk_heatmap.png"
            flow_path = output_dir / f"{question_slug}_aligned_topk_flow.png"
            draw_heatmap(item["question"], aligned_topk_result, heatmap_path)
            draw_flow_graph(item["question"], aligned_topk_result, flow_path)

        result["questions"].append(question_entry)

    save_results(result, output_dir / "layer_compare_results.json")

    summary = {
        "message": "对比完成",
        "output_dir": str(output_dir),
        "question_count": len(result["questions"]),
        "modes": args.mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
