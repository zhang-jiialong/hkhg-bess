from __future__ import annotations

from typing import Any

from hh_iterative_l2_path_beam_rerank import (
    BaseReranker,
    RetrievalPath,
    append_path,
    build_context,
    build_evidence_text,
    cross_layer_path_step,
    dedupe_paths,
    expand_l2_paths,
    expand_l0_paths,
    final_evidence_rerank,
    make_anchor_path,
    summarize_path_counts,
    retrieve_l2_path_beam_visited_aware_rerank,
)
from hh_iterative_retrieval import MultiOrderIterativeRetriever, jaccard_similarity


def _collect_l0_descendants(retriever: MultiOrderIterativeRetriever, hyperedge_id: str) -> set[str]:
    if retriever.hyperedges.get(hyperedge_id) and retriever.hyperedges[hyperedge_id].level == 0:
        return {hyperedge_id}
    out: set[str] = set()
    stack = [hyperedge_id]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        current_level = retriever.hyperedges.get(current).level if retriever.hyperedges.get(current) else None
        if current_level == 0:
            out.add(current)
            continue
        if current_level is None:
            continue
        for next_level in range(current_level - 1, -1, -1):
            children = retriever.children_by_level.get((current, next_level), set())
            if children:
                stack.extend(sorted(children))
                break
    return out


def _semantic_final(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    candidates: set[str],
    top_k_final: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = retriever.semantic_scores_for_level(query, 0)
    ranked: list[dict[str, Any]] = []
    for hid in sorted(candidates):
        hyperedge = retriever.hyperedges.get(hid)
        if not hyperedge or hyperedge.level != 0 or hid not in retriever.hyperedge_vector_map:
            continue
        ranked.append(
            {
                "hyperedge_id": hid,
                "level": 0,
                "final_score": float(scores.get(hid, 0.0)),
                "text": build_evidence_text(retriever, hid),
                "entities": sorted(hyperedge.entities),
                "source_text": retriever.source_text(hid),
            }
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[:top_k_final], {
        "stage": "semantic_final_rerank",
        "candidate_count": len(ranked),
        "kept_count": min(top_k_final, len(ranked)),
        "reranker": "semantic_similarity",
    }


def retrieve_layer_wise_topk(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    top_k_per_layer: int = 10,
    top_k_final: int = 20,
    max_layer: int = 2,
) -> dict[str, Any]:
    l0_candidates: set[str] = set()
    layer_logs: list[dict[str, Any]] = []
    for level in range(max_layer + 1):
        order = retriever.semantic_order_for_level(query, level)[:top_k_per_layer]
        level_l0: set[str] = set()
        for hid in order:
            level_l0.update(_collect_l0_descendants(retriever, hid))
        l0_candidates.update(level_l0)
        layer_logs.append(
            {
                "level": level,
                "selected_hyperedge_count": len(order),
                "descendant_l0_count": len(level_l0),
                "selected_hyperedges": order,
            }
        )

    top_l0, final_info = _semantic_final(retriever, query, l0_candidates, top_k_final)
    return {
        "query": query,
        "strategy": "Layer-wise Top-K Retrieval",
        "settings": {
            "retrieval_mode": "layer_wise_topk",
            "top_k_per_layer": top_k_per_layer,
            "top_k_final": top_k_final,
            "max_layer": max_layer,
            "reranker": "none",
        },
        "layer_logs": layer_logs,
        "final_rerank": final_info,
        "top_l0_hyperedges": top_l0,
        "selected_hyperedges": top_l0,
        "visited_l0_count": len(l0_candidates),
        "visited_count": sum(item["selected_hyperedge_count"] for item in layer_logs),
        "context": build_context(top_l0).replace(
            "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
            "[Layer-wise Top-K Retrieval: Final L0 Evidence]",
        ),
    }


def retrieve_layer_wise_topk_final_rerank(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    reranker: BaseReranker,
    top_k_per_layer: int = 10,
    top_k_final: int = 20,
    max_layer: int = 2,
) -> dict[str, Any]:
    l0_candidates: set[str] = set()
    layer_logs: list[dict[str, Any]] = []
    for level in range(max_layer + 1):
        order = retriever.semantic_order_for_level(query, level)[:top_k_per_layer]
        level_l0: set[str] = set()
        for hid in order:
            level_l0.update(_collect_l0_descendants(retriever, hid))
        l0_candidates.update(level_l0)
        layer_logs.append(
            {
                "level": level,
                "selected_hyperedge_count": len(order),
                "descendant_l0_count": len(level_l0),
                "selected_hyperedges": order,
                "initial_selection": "semantic_topk",
            }
        )

    top_l0, final_info = final_evidence_rerank(retriever, reranker, query, l0_candidates, top_k_final)
    context = build_context(top_l0).replace(
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        "[Layer-wise Top-K with Final Rerank: Final L0 Evidence]",
    )
    return {
        "query": query,
        "strategy": "Layer-wise Top-K with Final Rerank",
        "settings": {
            "retrieval_mode": "layer_wise_topk_final_rerank",
            "top_k_per_layer": top_k_per_layer,
            "top_k_final": top_k_final,
            "max_layer": max_layer,
            "initial_layer_selection": "semantic_topk",
            "final_selection": "reranker",
            "reranker": reranker.name,
        },
        "layer_logs": layer_logs,
        "final_rerank": final_info,
        "top_l0_hyperedges": top_l0,
        "selected_hyperedges": top_l0,
        "visited_l0_count": len(l0_candidates),
        "visited_count": sum(item["selected_hyperedge_count"] for item in layer_logs),
        "context": context,
    }


def expand_l0_paths_no_parent_overlap(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    l0_paths: list[RetrievalPath],
    *,
    visited: set[str],
    visited_l0: set[str],
    top_k_expand: int,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    by_end: dict[str, list[RetrievalPath]] = {}
    for path in l0_paths:
        if path.current_layer == 0:
            by_end.setdefault(path.last_edge, []).append(path)
    semantic_scores = retriever.semantic_scores_for_level(query, 0)
    semantic_fallback = retriever.semantic_order_for_level(query, 0)[: max(300, top_k_expand * 20)]
    expanded_paths: list[RetrievalPath] = []
    logs: list[dict[str, Any]] = []
    for source_id, paths in sorted(by_end.items()):
        source_entities = set(retriever.hyperedges[source_id].entities)
        candidate_pool: set[str] = set(semantic_fallback)
        for entity in source_entities:
            candidate_pool.update(retriever.hyperedges_by_entity_and_level.get((entity, 0), set()))
        scored: list[dict[str, Any]] = []
        for candidate_id in candidate_pool:
            if candidate_id not in retriever.hyperedge_vector_map:
                continue
            if candidate_id == source_id or candidate_id in visited:
                continue
            semantic = semantic_scores.get(candidate_id, 0.0)
            shared_entity_score = jaccard_similarity(source_entities, set(retriever.hyperedges[candidate_id].entities))
            score = (semantic + shared_entity_score) / 2.0
            scored.append(
                {
                    "source_hyperedge_id": source_id,
                    "candidate_hyperedge_id": candidate_id,
                    "level": 0,
                    "score": float(score),
                    "semantic_score": float(semantic),
                    "shared_entity_score": float(shared_entity_score),
                    "scoring": "without_parent_overlap",
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        picked = scored[:top_k_expand]
        logs.extend(picked)
        for item in picked:
            for path in paths:
                expanded_paths.append(append_path(path, item["candidate_hyperedge_id"], 0, "L0_expansion_no_parent_overlap"))
    expanded_paths = dedupe_paths(expanded_paths)
    expanded_ids = {path.last_edge for path in expanded_paths}
    visited.update(expanded_ids)
    visited_l0.update(expanded_ids)
    return expanded_paths, {
        "stage": "L0_same_layer_expansion_no_parent_overlap",
        "input_path_count": len(l0_paths),
        "expanded_path_count": len(expanded_paths),
        "expanded_l0_count": len(expanded_ids),
        "expansion_log_count": len(logs),
    }


def retrieve_path_beam_depth_ablation(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    reranker: BaseReranker,
    retrieval_depth: int,
    top_k_anchor: int = 2,
    expansion_top_k: int = 2,
    path_beam_size: int = 8,
    top_k_final: int = 20,
    max_iter: int = 3,
    candidate_cap: int | None = None,
) -> dict[str, Any]:
    if retrieval_depth < 0:
        raise ValueError("retrieval_depth must be >= 0")
    if retrieval_depth > retriever.max_level:
        raise ValueError(f"retrieval_depth={retrieval_depth} exceeds retriever.max_level={retriever.max_level}")

    anchors = retriever.initial_graph_anchoring(query, top_k_anchor=top_k_anchor)
    path_beam = [make_anchor_path(retriever, item) for item in anchors]
    visited: set[str] = {path.last_edge for path in path_beam}
    visited_l0: set[str] = set(visited)
    iteration_logs: list[dict[str, Any]] = []

    for iteration in range(max_iter):
        if not path_beam:
            break
        log: dict[str, Any] = {
            "iteration": iteration,
            "input_path_count": len(path_beam),
            "input_end_l0_count": len({path.last_edge for path in path_beam}),
            "stages": [],
        }

        if retrieval_depth == 0:
            next_paths, info = expand_l0_paths_no_parent_overlap(
                retriever,
                query,
                path_beam,
                visited=visited,
                visited_l0=visited_l0,
                top_k_expand=expansion_top_k,
            )
            log["stages"].append(info)
            log["output_path_count"] = len(next_paths)
            log["visited_count"] = len(visited)
            log["visited_l0_count"] = len(visited_l0)
            iteration_logs.append(log)
            path_beam = next_paths
            continue

        current_paths = path_beam
        for level in range(1, retrieval_depth + 1):
            current_paths, info = cross_layer_path_step(
                retriever,
                reranker,
                query,
                current_paths,
                from_level=level - 1,
                to_level=level,
                direction="up",
                visited=visited,
                path_beam_size=path_beam_size,
                candidate_cap=candidate_cap,
            )
            log["stages"].append(info)
            if not current_paths:
                break
        if not current_paths:
            iteration_logs.append(log)
            break

        top_entry_paths, info = expand_l2_paths(
            retriever,
            query,
            current_paths,
            visited=visited,
            top_k_expand=expansion_top_k,
        )
        info["stage"] = f"L{retrieval_depth}_same_layer_expansion"
        log["stages"].append(info)

        current_paths = top_entry_paths
        for level in range(retrieval_depth, 0, -1):
            current_paths, info = cross_layer_path_step(
                retriever,
                reranker,
                query,
                current_paths,
                from_level=level,
                to_level=level - 1,
                direction="down",
                visited=visited,
                path_beam_size=path_beam_size,
                candidate_cap=candidate_cap,
            )
            if level - 1 == 0:
                reached_l0 = {path.last_edge for path in current_paths if path.current_layer == 0}
                visited_l0.update(reached_l0)
                info["reached_l0_count"] = len(reached_l0)
            log["stages"].append(info)
            if not current_paths:
                break
        if not current_paths:
            iteration_logs.append(log)
            break

        next_paths, info = expand_l0_paths(
            retriever,
            query,
            current_paths,
            visited=visited,
            visited_l0=visited_l0,
            top_k_expand=expansion_top_k,
        )
        log["stages"].append(info)
        log["output_path_count"] = len(next_paths)
        log["visited_count"] = len(visited)
        log["visited_l0_count"] = len(visited_l0)
        iteration_logs.append(log)
        path_beam = next_paths
        if not path_beam:
            break

    top_l0, final_info = final_evidence_rerank(retriever, reranker, query, visited_l0, top_k_final)
    return {
        "query": query,
        "strategy": f"Path-Beam Depth Ablation L={retrieval_depth}",
        "settings": {
            "retrieval_mode": "path_beam_depth_ablation",
            "retrieval_depth": retrieval_depth,
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "candidate_cap": candidate_cap,
            "reranker": reranker.name,
        },
        "anchors": anchors,
        "iterations": iteration_logs,
        "final_rerank": final_info,
        "top_l0_hyperedges": top_l0,
        "selected_hyperedges": top_l0,
        "visited_l0_count": len(visited_l0),
        "visited_count": len(visited),
        "context": build_context(top_l0).replace(
            "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
            f"[Path-Beam Depth Ablation L={retrieval_depth}: Final L0 Evidence]",
        ),
    }


def retrieve_parallel_path_layer_fusion(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    reranker: BaseReranker,
    top_k_anchor: int = 2,
    expansion_top_k: int = 2,
    path_beam_size: int = 8,
    top_k_final: int = 20,
    max_iter: int = 3,
    candidate_cap: int | None = None,
    top_k_per_layer: int = 10,
    layer_wise_max_layer: int = 2,
) -> dict[str, Any]:
    path_result = retrieve_l2_path_beam_visited_aware_rerank(
        retriever=retriever,
        query=query,
        reranker=reranker,
        top_k_anchor=top_k_anchor,
        expansion_top_k=expansion_top_k,
        path_beam_size=path_beam_size,
        top_k_final=top_k_final,
        max_iter=max_iter,
        candidate_cap=candidate_cap,
    )
    layer_result = retrieve_layer_wise_topk(
        retriever=retriever,
        query=query,
        top_k_per_layer=top_k_per_layer,
        top_k_final=top_k_final,
        max_layer=layer_wise_max_layer,
    )

    path_items = path_result.get("top_l0_hyperedges", [])
    layer_items = layer_result.get("top_l0_hyperedges", [])
    path_ids = [str(item.get("hyperedge_id")) for item in path_items if item.get("hyperedge_id")]
    layer_ids = [str(item.get("hyperedge_id")) for item in layer_items if item.get("hyperedge_id")]
    path_id_set = set(path_ids)
    layer_id_set = set(layer_ids)

    fused: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in path_items:
        hid = str(item.get("hyperedge_id", ""))
        if not hid or hid in seen:
            continue
        merged = dict(item)
        merged["fusion_source"] = "path"
        fused.append(merged)
        seen.add(hid)
    for item in layer_items:
        hid = str(item.get("hyperedge_id", ""))
        if not hid or hid in seen:
            continue
        merged = dict(item)
        merged["fusion_source"] = "layer"
        fused.append(merged)
        seen.add(hid)

    overlap = path_id_set & layer_id_set
    context = build_context(fused).replace(
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        "[Parallel Path-Layer Fusion: Final L0 Evidence]",
    )
    return {
        "query": query,
        "strategy": "Parallel Path-Layer Fusion",
        "settings": {
            "retrieval_mode": "parallel_path_layer_fusion",
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "candidate_cap": candidate_cap,
            "top_k_per_layer": top_k_per_layer,
            "layer_wise_max_layer": layer_wise_max_layer,
            "reranker": reranker.name,
        },
        "path_result": path_result,
        "layer_result": layer_result,
        "fusion_log": {
            "path_count": len(path_ids),
            "layer_count": len(layer_ids),
            "fusion_count": len(fused),
            "overlap_count": len(overlap),
            "final_evidence_count": len(fused),
            "path_ids": path_ids,
            "layer_ids": layer_ids,
            "overlap_ids": sorted(overlap),
        },
        "final_rerank": {
            "stage": "parallel_path_layer_fusion_no_rerank",
            "candidate_count": len(fused),
            "kept_count": len(fused),
            "reranker": "none_after_fusion",
        },
        "top_l0_hyperedges": fused,
        "selected_hyperedges": fused,
        "visited_l0_count": len(fused),
        "visited_count": len(fused),
        "context": context,
    }


def retrieve_parallel_path_layer_fusion_final_rerank(
    *,
    retriever: MultiOrderIterativeRetriever,
    query: str,
    reranker: BaseReranker,
    top_k_anchor: int = 2,
    expansion_top_k: int = 2,
    path_beam_size: int = 8,
    top_k_final: int = 20,
    max_iter: int = 3,
    candidate_cap: int | None = None,
    top_k_per_layer: int = 10,
    layer_wise_max_layer: int = 2,
) -> dict[str, Any]:
    """Parallel path-layer fusion followed by final evidence reranking."""
    path_result = retrieve_l2_path_beam_visited_aware_rerank(
        retriever=retriever,
        query=query,
        reranker=reranker,
        top_k_anchor=top_k_anchor,
        expansion_top_k=expansion_top_k,
        path_beam_size=path_beam_size,
        top_k_final=top_k_final,
        max_iter=max_iter,
        candidate_cap=candidate_cap,
    )
    layer_result = retrieve_layer_wise_topk(
        retriever=retriever,
        query=query,
        top_k_per_layer=top_k_per_layer,
        top_k_final=top_k_final,
        max_layer=layer_wise_max_layer,
    )

    path_items = path_result.get("top_l0_hyperedges", [])
    layer_items = layer_result.get("top_l0_hyperedges", [])
    path_ids = [str(item.get("hyperedge_id")) for item in path_items if item.get("hyperedge_id")]
    layer_ids = [str(item.get("hyperedge_id")) for item in layer_items if item.get("hyperedge_id")]
    path_id_set = set(path_ids)
    layer_id_set = set(layer_ids)
    fused_ids = list(dict.fromkeys(path_ids + layer_ids))

    top_l0, final_info = final_evidence_rerank(
        retriever,
        reranker,
        query,
        set(fused_ids),
        top_k_final,
    )
    context = build_context(top_l0).replace(
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        "[Parallel Path-Layer Fusion + Final Rerank: Final L0 Evidence]",
    )
    overlap = path_id_set & layer_id_set
    final_info = dict(final_info)
    final_info["stage"] = "parallel_path_layer_fusion_final_rerank"
    final_info["fusion_candidate_count"] = len(fused_ids)
    return {
        "query": query,
        "strategy": "Parallel Path-Layer Fusion + Final Rerank",
        "settings": {
            "retrieval_mode": "parallel_path_layer_fusion_final_rerank",
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "candidate_cap": candidate_cap,
            "top_k_per_layer": top_k_per_layer,
            "layer_wise_max_layer": layer_wise_max_layer,
            "reranker": reranker.name,
        },
        "path_result": path_result,
        "layer_result": layer_result,
        "fusion_log": {
            "path_count": len(path_ids),
            "layer_count": len(layer_ids),
            "fusion_count": len(fused_ids),
            "overlap_count": len(overlap),
            "final_evidence_count": len(top_l0),
            "path_ids": path_ids,
            "layer_ids": layer_ids,
            "overlap_ids": sorted(overlap),
        },
        "final_rerank": final_info,
        "top_l0_hyperedges": top_l0,
        "selected_hyperedges": top_l0,
        "visited_l0_count": len(fused_ids),
        "visited_count": len(fused_ids),
        "context": context,
    }



__all__ = [
    "retrieve_layer_wise_topk",
    "retrieve_layer_wise_topk_final_rerank",
    "retrieve_path_beam_depth_ablation",
    "retrieve_parallel_path_layer_fusion",
    "retrieve_parallel_path_layer_fusion_final_rerank",
    "summarize_path_counts",
]
