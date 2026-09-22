from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np

from hh_iterative_l2_path_beam_rerank import (
    BaseReranker,
    RetrievalPath,
    append_path,
    build_path_text,
    build_context,
    cross_layer_path_step,
    dedupe_paths,
    final_evidence_rerank,
    inject_l0_semantic_anchor_paths,
    make_anchor_path,
    summarize_path_counts,
)
from hh_iterative_retrieval import MultiOrderIterativeRetriever, cosine_similarity
from generate_v2_semantic_weak_structure_mechanism_chain_aware_compact_synthesis_prompt_20260603 import (
    cosine_counter,
    mechanism_chain_aware_select,
    words,
)


EXPANSION_MODES = {
    "semantic_only",
    "semantic_weak_structure",
    "semantic_weak_structure_redundancy",
}


MECHANISM_EXTRACTION_PROMPT = """You extract mechanistic relations from retrieved evidence for answering one battery-management question.

For each evidence item, decide whether the item itself states a concrete mechanistic or causal relation that is directly useful for answering the specific question.
Do not use any predefined vocabulary. Do not infer mechanisms from general domain knowledge.
If an item is only a definition, component description, interface description, parameter value, or isolated fact with no causal/mechanistic relation, return an empty mechanisms array.
If an item contains a mechanism but that mechanism is not directly useful for the question, return an empty mechanisms array.

A mechanism relation may include condition, cause, process/change, and effect/impact. It is OK if only part of this chain is explicitly stated, but every returned field must be grounded in the evidence text.

Return strict JSON only:
{
  "items": [
    {
      "index": 0,
      "mechanisms": [
        {
          "condition": "...",
          "cause": "...",
          "process": "...",
          "effect": "..."
        }
      ]
    }
  ]
}

Use at most 2 mechanisms per item. Use empty strings for missing fields. Do not include explanations."""


def _json_from_text(text: str) -> dict[str, Any]:
    text = str(text or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _mechanism_evidence_text(item: dict[str, Any], limit: int = 900) -> str:
    text = str(item.get('text') or item.get('fact_core') or '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:limit]


def extract_dynamic_mechanisms_for_candidates(
    *,
    client: Any,
    model: str,
    question: str,
    candidates: list[dict[str, Any]],
    batch_size: int = 20,
    timeout: float = 120.0,
) -> dict[str, Any]:
    if client is None or not model or not candidates:
        for item in candidates:
            item['_mechanisms'] = []
        return {
            'extractor': 'none',
            'candidate_count': len(candidates),
            'items_with_mechanisms': 0,
            'mechanism_count': 0,
            'errors': [],
        }

    errors: list[dict[str, Any]] = []
    items_with_mechanisms = 0
    mechanism_count = 0
    for start in range(0, len(candidates), max(1, batch_size)):
        batch = candidates[start : start + max(1, batch_size)]
        payload = [
            {
                'index': start + offset,
                'evidence_id': str(item.get('hyperedge_id') or item.get('id') or start + offset)[:180],
                'text': _mechanism_evidence_text(item),
            }
            for offset, item in enumerate(batch)
        ]
        user_prompt = json.dumps(
            {
                'question': question,
                'evidence_items': payload,
            },
            ensure_ascii=False,
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': MECHANISM_EXTRACTION_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0,
                timeout=timeout,
            )
            data = _json_from_text(response.choices[0].message.content or '{}')
            by_index = {}
            for entry in data.get('items') or []:
                try:
                    by_index[int(entry.get('index'))] = entry
                except (TypeError, ValueError):
                    continue
            for offset, item in enumerate(batch):
                idx = start + offset
                mechanisms = by_index.get(idx, {}).get('mechanisms') or []
                if not isinstance(mechanisms, list):
                    mechanisms = []
                mechanisms = mechanisms[:2]
                item['_mechanisms'] = mechanisms
                if mechanisms:
                    items_with_mechanisms += 1
                    mechanism_count += len(mechanisms)
        except Exception as exc:
            errors.append({'batch_start': start, 'error': repr(exc)})
            for item in batch:
                item['_mechanisms'] = []

    return {
        'extractor': f'llm:{model}',
        'candidate_count': len(candidates),
        'items_with_mechanisms': items_with_mechanisms,
        'mechanism_count': mechanism_count,
        'batch_size': batch_size,
        'errors': errors,
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def vector_redundancy(retriever: MultiOrderIterativeRetriever, candidate_id: str, selected_ids: set[str]) -> float:
    candidate_vec = retriever.hyperedge_vector_map.get(candidate_id)
    if candidate_vec is None or not selected_ids:
        return 0.0
    best = 0.0
    for other_id in selected_ids:
        other_vec = retriever.hyperedge_vector_map.get(other_id)
        if other_vec is None:
            continue
        best = max(best, float(cosine_similarity(np.asarray(candidate_vec), np.asarray(other_vec))))
    return best


def same_layer_candidates(
    retriever: MultiOrderIterativeRetriever,
    source_id: str,
    *,
    level: int,
    semantic_order: list[str],
    top_k_expand: int,
) -> set[str]:
    if level == retriever.max_level:
        source_children = retriever.children_by_level.get((source_id, retriever.max_level - 1), set())
        candidate_pool = set(semantic_order[: max(200, top_k_expand * 20)])
        for child in source_children:
            candidate_pool.update(retriever.l4_by_l3_child.get(child, set()))
    elif level == 0:
        source_parents = retriever.parent_by_level.get((source_id, 1), set())
        candidate_pool = set(semantic_order[: max(300, top_k_expand * 20)])
        for parent in source_parents:
            candidate_pool.update(retriever.l0_by_l1_parent.get(parent, set()))
    else:
        candidate_pool = set(semantic_order[: max(200, top_k_expand * 20)])

    source_entities = set(retriever.hyperedges[source_id].entities)
    for entity in source_entities:
        candidate_pool.update(retriever.hyperedges_by_entity_and_level.get((entity, level), set()))
    return candidate_pool


def structure_score(retriever: MultiOrderIterativeRetriever, source_id: str, candidate_id: str, level: int) -> float:
    if level == retriever.max_level:
        source_neighbors = retriever.children_by_level.get((source_id, retriever.max_level - 1), set())
        candidate_neighbors = retriever.children_by_level.get((candidate_id, retriever.max_level - 1), set())
    elif level == 0:
        source_neighbors = retriever.parent_by_level.get((source_id, 1), set())
        candidate_neighbors = retriever.parent_by_level.get((candidate_id, 1), set())
    else:
        source_neighbors = set()
        candidate_neighbors = set()
    return jaccard(set(source_neighbors), set(candidate_neighbors))


def expand_same_layer_variant(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    paths: list[RetrievalPath],
    *,
    level: int,
    visited: set[str],
    visited_l0: set[str],
    top_k_expand: int,
    mode: str,
    structure_alpha: float,
    redundancy_gamma: float,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    if mode not in EXPANSION_MODES:
        raise ValueError(f"Unsupported expansion mode: {mode}")

    by_end: dict[str, list[RetrievalPath]] = {}
    for path in paths:
        if path.current_layer == level:
            by_end.setdefault(path.last_edge, []).append(path)

    semantic_scores = retriever.semantic_scores_for_level(query, level)
    semantic_order = retriever.semantic_order_for_level(query, level)
    expanded_paths: list[RetrievalPath] = []
    logs: list[dict[str, Any]] = []
    selected_for_redundancy = set(visited_l0 if level == 0 else visited)

    for source_id, source_paths in by_end.items():
        scored: list[dict[str, Any]] = []
        for candidate_id in same_layer_candidates(
            retriever,
            source_id,
            level=level,
            semantic_order=semantic_order,
            top_k_expand=top_k_expand,
        ):
            if candidate_id not in retriever.hyperedge_vector_map:
                continue
            if candidate_id == source_id or candidate_id in visited:
                continue

            semantic = float(semantic_scores.get(candidate_id, 0.0))
            structure = structure_score(retriever, source_id, candidate_id, level)
            redundancy = 0.0
            if mode == "semantic_only":
                score = semantic
            elif mode == "semantic_weak_structure":
                score = semantic + structure_alpha * structure
            else:
                redundancy = vector_redundancy(retriever, candidate_id, selected_for_redundancy)
                score = semantic + structure_alpha * structure - redundancy_gamma * redundancy
            scored.append(
                {
                    "source_hyperedge_id": source_id,
                    "candidate_hyperedge_id": candidate_id,
                    "level": level,
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
                expanded_paths.append(append_path(path, candidate_id, level, f"L{level}_{mode}_expansion"))
            logs.append(item)

    expanded_paths = dedupe_paths(expanded_paths)
    expanded_ids = {path.last_edge for path in expanded_paths}
    visited.update(expanded_ids)
    if level == 0:
        visited_l0.update(expanded_ids)

    if level == retriever.max_level:
        output_paths = dedupe_paths(paths + expanded_paths)
    else:
        output_paths = expanded_paths

    return output_paths, {
        "stage": f"L{level}_same_layer_{mode}_expansion",
        "input_path_count": len(paths),
        "expanded_path_count": len(expanded_paths),
        "output_path_count": len(output_paths),
        "expanded_l2_count": len(expanded_ids) if level == retriever.max_level else None,
        "expanded_l0_count": len(expanded_ids) if level == 0 else None,
        "expansion_log_count": len(logs),
        "scoring": mode,
        "structure_alpha": structure_alpha,
        "redundancy_gamma": redundancy_gamma,
    }


def _normalize_path_scores(paths: list[RetrievalPath]) -> dict[str, float]:
    raw = [float(path.score) for path in paths]
    lo = min(raw) if raw else 0.0
    hi = max(raw) if raw else 0.0
    out: dict[str, float] = {}
    for path in paths:
        key = "\u241f".join(path.path_edges)
        if hi > lo:
            out[key] = (float(path.score) - lo) / (hi - lo)
        else:
            out[key] = 1.0 if raw else 0.0
    return out


def _path_similarity(a: RetrievalPath, b: RetrievalPath) -> float:
    avec = set(a.path_edges)
    bvec = set(b.path_edges)
    union = avec | bvec
    edge_sim = (len(avec & bvec) / len(union)) if union else 0.0
    text_sim = 0.0
    if a.text and b.text:
        text_sim = cosine_counter(Counter(words(a.text)), Counter(words(b.text)))
    return max(edge_sim, text_sim)


def _path_novelty(path: RetrievalPath, selected: list[RetrievalPath]) -> float:
    if not selected:
        return 1.0
    return 1.0 - max(_path_similarity(path, other) for other in selected)


def _path_branch_penalty(path: RetrievalPath, selected: list[RetrievalPath]) -> float:
    if not selected:
        return 0.0
    same_anchor = sum(1 for other in selected if other.anchor_l0 and other.anchor_l0 == path.anchor_l0)
    parent = path.path_edges[-2] if len(path.path_edges) >= 2 else ""
    same_parent = sum(1 for other in selected if parent and len(other.path_edges) >= 2 and other.path_edges[-2] == parent)
    return max(same_anchor, same_parent) / max(1, len(selected))


def path_level_mmr_select(
    retriever: MultiOrderIterativeRetriever,
    reranker: BaseReranker,
    query: str,
    candidate_paths: list[RetrievalPath],
    path_beam_size: int,
    *,
    candidate_cap: int | None = None,
    alpha: float = 0.85,
    beta: float = 0.15,
    delta: float = 0.05,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    candidate_paths = dedupe_paths(candidate_paths)
    candidate_count_before_cap = len(candidate_paths)
    if candidate_cap is not None and candidate_cap > 0 and len(candidate_paths) > candidate_cap:
        candidate_paths = sorted(candidate_paths, key=lambda path: "\u241f".join(path.path_edges))[:candidate_cap]
    for path in candidate_paths:
        path.text = build_path_text(retriever, path)
    scores = reranker.batch_score(query, [path.text for path in candidate_paths])
    scored = [RetrievalPath(**{**path.__dict__, "score": float(score)}) for path, score in zip(candidate_paths, scores)]
    scored.sort(key=lambda item: item.score, reverse=True)
    if not scored or path_beam_size <= 0:
        return [], {
            "candidate_count_before_cap": candidate_count_before_cap,
            "candidate_count_reranked": len(candidate_paths),
            "kept_count": 0,
            "path_beam_size": path_beam_size,
            "candidate_cap": candidate_cap,
            "reranker": reranker.name,
            "path_selection": "mmr_relevance_novelty_branch",
        }

    norm_scores = _normalize_path_scores(scored)
    selected = [scored[0]]
    remaining = scored[1:]
    debug = [
        {
            "path_edges": selected[0].path_edges,
            "reranker_score": float(selected[0].score),
            "reranker_score_normalized": norm_scores.get("\u241f".join(selected[0].path_edges), 0.0),
            "novelty": 1.0,
            "branch_penalty": 0.0,
            "final_score": None,
            "selection_step": 1,
            "selection_reason": "highest_reranker_score",
        }
    ]
    while remaining and len(selected) < path_beam_size:
        best_item = None
        best_score = -float("inf")
        best_debug: dict[str, Any] = {}
        for path in remaining:
            key = "\u241f".join(path.path_edges)
            rel = norm_scores.get(key, 0.0)
            nov = _path_novelty(path, selected)
            penalty = _path_branch_penalty(path, selected)
            final_score = alpha * rel + beta * nov - delta * penalty
            if final_score > best_score:
                best_score = final_score
                best_item = path
                best_debug = {
                    "path_edges": path.path_edges,
                    "reranker_score": float(path.score),
                    "reranker_score_normalized": rel,
                    "novelty": nov,
                    "branch_penalty": penalty,
                    "final_score": final_score,
                    "selection_step": len(selected) + 1,
                }
        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)
        debug.append(best_debug)

    return selected, {
        "candidate_count_before_cap": candidate_count_before_cap,
        "candidate_count_reranked": len(candidate_paths),
        "kept_count": len(selected),
        "path_beam_size": path_beam_size,
        "candidate_cap": candidate_cap,
        "reranker": reranker.name,
        "path_selection": "mmr_relevance_novelty_branch",
        "path_alpha": alpha,
        "path_beta": beta,
        "path_delta": delta,
        "selected_paths": debug,
    }


def cross_layer_path_step_path_mmr(
    retriever: MultiOrderIterativeRetriever,
    reranker: BaseReranker,
    query: str,
    paths: list[RetrievalPath],
    *,
    from_level: int,
    to_level: int,
    direction: str,
    visited: set[str],
    path_beam_size: int,
    candidate_cap: int | None = None,
    path_alpha: float = 0.85,
    path_beta: float = 0.15,
    path_delta: float = 0.05,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    stage_visited_snapshot = set(visited)
    candidates: list[RetrievalPath] = []
    for path in paths:
        if path.current_layer != from_level:
            continue
        if direction == "up":
            neighbor_ids = sorted(retriever.parent_by_level.get((path.last_edge, to_level), set()))
        elif direction == "down":
            neighbor_ids = sorted(retriever.children_by_level.get((path.last_edge, to_level), set()))
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        for neighbor_id in neighbor_ids:
            if neighbor_id in stage_visited_snapshot:
                continue
            candidates.append(append_path(path, neighbor_id, to_level, f"{direction}_L{from_level}_to_L{to_level}"))

    kept, info = path_level_mmr_select(
        retriever,
        reranker,
        query,
        candidates,
        path_beam_size,
        candidate_cap=candidate_cap,
        alpha=path_alpha,
        beta=path_beta,
        delta=path_delta,
    )
    added = {path.last_edge for path in kept if path.last_edge not in stage_visited_snapshot}
    visited.update(added)
    info.update(
        {
            "stage": f"{direction}_L{from_level}_to_L{to_level}",
            "from_level": from_level,
            "to_level": to_level,
            "input_path_count": len(paths),
            "added_visited_count": len(added),
            "unique_end_count": len({path.last_edge for path in kept}),
        }
    )
    return kept, info


def retrieve_l2_path_beam_v2_expansion_variant(
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
    l0_semantic_inject_k: int = 10,
    expansion_mode: str = "semantic_only",
    structure_alpha: float = 0.1,
    redundancy_gamma: float = 0.2,
    mechanism_top_m: int = 50,
    mechanism_alpha: float = 0.55,
    mechanism_beta: float = 0.20,
    mechanism_gamma: float = 0.20,
    mechanism_delta: float = 0.05,
    mechanism_extractor: str = "llm",
    mechanism_client: Any = None,
    mechanism_model: str = "",
    mechanism_extract_batch_size: int = 20,
    mechanism_extract_timeout: float = 120.0,
    path_selection_mode: str = "rerank",
    path_alpha: float = 0.85,
    path_beta: float = 0.15,
    path_delta: float = 0.05,
    final_selection_mode: str = "greedy",
) -> dict[str, Any]:
    cross_layer_step = cross_layer_path_step_path_mmr if path_selection_mode == "mmr" else cross_layer_path_step
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

        l1_paths, info = cross_layer_step(
            retriever,
            reranker,
            query,
            path_beam,
            from_level=0,
            to_level=1,
            direction="up",
            visited=visited,
            path_beam_size=path_beam_size,
            candidate_cap=candidate_cap,
            **({"path_alpha": path_alpha, "path_beta": path_beta, "path_delta": path_delta} if path_selection_mode == "mmr" else {}),
        )
        log["stages"].append(info)
        if not l1_paths:
            iteration_logs.append(log)
            break

        l2_paths, info = cross_layer_step(
            retriever,
            reranker,
            query,
            l1_paths,
            from_level=1,
            to_level=2,
            direction="up",
            visited=visited,
            path_beam_size=path_beam_size,
            candidate_cap=candidate_cap,
            **({"path_alpha": path_alpha, "path_beta": path_beta, "path_delta": path_delta} if path_selection_mode == "mmr" else {}),
        )
        log["stages"].append(info)
        if not l2_paths:
            iteration_logs.append(log)
            break

        l2_entry_paths, info = expand_same_layer_variant(
            retriever,
            query,
            l2_paths,
            level=2,
            visited=visited,
            visited_l0=visited_l0,
            top_k_expand=expansion_top_k,
            mode=expansion_mode,
            structure_alpha=structure_alpha,
            redundancy_gamma=redundancy_gamma,
        )
        log["stages"].append(info)

        down_l1_paths, info = cross_layer_step(
            retriever,
            reranker,
            query,
            l2_entry_paths,
            from_level=2,
            to_level=1,
            direction="down",
            visited=visited,
            path_beam_size=path_beam_size,
            candidate_cap=candidate_cap,
            **({"path_alpha": path_alpha, "path_beta": path_beta, "path_delta": path_delta} if path_selection_mode == "mmr" else {}),
        )
        log["stages"].append(info)
        if not down_l1_paths:
            iteration_logs.append(log)
            break

        down_l0_paths, info = cross_layer_step(
            retriever,
            reranker,
            query,
            down_l1_paths,
            from_level=1,
            to_level=0,
            direction="down",
            visited=visited,
            path_beam_size=path_beam_size,
            candidate_cap=candidate_cap,
            **({"path_alpha": path_alpha, "path_beta": path_beta, "path_delta": path_delta} if path_selection_mode == "mmr" else {}),
        )
        reached_l0 = {path.last_edge for path in down_l0_paths}
        visited_l0.update(reached_l0)
        info["reached_l0_count"] = len(reached_l0)
        log["stages"].append(info)
        if not down_l0_paths:
            iteration_logs.append(log)
            break

        next_paths, info = expand_same_layer_variant(
            retriever,
            query,
            down_l0_paths,
            level=0,
            visited=visited,
            visited_l0=visited_l0,
            top_k_expand=expansion_top_k,
            mode=expansion_mode,
            structure_alpha=structure_alpha,
            redundancy_gamma=redundancy_gamma,
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

        log["output_path_count"] = len(next_paths)
        log["visited_count"] = len(visited)
        log["visited_l0_count"] = len(visited_l0)
        iteration_logs.append(log)
        path_beam = next_paths

    reranker_top_m = len(visited_l0) if mechanism_top_m <= 0 else max(top_k_final, mechanism_top_m)
    reranked_l0, final_info = final_evidence_rerank(retriever, reranker, query, visited_l0, reranker_top_m)
    selection_pool_size = len(reranked_l0) if mechanism_top_m <= 0 else mechanism_top_m
    extraction_info = {'extractor': 'none'}
    if mechanism_extractor == "llm":
        extraction_info = extract_dynamic_mechanisms_for_candidates(
            client=mechanism_client,
            model=mechanism_model,
            question=query,
            candidates=reranked_l0[:selection_pool_size],
            batch_size=mechanism_extract_batch_size,
            timeout=mechanism_extract_timeout,
        )
    else:
        for item in reranked_l0:
            item['_mechanisms'] = []
    if final_selection_mode == "all_visited":
        top_l0 = reranked_l0[:selection_pool_size]
        selection_info = {
            "strategy": "all_visited_l0_no_final_selection",
            "candidate_count": len(reranked_l0),
            "candidate_pool_count": len(top_l0),
            "selected_count": len(top_l0),
            "top_m": selection_pool_size,
            "final_k": len(top_l0),
        }
    else:
        top_l0, selection_info = mechanism_chain_aware_select(
            reranked_l0,
            top_m=selection_pool_size,
            final_k=top_k_final,
            alpha=mechanism_alpha,
            beta=mechanism_beta,
            gamma=mechanism_gamma,
            delta=mechanism_delta,
        )
    final_info = {
        **final_info,
        "kept_count": len(top_l0),
        "reranker_top_m": reranker_top_m,
        "selection_pool_size": selection_pool_size,
        "mechanism_extraction": extraction_info,
        "selection": selection_info,
    }
    context = build_context(top_l0).replace(
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        f"[L2-PathBeam-VisitedAware-Rerank-v2 {expansion_mode} mechanism-chain-aware: Final L0 Evidence]",
    )
    return {
        "query": query,
        "strategy": f"L2-PathBeam-VisitedAware-Rerank-v2-{expansion_mode}-mechanism-chain-aware",
        "settings": {
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "candidate_cap": candidate_cap,
            "l0_semantic_inject_k": l0_semantic_inject_k,
            "same_layer_expansion_mode": expansion_mode,
            "structure_alpha": structure_alpha,
            "redundancy_gamma": redundancy_gamma,
            "mechanism_top_m": mechanism_top_m,
            "mechanism_alpha": mechanism_alpha,
            "mechanism_beta": mechanism_beta,
            "mechanism_gamma": mechanism_gamma,
            "mechanism_delta": mechanism_delta,
            "mechanism_extractor": mechanism_extractor,
            "mechanism_extract_batch_size": mechanism_extract_batch_size,
            "path_selection_mode": path_selection_mode,
            "path_alpha": path_alpha,
            "path_beta": path_beta,
            "path_delta": path_delta,
            "final_selection_mode": final_selection_mode,
            "reranker": reranker.name,
            "max_level": retriever.max_level,
        },
        "anchors": anchors,
        "iterations": iteration_logs,
        "final_rerank": final_info,
        "top_l0_hyperedges": top_l0,
        "selected_hyperedges": top_l0,
        "visited_l0_count": len(visited_l0),
        "visited_count": len(visited),
        "context": context,
    }


__all__ = ["retrieve_l2_path_beam_v2_expansion_variant", "summarize_path_counts"]
