#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from HKHG.prompt import GRAPH_FIELD_SEP
from layer_retrieval_compare import cosine_scores, embed_texts, get_hyperedge_level


def _call_with_retry(label: str, func, *args, attempts: int = 5, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime defensive
            last_exc = exc
            message = str(exc).lower()
            retryable = any(token in message for token in ["429", "rate limit", "too many requests", "timeout", "connection"])
            if not retryable or attempt >= attempts:
                raise
            sleep_seconds = min(4.0 * (2 ** (attempt - 1)), 60.0)
            print(f"    [retry] {label} attempt {attempt}/{attempts}; sleep {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
    raise last_exc  # pragma: no cover - defensive


def _strip_hyperedge_prefix(hyperedge_id: str) -> str:
    text = str(hyperedge_id)
    for prefix in ["<hyperedge L4>", "<hyperedge L3>", "<hyperedge L2>", "<hyperedge L1>", "<hyperedge>"]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class Hyperedge:
    hyperedge_id: str
    level: int
    text: str
    entities: frozenset[str]


@dataclass
class MultiOrderIterativeRetriever:
    graph: nx.Graph
    text_chunks: dict[str, Any]
    hyperedge_vector_map: dict[str, np.ndarray]
    client: Any
    embedding_model: str
    max_level: int = 4

    def __post_init__(self) -> None:
        self.hyperedges: dict[str, Hyperedge] = {}
        self.level_index: dict[int, list[str]] = defaultdict(list)
        self.parent_map: dict[str, list[str]] = defaultdict(list)
        self.children_map: dict[str, list[str]] = defaultdict(list)
        self.parent_by_level: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.children_by_level: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.l0_by_l1_parent: dict[str, set[str]] = defaultdict(set)
        self.l4_by_l3_child: dict[str, set[str]] = defaultdict(set)
        self.hyperedges_by_entity_and_level: dict[tuple[str, int], set[str]] = defaultdict(set)
        self._query_vec_cache: dict[str, np.ndarray] = {}
        self._semantic_cache: dict[tuple[str, int], dict[str, float]] = {}
        self._semantic_order_cache: dict[tuple[str, int], list[str]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for node_id, data in self.graph.nodes(data=True):
            if data.get("role") != "hyperedge":
                continue
            level = get_hyperedge_level(self.graph, node_id)
            if level > self.max_level:
                continue
            hyperedge = Hyperedge(
                hyperedge_id=node_id,
                level=level,
                text=self._format_hyperedge_text(node_id, data),
                entities=frozenset(self._get_entities(node_id)),
            )
            self.hyperedges[node_id] = hyperedge
            self.level_index[level].append(node_id)
            for entity in hyperedge.entities:
                self.hyperedges_by_entity_and_level[(entity, level)].add(node_id)

        for left, right, edge_data in self.graph.edges(data=True):
            if edge_data.get("role") != "contains_hyperedge":
                continue
            if left not in self.hyperedges or right not in self.hyperedges:
                continue
            left_level = self.hyperedges[left].level
            right_level = self.hyperedges[right].level
            if left_level == right_level:
                continue
            parent_id, child_id = (left, right) if left_level > right_level else (right, left)
            self.parent_map[child_id].append(parent_id)
            self.children_map[parent_id].append(child_id)
            self.parent_by_level[(child_id, self.hyperedges[parent_id].level)].add(parent_id)
            self.children_by_level[(parent_id, self.hyperedges[child_id].level)].add(child_id)
            if self.hyperedges[child_id].level == 0 and self.hyperedges[parent_id].level == 1:
                self.l0_by_l1_parent[parent_id].add(child_id)
            if self.hyperedges[parent_id].level == self.max_level and self.hyperedges[child_id].level == self.max_level - 1:
                self.l4_by_l3_child[child_id].add(parent_id)

    def _format_hyperedge_text(self, hyperedge_id: str, data: dict[str, Any]) -> str:
        parts: list[str] = []
        fact_core = str(data.get("fact_core", "")).strip()
        if fact_core:
            parts.append(fact_core.replace(GRAPH_FIELD_SEP, " "))
        generated_entities = str(data.get("generated_entities", "")).strip()
        if generated_entities:
            parts.append(generated_entities)
        parts.append(_strip_hyperedge_prefix(hyperedge_id))
        return "\n".join(dict.fromkeys(part for part in parts if part)).strip()

    def _get_entities(self, hyperedge_id: str) -> set[str]:
        entities: set[str] = set()
        if hyperedge_id not in self.graph:
            return entities
        for neighbor in self.graph.neighbors(hyperedge_id):
            edge_data = self.graph.get_edge_data(hyperedge_id, neighbor) or {}
            node_data = self.graph.nodes[neighbor]
            if edge_data.get("role") == "link" and node_data.get("role") == "entity":
                entities.add(str(neighbor).strip().upper())
        return entities

    def embed_query(self, query: str) -> np.ndarray:
        if query not in self._query_vec_cache:
            self._query_vec_cache[query] = _call_with_retry(
                "multi_order_embed",
                embed_texts,
                self.client,
                self.embedding_model,
                [query],
            )[0]
        return self._query_vec_cache[query]

    def semantic_score(self, query_vec: np.ndarray, hyperedge_id: str) -> float:
        return cosine_similarity(query_vec, self.hyperedge_vector_map.get(hyperedge_id))

    def semantic_scores_for_level(self, query: str, level: int) -> dict[str, float]:
        key = (query, level)
        if key in self._semantic_cache:
            return self._semantic_cache[key]
        query_vec = self.embed_query(query)
        ids = [hid for hid in self.level_index.get(level, []) if hid in self.hyperedge_vector_map]
        if not ids:
            self._semantic_cache[key] = {}
            self._semantic_order_cache[key] = []
            return self._semantic_cache[key]
        matrix = np.asarray([self.hyperedge_vector_map[hid] for hid in ids], dtype=np.float32)
        scores = cosine_scores(query_vec, matrix)
        score_map = {hid: float(score) for hid, score in zip(ids, scores)}
        self._semantic_cache[key] = score_map
        self._semantic_order_cache[key] = sorted(ids, key=lambda hid: score_map[hid], reverse=True)
        return score_map

    def semantic_order_for_level(self, query: str, level: int) -> list[str]:
        self.semantic_scores_for_level(query, level)
        return self._semantic_order_cache.get((query, level), [])

    def initial_graph_anchoring(self, query: str, top_k_anchor: int = 5) -> list[dict[str, Any]]:
        query_vec = self.embed_query(query)
        l0_ids = [hid for hid in self.level_index.get(0, []) if hid in self.hyperedge_vector_map]
        if not l0_ids:
            return []
        matrix = np.asarray([self.hyperedge_vector_map[hid] for hid in l0_ids], dtype=np.float32)
        scores = cosine_scores(query_vec, matrix)
        order = np.argsort(scores)[::-1][:top_k_anchor]
        return [
            {
                "hyperedge_id": l0_ids[int(idx)],
                "level": 0,
                "semantic_score": float(scores[int(idx)]),
                "text": self.hyperedges[l0_ids[int(idx)]].text,
                "entities": sorted(self.hyperedges[l0_ids[int(idx)]].entities),
            }
            for idx in order
        ]

    def upward_traverse(self, current_l0: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
        frontier = {hid for hid in current_l0 if self.hyperedges.get(hid) and self.hyperedges[hid].level == 0}
        paths: list[dict[str, Any]] = []
        for next_level in range(1, self.max_level + 1):
            next_frontier: set[str] = set()
            for child in frontier:
                for parent in self.parent_by_level.get((child, next_level), set()):
                    next_frontier.add(parent)
                    paths.append({"direction": "up", "from": child, "to": parent, "from_level": next_level - 1, "to_level": next_level})
            frontier = next_frontier
            if not frontier:
                break
        return frontier if frontier else set(), paths

    def downward_traverse(self, current_l4: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
        frontier = {hid for hid in current_l4 if self.hyperedges.get(hid) and self.hyperedges[hid].level == self.max_level}
        paths: list[dict[str, Any]] = []
        for next_level in range(self.max_level - 1, -1, -1):
            next_frontier: set[str] = set()
            for parent in frontier:
                for child in self.children_by_level.get((parent, next_level), set()):
                    next_frontier.add(child)
                    paths.append({"direction": "down", "from": parent, "to": child, "from_level": next_level + 1, "to_level": next_level})
            frontier = next_frontier
            if not frontier:
                break
        return {hid for hid in frontier if self.hyperedges.get(hid) and self.hyperedges[hid].level == 0}, paths

    def downward_traverse_constrained(
        self,
        current_l4: set[str],
        query: str,
        max_l3_per_l4: int = 5,
        max_l2_per_l3: int = 5,
        max_l1_per_l2: int = 5,
        max_l0_per_l1: int = 10,
        max_downward_l0_per_iter: int = 300,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        frontier = {hid for hid in current_l4 if self.hyperedges.get(hid) and self.hyperedges[hid].level == self.max_level}
        paths: list[dict[str, Any]] = []
        beam_by_level = {
            self.max_level - 1: max_l3_per_l4,
            self.max_level - 2: max_l2_per_l3,
            self.max_level - 3: max_l1_per_l2,
            0: max_l0_per_l1,
        }
        for next_level in range(self.max_level - 1, -1, -1):
            semantic_scores = self.semantic_scores_for_level(query, next_level)
            next_frontier: set[str] = set()
            per_parent_limit = max(beam_by_level.get(next_level, 5), 1)
            for parent in frontier:
                children = list(self.children_by_level.get((parent, next_level), set()))
                children.sort(key=lambda child: semantic_scores.get(child, 0.0), reverse=True)
                for child in children[:per_parent_limit]:
                    next_frontier.add(child)
                    paths.append(
                        {
                            "direction": "down",
                            "from": parent,
                            "to": child,
                            "from_level": next_level + 1,
                            "to_level": next_level,
                            "semantic_score": float(semantic_scores.get(child, 0.0)),
                            "beam_limit": per_parent_limit,
                            "constrained": True,
                        }
                    )
            frontier = next_frontier
            if not frontier:
                break
        l0 = {hid for hid in frontier if self.hyperedges.get(hid) and self.hyperedges[hid].level == 0}
        if len(l0) > max_downward_l0_per_iter:
            semantic_scores = self.semantic_scores_for_level(query, 0)
            l0 = set(sorted(l0, key=lambda hid: semantic_scores.get(hid, 0.0), reverse=True)[:max_downward_l0_per_iter])
        return l0, paths

    def expand_l4(
        self,
        current_l4: set[str],
        query: str,
        top_k_l4_expand: int = 3,
        visited: set[str] | None = None,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        visited = visited or set()
        semantic_scores = self.semantic_scores_for_level(query, self.max_level)
        semantic_fallback = self.semantic_order_for_level(query, self.max_level)[: max(200, top_k_l4_expand * 20)]
        expanded: set[str] = set()
        logs: list[dict[str, Any]] = []
        for source_id in sorted(current_l4):
            source_children_l3 = self.children_by_level.get((source_id, self.max_level - 1), set())
            source_entities = set(self.hyperedges[source_id].entities)
            candidate_pool: set[str] = set(semantic_fallback)
            for child_l3 in source_children_l3:
                candidate_pool.update(self.l4_by_l3_child.get(child_l3, set()))
            for entity in source_entities:
                candidate_pool.update(self.hyperedges_by_entity_and_level.get((entity, self.max_level), set()))
            scored: list[dict[str, Any]] = []
            for candidate_id in candidate_pool:
                if candidate_id not in self.hyperedge_vector_map:
                    continue
                if candidate_id == source_id or candidate_id in visited:
                    continue
                candidate_children_l3 = self.children_by_level.get((candidate_id, self.max_level - 1), set())
                shared_child_score = jaccard_similarity(source_children_l3, candidate_children_l3)
                semantic = semantic_scores.get(candidate_id, 0.0)
                shared_entity_score = jaccard_similarity(source_entities, set(self.hyperedges[candidate_id].entities))
                score = (shared_child_score + semantic + shared_entity_score) / 3.0
                scored.append(
                    {
                        "source_hyperedge_id": source_id,
                        "candidate_hyperedge_id": candidate_id,
                        "level": self.max_level,
                        "score": float(score),
                        "shared_child_score": float(shared_child_score),
                        "semantic_score": float(semantic),
                        "shared_entity_score": float(shared_entity_score),
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            picked = scored[:top_k_l4_expand]
            expanded.update(item["candidate_hyperedge_id"] for item in picked)
            logs.extend(picked)
        return expanded, logs

    def expand_l0(
        self,
        current_l0: set[str],
        query: str,
        top_k_l0_expand: int = 5,
        visited: set[str] | None = None,
    ) -> tuple[set[str], list[dict[str, Any]]]:
        visited = visited or set()
        semantic_scores = self.semantic_scores_for_level(query, 0)
        semantic_fallback = self.semantic_order_for_level(query, 0)[: max(300, top_k_l0_expand * 20)]
        expanded: set[str] = set()
        logs: list[dict[str, Any]] = []
        for source_id in sorted(current_l0):
            source_parents_l1 = self.parent_by_level.get((source_id, 1), set())
            source_entities = set(self.hyperedges[source_id].entities)
            candidate_pool: set[str] = set(semantic_fallback)
            for parent_l1 in source_parents_l1:
                candidate_pool.update(self.l0_by_l1_parent.get(parent_l1, set()))
            for entity in source_entities:
                candidate_pool.update(self.hyperedges_by_entity_and_level.get((entity, 0), set()))
            scored: list[dict[str, Any]] = []
            for candidate_id in candidate_pool:
                if candidate_id not in self.hyperedge_vector_map:
                    continue
                if candidate_id == source_id or candidate_id in visited:
                    continue
                candidate_parents_l1 = self.parent_by_level.get((candidate_id, 1), set())
                shared_parent_score = jaccard_similarity(source_parents_l1, candidate_parents_l1)
                semantic = semantic_scores.get(candidate_id, 0.0)
                shared_entity_score = jaccard_similarity(source_entities, set(self.hyperedges[candidate_id].entities))
                score = (shared_parent_score + semantic + shared_entity_score) / 3.0
                scored.append(
                    {
                        "source_hyperedge_id": source_id,
                        "candidate_hyperedge_id": candidate_id,
                        "level": 0,
                        "score": float(score),
                        "shared_parent_score": float(shared_parent_score),
                        "semantic_score": float(semantic),
                        "shared_entity_score": float(shared_entity_score),
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            picked = scored[:top_k_l0_expand]
            expanded.update(item["candidate_hyperedge_id"] for item in picked)
            logs.extend(picked)
        return expanded, logs

    def final_rerank(self, query: str, l0_hyperedges: set[str], top_k_final: int = 10) -> list[dict[str, Any]]:
        query_vec = self.embed_query(query)
        ranked = []
        for hid in l0_hyperedges:
            if self.hyperedges.get(hid) and self.hyperedges[hid].level == 0 and hid in self.hyperedge_vector_map:
                ranked.append(
                    {
                        "hyperedge_id": hid,
                        "level": 0,
                        "final_score": self.semantic_score(query_vec, hid),
                        "text": self.hyperedges[hid].text,
                        "entities": sorted(self.hyperedges[hid].entities),
                        "source_text": self.source_text(hid),
                    }
                )
        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked[:top_k_final]

    def source_text(self, hyperedge_id: str) -> str:
        data = self.graph.nodes.get(hyperedge_id, {})
        source_id = str(data.get("source_id", "")).strip()
        if not source_id:
            return ""
        chunks: list[str] = []
        for chunk_id in [item.strip() for item in source_id.split(GRAPH_FIELD_SEP) if item.strip()]:
            chunk = self.text_chunks.get(chunk_id)
            if not chunk:
                continue
            content = str(chunk.get("content", "")).strip()
            if content:
                chunks.append(content)
        return "\n".join(dict.fromkeys(chunks))

    def multi_order_iterative_retrieval(
        self,
        query: str,
        top_k_anchor: int = 5,
        top_k_l4_expand: int = 3,
        top_k_l0_expand: int = 5,
        top_k_final: int = 10,
        max_iter: int = 3,
    ) -> dict[str, Any]:
        anchors = self.initial_graph_anchoring(query, top_k_anchor)
        current_l0 = {item["hyperedge_id"] for item in anchors}
        visited: set[str] = set(current_l0)
        visited_l0: set[str] = set(current_l0)
        iteration_logs: list[dict[str, Any]] = []

        for iteration in range(max_iter):
            input_l0 = set(current_l0)
            reached_l4, upward_paths = self.upward_traverse(input_l0)
            new_reached_l4 = reached_l4 - visited
            visited.update(new_reached_l4)

            expanded_l4, l4_expansion_log = self.expand_l4(
                current_l4=new_reached_l4,
                query=query,
                top_k_l4_expand=top_k_l4_expand,
                visited=visited,
            )
            visited.update(expanded_l4)
            l4_for_downward = new_reached_l4 | expanded_l4

            reached_l0, downward_paths = self.downward_traverse(l4_for_downward)
            new_reached_l0 = reached_l0 - visited
            visited.update(new_reached_l0)
            visited_l0.update(new_reached_l0)

            expanded_l0, l0_expansion_log = self.expand_l0(
                current_l0=new_reached_l0,
                query=query,
                top_k_l0_expand=top_k_l0_expand,
                visited=visited,
            )
            visited.update(expanded_l0)
            visited_l0.update(expanded_l0)

            current_l0 = expanded_l0
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "input_l0": sorted(input_l0),
                    "reached_l4": sorted(new_reached_l4),
                    "expanded_l4": sorted(expanded_l4),
                    "reached_l0": sorted(new_reached_l0),
                    "expanded_l0": sorted(expanded_l0),
                    "upward_paths": upward_paths,
                    "downward_paths": downward_paths,
                    "l4_expansion_log": l4_expansion_log,
                    "l0_expansion_log": l0_expansion_log,
                }
            )
            if not current_l0:
                break

        top_l0 = self.final_rerank(query, visited_l0, top_k_final)
        return {
            "query": query,
            "top_l0_hyperedges": top_l0,
            "selected_hyperedges": top_l0,
            "anchors": anchors,
            "iterations": iteration_logs,
            "visited_l0_count": len(visited_l0),
            "visited_count": len(visited),
            "context": build_retrieval_context(top_l0),
        }

    def multi_order_iterative_retrieval_constrained(
        self,
        query: str,
        top_k_anchor: int = 5,
        top_k_l4_expand: int = 3,
        top_k_l0_expand: int = 5,
        top_k_final: int = 10,
        max_iter: int = 3,
        max_l3_per_l4: int = 5,
        max_l2_per_l3: int = 5,
        max_l1_per_l2: int = 5,
        max_l0_per_l1: int = 10,
        max_downward_l0_per_iter: int = 300,
    ) -> dict[str, Any]:
        anchors = self.initial_graph_anchoring(query, top_k_anchor)
        current_l0 = {item["hyperedge_id"] for item in anchors}
        visited: set[str] = set(current_l0)
        visited_l0: set[str] = set(current_l0)
        iteration_logs: list[dict[str, Any]] = []

        for iteration in range(max_iter):
            input_l0 = set(current_l0)
            reached_l4, upward_paths = self.upward_traverse(input_l0)
            new_reached_l4 = reached_l4 - visited
            visited.update(new_reached_l4)

            expanded_l4, l4_expansion_log = self.expand_l4(
                current_l4=new_reached_l4,
                query=query,
                top_k_l4_expand=top_k_l4_expand,
                visited=visited,
            )
            visited.update(expanded_l4)
            l4_for_downward = new_reached_l4 | expanded_l4

            reached_l0, downward_paths = self.downward_traverse_constrained(
                current_l4=l4_for_downward,
                query=query,
                max_l3_per_l4=max_l3_per_l4,
                max_l2_per_l3=max_l2_per_l3,
                max_l1_per_l2=max_l1_per_l2,
                max_l0_per_l1=max_l0_per_l1,
                max_downward_l0_per_iter=max_downward_l0_per_iter,
            )
            new_reached_l0 = reached_l0 - visited
            visited.update(new_reached_l0)
            visited_l0.update(new_reached_l0)

            expanded_l0, l0_expansion_log = self.expand_l0(
                current_l0=new_reached_l0,
                query=query,
                top_k_l0_expand=top_k_l0_expand,
                visited=visited,
            )
            visited.update(expanded_l0)
            visited_l0.update(expanded_l0)

            current_l0 = expanded_l0
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "constrained": True,
                    "input_l0": sorted(input_l0),
                    "reached_l4": sorted(new_reached_l4),
                    "expanded_l4": sorted(expanded_l4),
                    "reached_l0": sorted(new_reached_l0),
                    "expanded_l0": sorted(expanded_l0),
                    "upward_paths": upward_paths,
                    "downward_paths": downward_paths,
                    "l4_expansion_log": l4_expansion_log,
                    "l0_expansion_log": l0_expansion_log,
                    "constraints": {
                        "max_l3_per_l4": max_l3_per_l4,
                        "max_l2_per_l3": max_l2_per_l3,
                        "max_l1_per_l2": max_l1_per_l2,
                        "max_l0_per_l1": max_l0_per_l1,
                        "max_downward_l0_per_iter": max_downward_l0_per_iter,
                    },
                }
            )
            if not current_l0:
                break

        top_l0 = self.final_rerank(query, visited_l0, top_k_final)
        return {
            "query": query,
            "strategy": "multi_order_iterative_retrieval_constrained",
            "top_l0_hyperedges": top_l0,
            "selected_hyperedges": top_l0,
            "anchors": anchors,
            "iterations": iteration_logs,
            "visited_l0_count": len(visited_l0),
            "visited_count": len(visited),
            "context": build_retrieval_context(top_l0),
        }


def build_retrieval_context(top_l0_hyperedges: list[dict[str, Any]]) -> str:
    lines = ["[Multi-order Iterative Retrieval: Final L0 Hyperedges]"]
    for idx, item in enumerate(top_l0_hyperedges, start=1):
        lines.append(f"{idx}. ({item['final_score']:.4f}) {item['hyperedge_id']}")
        if item.get("source_text"):
            lines.append(str(item["source_text"])[:1000])
    return "\n".join(lines)


def initial_graph_anchoring(retriever: MultiOrderIterativeRetriever, query: str, top_k_anchor: int = 5) -> list[dict[str, Any]]:
    return retriever.initial_graph_anchoring(query, top_k_anchor)


def upward_traverse(retriever: MultiOrderIterativeRetriever, current_l0: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
    return retriever.upward_traverse(current_l0)


def downward_traverse(retriever: MultiOrderIterativeRetriever, current_l4: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
    return retriever.downward_traverse(current_l4)


def expand_l4(
    retriever: MultiOrderIterativeRetriever,
    current_l4: set[str],
    query: str,
    top_k_l4_expand: int = 3,
    visited: set[str] | None = None,
) -> tuple[set[str], list[dict[str, Any]]]:
    return retriever.expand_l4(current_l4, query, top_k_l4_expand, visited)


def expand_l0(
    retriever: MultiOrderIterativeRetriever,
    current_l0: set[str],
    query: str,
    top_k_l0_expand: int = 5,
    visited: set[str] | None = None,
) -> tuple[set[str], list[dict[str, Any]]]:
    return retriever.expand_l0(current_l0, query, top_k_l0_expand, visited)


def final_rerank(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    l0_hyperedges: set[str],
    top_k_final: int = 10,
) -> list[dict[str, Any]]:
    return retriever.final_rerank(query, l0_hyperedges, top_k_final)


def multi_order_iterative_retrieval(
    *,
    query: str,
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    hyperedge_vector_map: dict[str, np.ndarray],
    client: Any,
    embedding_model: str,
    top_k_anchor: int = 5,
    top_k_l4_expand: int = 3,
    top_k_l0_expand: int = 5,
    top_k_final: int = 10,
    max_iter: int = 3,
    max_level: int = 4,
) -> dict[str, Any]:
    retriever = MultiOrderIterativeRetriever(
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        max_level=max_level,
    )
    return retriever.multi_order_iterative_retrieval(
        query=query,
        top_k_anchor=top_k_anchor,
        top_k_l4_expand=top_k_l4_expand,
        top_k_l0_expand=top_k_l0_expand,
        top_k_final=top_k_final,
        max_iter=max_iter,
    )


def multi_order_iterative_retrieval_constrained(
    *,
    query: str,
    graph: nx.Graph,
    text_chunks: dict[str, Any],
    hyperedge_vector_map: dict[str, np.ndarray],
    client: Any,
    embedding_model: str,
    top_k_anchor: int = 5,
    top_k_l4_expand: int = 3,
    top_k_l0_expand: int = 5,
    top_k_final: int = 10,
    max_iter: int = 3,
    max_level: int = 4,
    max_l3_per_l4: int = 5,
    max_l2_per_l3: int = 5,
    max_l1_per_l2: int = 5,
    max_l0_per_l1: int = 10,
    max_downward_l0_per_iter: int = 300,
) -> dict[str, Any]:
    retriever = MultiOrderIterativeRetriever(
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        max_level=max_level,
    )
    return retriever.multi_order_iterative_retrieval_constrained(
        query=query,
        top_k_anchor=top_k_anchor,
        top_k_l4_expand=top_k_l4_expand,
        top_k_l0_expand=top_k_l0_expand,
        top_k_final=top_k_final,
        max_iter=max_iter,
        max_l3_per_l4=max_l3_per_l4,
        max_l2_per_l3=max_l2_per_l3,
        max_l1_per_l2=max_l1_per_l2,
        max_l0_per_l1=max_l0_per_l1,
        max_downward_l0_per_iter=max_downward_l0_per_iter,
    )


def hierarchical_reasoning_path_retrieve(**kwargs) -> dict[str, Any]:
    """Backward-compatible entrypoint for older scripts; now uses the new pure graph strategy."""
    mapping = {
        "top_n_seed": "top_k_anchor",
        "max_evidence": "top_k_final",
    }
    normalized = dict(kwargs)
    for old_key, new_key in mapping.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)
    for unsupported in ["llm_model_name", "top_p_parent", "top_k_expand", "max_paths", "confidence_threshold"]:
        normalized.pop(unsupported, None)
    constrained = bool(normalized.pop("constrained", False))
    if constrained:
        return multi_order_iterative_retrieval_constrained(**normalized)
    return multi_order_iterative_retrieval(**normalized)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)
