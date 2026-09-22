#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import requests
from openai import OpenAI

from HKHG.prompt import GRAPH_FIELD_SEP
from hh_iterative_retrieval import MultiOrderIterativeRetriever, _json_default, cosine_similarity
from layer_retrieval_compare import embed_texts, load_graph, load_hyperedge_vectors, load_json, load_runtime_config
from Retrieve.lb_hgr_retrieval import answer_with_retrieval_context, load_retrieval_inputs, make_openai_client_from_values


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

ROLE_PHRASE_MAP = {
    "Condition": "触发条件",
    "Mechanism": "作用机制",
    "Result": "结果",
    "Object": "作用对象",
    "Subject": "研究对象",
    "Parameter": "关键参数",
    "Method": "处理方法",
    "Evidence": "证据信息",
    "Comparison": "对比对象",
    "Time": "时间信息",
    "Location": "位置信息",
}


@dataclass
class RetrievalPath:
    path_edges: list[str]
    path_layers: list[int]
    last_edge: str
    current_layer: int
    score: float = 0.0
    text: str = ""
    anchor_l0: str = ""
    reached_l0: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker:
    name = "base"

    def batch_score(self, query: str, evidence_texts: list[str]) -> list[float]:
        raise NotImplementedError


class ApiReranker(BaseReranker):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.name = f"api:{model}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/rerank"

    def batch_score(self, query: str, evidence_texts: list[str]) -> list[float]:
        if not evidence_texts:
            return []
        scores: list[float] = []
        for start in range(0, len(evidence_texts), self.batch_size):
            batch = evidence_texts[start : start + self.batch_size]
            scores.extend(_call_with_retry("rerank", self._score_batch, query, batch))
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return _parse_rerank_scores(response.json(), len(documents))


class LocalReranker(BaseReranker):
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else None,
        )
        self.reranker.to(device)
        self.reranker.eval()
        self._lock = threading.Lock()
        self.name = f"local:{model_path}:{device}"

    def batch_score(self, query: str, evidence_texts: list[str]) -> list[float]:
        if not evidence_texts:
            return []
        scores: list[float] = []
        # A single local cross-encoder instance is shared by question workers.
        # Serialize GPU forward passes to keep memory stable.
        with self._lock:
            for start in range(0, len(evidence_texts), self.batch_size):
                batch = evidence_texts[start : start + self.batch_size]
                pairs = [[query, text] for text in batch]
                with self.torch.no_grad():
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self.reranker(**inputs).logits
                    batch_scores = logits.view(-1).float().detach().cpu().tolist()
                scores.extend(float(score) for score in batch_scores)
        return scores


class SemanticFallbackReranker(BaseReranker):
    """Only for debugging path counts; not a formal reranker experiment."""

    name = "semantic-fallback-debug-only"

    def __init__(self, retriever: MultiOrderIterativeRetriever, client: Any, embedding_model: str) -> None:
        self.retriever = retriever
        self.client = client
        self.embedding_model = embedding_model

    def batch_score(self, query: str, evidence_texts: list[str]) -> list[float]:
        if not evidence_texts:
            return []
        query_vec = self.retriever.embed_query(query)
        vectors = embed_texts(self.client, self.embedding_model, evidence_texts)
        return [cosine_similarity(query_vec, np.asarray(vector, dtype=np.float32)) for vector in vectors]


def _call_with_retry(label: str, func, *args, attempts: int = 5, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()
            retryable = any(token in message for token in ["429", "rate limit", "timeout", "connection", "temporarily"])
            if not retryable or attempt >= attempts:
                raise
            sleep_seconds = min(4.0 * (2 ** (attempt - 1)), 60.0)
            print(f"    [retry] {label} attempt {attempt}/{attempts}; sleep {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
    raise last_exc


def _parse_rerank_scores(data: Any, expected: int) -> list[float]:
    scores = [0.0] * expected
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for idx, item in enumerate(data["results"]):
            doc_index = int(item.get("index", idx))
            if 0 <= doc_index < expected:
                scores[doc_index] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for idx, item in enumerate(data["data"][:expected]):
            scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores
    if isinstance(data, dict) and isinstance(data.get("scores"), list):
        return [float(value) for value in data["scores"][:expected]] + [0.0] * max(0, expected - len(data["scores"]))
    if isinstance(data, list):
        return [float(value) for value in data[:expected]] + [0.0] * max(0, expected - len(data))
    return scores


def clean_entity(raw: str) -> str:
    text = str(raw).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_fact_core(raw: str, max_chars: int = 500) -> str:
    text = str(raw or "").replace(GRAPH_FIELD_SEP, "<SEP>")
    parts = []
    for part in text.split("<SEP>"):
        cleaned = clean_entity(part)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    if not parts:
        return ""
    # L0 often stores the same sentence twice around <SEP>; keep the first
    # non-empty unique fact to avoid duplicating evidence for the reranker.
    fact = parts[0]
    if len(fact) > max_chars:
        fact = fact[:max_chars].rstrip() + "..."
    return fact


def parse_argument_roles(raw: str) -> list[tuple[str, str]]:
    try:
        parsed = json.loads(raw or "[]")
    except Exception:
        return []
    pairs: list[tuple[str, str]] = []
    if not isinstance(parsed, list):
        return pairs
    seen: set[tuple[str, str]] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entity = clean_entity(str(item.get("entity", "")))
        role = clean_entity(str(item.get("role", "")))
        if not entity or not role:
            continue
        key = (entity, role)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def build_evidence_text(retriever: MultiOrderIterativeRetriever, hyperedge_id: str, max_chars: int = 800) -> str:
    data = retriever.graph.nodes.get(hyperedge_id, {})
    fact = clean_fact_core(str(data.get("fact_core", "")))
    if not fact:
        hyperedge = retriever.hyperedges.get(hyperedge_id)
        fact = clean_fact_core(hyperedge.text if hyperedge else str(hyperedge_id))

    role_pairs = parse_argument_roles(str(data.get("argument_roles_json", "[]")))
    fragments = []
    for entity, role in role_pairs[:12]:
        phrase = ROLE_PHRASE_MAP.get(role, role or "相关实体")
        fragments.append(f"{entity}为{phrase}")
    if fragments:
        text = f"{fact}。其中，{'，'.join(fragments)}。"
    else:
        text = f"{fact}。"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def build_path_text(
    retriever: MultiOrderIterativeRetriever,
    path: RetrievalPath,
    max_edges: int = 5,
    max_chars: int = 3000,
) -> str:
    selected = path.path_edges[-max_edges:] if max_edges > 0 else path.path_edges
    text = "\n".join(build_evidence_text(retriever, edge_id) for edge_id in selected)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def path_key(path: RetrievalPath) -> str:
    return "\u241f".join(path.path_edges)


def dedupe_paths(paths: list[RetrievalPath]) -> list[RetrievalPath]:
    seen: set[str] = set()
    result: list[RetrievalPath] = []
    for path in paths:
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def make_anchor_path(retriever: MultiOrderIterativeRetriever, item: dict[str, Any]) -> RetrievalPath:
    edge_id = item["hyperedge_id"]
    return RetrievalPath(
        path_edges=[edge_id],
        path_layers=[0],
        last_edge=edge_id,
        current_layer=0,
        score=float(item.get("semantic_score", 0.0)),
        anchor_l0=edge_id,
        reached_l0=[edge_id],
        metadata={"source": "anchor"},
    )


def append_path(path: RetrievalPath, edge_id: str, layer: int, stage: str) -> RetrievalPath:
    reached_l0 = list(path.reached_l0)
    if layer == 0:
        reached_l0.append(edge_id)
    return RetrievalPath(
        path_edges=path.path_edges + [edge_id],
        path_layers=path.path_layers + [layer],
        last_edge=edge_id,
        current_layer=layer,
        score=path.score,
        anchor_l0=path.anchor_l0,
        reached_l0=reached_l0,
        metadata={"stage": stage},
    )


def path_level_rerank(
    retriever: MultiOrderIterativeRetriever,
    reranker: BaseReranker,
    query: str,
    candidate_paths: list[RetrievalPath],
    path_beam_size: int,
    candidate_cap: int | None = None,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    candidate_paths = dedupe_paths(candidate_paths)
    candidate_count_before_cap = len(candidate_paths)
    if candidate_cap is not None and candidate_cap > 0 and len(candidate_paths) > candidate_cap:
        # Engineering protection only. This is intentionally not semantic-topK;
        # it keeps deterministic path order and should remain disabled by default.
        candidate_paths = sorted(candidate_paths, key=path_key)[:candidate_cap]
    for path in candidate_paths:
        path.text = build_path_text(retriever, path)
    scores = reranker.batch_score(query, [path.text for path in candidate_paths])
    scored: list[RetrievalPath] = []
    for path, score in zip(candidate_paths, scores):
        scored.append(replace(path, score=float(score)))
    scored.sort(key=lambda item: item.score, reverse=True)
    kept = scored[:path_beam_size]
    return kept, {
        "candidate_count_before_cap": candidate_count_before_cap,
        "candidate_count_reranked": len(candidate_paths),
        "kept_count": len(kept),
        "path_beam_size": path_beam_size,
        "candidate_cap": candidate_cap,
        "reranker": reranker.name,
    }


def cross_layer_path_step(
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

    kept, info = path_level_rerank(retriever, reranker, query, candidates, path_beam_size, candidate_cap)
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


def expand_l2_paths(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    l2_paths: list[RetrievalPath],
    *,
    visited: set[str],
    top_k_expand: int,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    by_end: dict[str, list[RetrievalPath]] = {}
    for path in l2_paths:
        if path.current_layer == retriever.max_level:
            by_end.setdefault(path.last_edge, []).append(path)
    expanded, logs = retriever.expand_l4(set(by_end), query, top_k_l4_expand=top_k_expand, visited=visited)
    del expanded
    expanded_paths: list[RetrievalPath] = []
    for item in logs:
        source = item["source_hyperedge_id"]
        candidate = item["candidate_hyperedge_id"]
        for path in by_end.get(source, []):
            expanded_paths.append(append_path(path, candidate, retriever.max_level, "L2_expansion"))
    expanded_paths = dedupe_paths(expanded_paths)
    expanded_ids = {path.last_edge for path in expanded_paths}
    visited.update(expanded_ids)
    # Downward entry includes both original L2 paths and appended expanded-L2 paths.
    combined = dedupe_paths(l2_paths + expanded_paths)
    return combined, {
        "stage": "L2_same_layer_expansion",
        "input_path_count": len(l2_paths),
        "expanded_path_count": len(expanded_paths),
        "kept_original_path_count": len(l2_paths),
        "output_path_count": len(combined),
        "expanded_l2_count": len(expanded_ids),
        "expansion_log_count": len(logs),
    }


def expand_l0_paths(
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
    expanded, logs = retriever.expand_l0(set(by_end), query, top_k_l0_expand=top_k_expand, visited=visited)
    del expanded
    expanded_paths: list[RetrievalPath] = []
    for item in logs:
        source = item["source_hyperedge_id"]
        candidate = item["candidate_hyperedge_id"]
        for path in by_end.get(source, []):
            expanded_paths.append(append_path(path, candidate, 0, "L0_expansion"))
    expanded_paths = dedupe_paths(expanded_paths)
    expanded_ids = {path.last_edge for path in expanded_paths}
    visited.update(expanded_ids)
    visited_l0.update(expanded_ids)
    return expanded_paths, {
        "stage": "L0_same_layer_expansion",
        "input_path_count": len(l0_paths),
        "expanded_path_count": len(expanded_paths),
        "expanded_l0_count": len(expanded_ids),
        "expansion_log_count": len(logs),
    }


def inject_l0_semantic_anchor_paths(
    retriever: MultiOrderIterativeRetriever,
    query: str,
    *,
    visited: set[str],
    visited_l0: set[str],
    top_k: int,
    iteration: int,
) -> tuple[list[RetrievalPath], dict[str, Any]]:
    if top_k <= 0:
        return [], {
            "stage": "L0_semantic_anchor_injection",
            "iteration": iteration,
            "candidate_count_before_filter": 0,
            "candidate_count_after_filter": 0,
            "kept_count": 0,
            "top_k": top_k,
        }

    semantic_scores = retriever.semantic_scores_for_level(query, 0)
    semantic_order = retriever.semantic_order_for_level(query, 0)
    candidates = [hid for hid in semantic_order if hid not in visited]
    selected = candidates[:top_k]
    injected_paths = [
        RetrievalPath(
            path_edges=[hid],
            path_layers=[0],
            last_edge=hid,
            current_layer=0,
            score=float(semantic_scores.get(hid, 0.0)),
            text=build_evidence_text(retriever, hid),
            anchor_l0=hid,
            reached_l0=[hid],
            metadata={
                "source": "L0_semantic_anchor_injection",
                "iteration": iteration,
                "semantic_score": float(semantic_scores.get(hid, 0.0)),
            },
        )
        for hid in selected
    ]
    selected_ids = {path.last_edge for path in injected_paths}
    visited.update(selected_ids)
    visited_l0.update(selected_ids)
    return injected_paths, {
        "stage": "L0_semantic_anchor_injection",
        "iteration": iteration,
        "candidate_count_before_filter": len(semantic_order),
        "candidate_count_after_filter": len(candidates),
        "kept_count": len(injected_paths),
        "top_k": top_k,
        "semantic_injected_l0_count": len(selected_ids),
        "injected_l0_ids": selected,
    }


def final_evidence_rerank(
    retriever: MultiOrderIterativeRetriever,
    reranker: BaseReranker,
    query: str,
    visited_l0: set[str],
    top_k_final: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(hid for hid in visited_l0 if retriever.hyperedges.get(hid) and retriever.hyperedges[hid].level == 0)
    texts = [build_evidence_text(retriever, hid) for hid in candidates]
    scores = reranker.batch_score(query, texts)
    ranked: list[dict[str, Any]] = []
    for hid, text, score in zip(candidates, texts, scores):
        hyperedge = retriever.hyperedges.get(hid)
        ranked.append(
            {
                "hyperedge_id": hid,
                "level": 0,
                "final_score": float(score),
                "text": text,
                "entities": sorted(hyperedge.entities) if hyperedge else [],
                "source_text": retriever.source_text(hid),
            }
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[:top_k_final], {
        "stage": "final_evidence_rerank",
        "candidate_count": len(candidates),
        "kept_count": min(top_k_final, len(candidates)),
        "reranker": reranker.name,
    }


def build_context(top_l0_hyperedges: list[dict[str, Any]]) -> str:
    lines = [
        "[L2 Path-Beam Visited-Aware Rerank: Final L0 Evidence]",
        "Final evidence is selected by reranking rule-verbalized L0 facts against the question.",
    ]
    for idx, item in enumerate(top_l0_hyperedges, start=1):
        lines.append(f"{idx}. ({float(item.get('final_score', 0.0)):.4f}) {item['hyperedge_id']}")
        lines.append(str(item.get("text", ""))[:1200])
    return "\n".join(lines)


def retrieve_l2_path_beam_visited_aware_rerank(
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
    l0_semantic_inject_k: int = 0,
) -> dict[str, Any]:
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

        l1_paths, info = cross_layer_path_step(
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
        )
        log["stages"].append(info)
        if not l1_paths:
            iteration_logs.append(log)
            break

        l2_paths, info = cross_layer_path_step(
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
        )
        log["stages"].append(info)
        if not l2_paths:
            iteration_logs.append(log)
            break

        l2_entry_paths, info = expand_l2_paths(
            retriever,
            query,
            l2_paths,
            visited=visited,
            top_k_expand=expansion_top_k,
        )
        log["stages"].append(info)

        down_l1_paths, info = cross_layer_path_step(
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
        )
        log["stages"].append(info)
        if not down_l1_paths:
            iteration_logs.append(log)
            break

        down_l0_paths, info = cross_layer_path_step(
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
        )
        reached_l0 = {path.last_edge for path in down_l0_paths}
        visited_l0.update(reached_l0)
        info["reached_l0_count"] = len(reached_l0)
        log["stages"].append(info)
        if not down_l0_paths:
            iteration_logs.append(log)
            break

        next_paths, info = expand_l0_paths(
            retriever,
            query,
            down_l0_paths,
            visited=visited,
            visited_l0=visited_l0,
            top_k_expand=expansion_top_k,
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
        if not path_beam:
            break

    top_l0, final_info = final_evidence_rerank(retriever, reranker, query, visited_l0, top_k_final)
    strategy = "L2-PathBeam-VisitedAware-Rerank"
    if l0_semantic_inject_k > 0:
        strategy = "L2-PathBeam-VisitedAware-Rerank-v2"
    return {
        "query": query,
        "strategy": strategy,
        "settings": {
            "top_k_anchor": top_k_anchor,
            "expansion_top_k": expansion_top_k,
            "path_beam_size": path_beam_size,
            "top_k_final": top_k_final,
            "max_iter": max_iter,
            "candidate_cap": candidate_cap,
            "l0_semantic_inject_k": l0_semantic_inject_k,
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
        "context": build_context(top_l0),
    }


def retrieve_l2_path_beam_visited_aware_rerank_l0_semantic_inject(
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
) -> dict[str, Any]:
    return retrieve_l2_path_beam_visited_aware_rerank(
        retriever=retriever,
        query=query,
        reranker=reranker,
        top_k_anchor=top_k_anchor,
        expansion_top_k=expansion_top_k,
        path_beam_size=path_beam_size,
        top_k_final=top_k_final,
        max_iter=max_iter,
        candidate_cap=candidate_cap,
        l0_semantic_inject_k=l0_semantic_inject_k,
    )


def summarize_path_counts(result: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_candidate = 0
    max_kept = 0
    for item in result.get("iterations", []):
        iteration = item["iteration"]
        for stage in item.get("stages", []):
            candidate_count = int(stage.get("candidate_count_before_cap", stage.get("expanded_path_count", 0)) or 0)
            kept_count = int(stage.get("kept_count", stage.get("output_path_count", stage.get("expanded_path_count", 0))) or 0)
            max_candidate = max(max_candidate, candidate_count)
            max_kept = max(max_kept, kept_count)
            rows.append(
                {
                    "iteration": iteration,
                    "stage": stage.get("stage"),
                    "candidate_count": candidate_count,
                    "reranked_count": stage.get("candidate_count_reranked"),
                    "kept_count": kept_count,
                    "unique_end_count": stage.get("unique_end_count"),
                    "expanded_l2_count": stage.get("expanded_l2_count"),
                    "expanded_l0_count": stage.get("expanded_l0_count"),
                    "semantic_injected_l0_count": stage.get("semantic_injected_l0_count"),
                }
            )
    return {
        "max_stage_candidate_count": max_candidate,
        "max_stage_kept_count": max_kept,
        "visited_l0_count": result.get("visited_l0_count"),
        "visited_count": result.get("visited_count"),
        "final_evidence_candidate_count": (result.get("final_rerank") or {}).get("candidate_count"),
        "final_top_count": len(result.get("top_l0_hyperedges", [])),
        "stages": rows,
    }


def resolve_config(args: argparse.Namespace) -> tuple[OpenAI, str, str, str, str]:
    config: dict[str, Any] = {}
    if args.config:
        config = load_runtime_config(Path(args.config))
    llm_cfg = config.get("llm", {})
    emb_cfg = config.get("embedding", {})
    base_url = args.base_url or llm_cfg.get("base_url", "") or os.environ.get("OPENAI_BASE_URL", "")
    api_key = args.api_key or llm_cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    embedding_model = (
        args.embedding_model
        or emb_cfg.get("model_path", "")
        or emb_cfg.get("model", "")
        or os.environ.get("OPENAI_EMBEDDING_MODEL", "")
        or "text-embedding-3-large"
    )
    llm_model = args.llm_model or llm_cfg.get("model", "") or os.environ.get("OPENAI_MODEL", "") or "gpt-4o-mini"
    client, embedding_model = make_openai_client_from_values(
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
    )
    return client, embedding_model, llm_model, base_url, api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L2 path-beam visited-aware retrieval with path and evidence reranking.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--answer-style", choices=["short", "long"], default="short")
    parser.add_argument("--top-k-anchor", type=int, default=2)
    parser.add_argument("--expansion-top-k", type=int, default=2)
    parser.add_argument("--path-beam-size", type=int, default=8)
    parser.add_argument("--top-k-final", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--rerank-base-url", default="")
    parser.add_argument("--rerank-api-key", default="")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--semantic-fallback", action="store_true", help="Debug only; not a formal reranker result.")
    parser.add_argument("--include-summary", action="store_true")
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    client, embedding_model, llm_model, base_url, api_key = resolve_config(args)
    graph, text_chunks, hyperedge_vector_map, client, embedding_model = load_retrieval_inputs(
        args.workdir,
        client,
        embedding_model,
    )
    retriever = MultiOrderIterativeRetriever(
        graph=graph,
        text_chunks=text_chunks,
        hyperedge_vector_map=hyperedge_vector_map,
        client=client,
        embedding_model=embedding_model,
        max_level=2,
    )
    if args.semantic_fallback:
        reranker: BaseReranker = SemanticFallbackReranker(retriever, client, embedding_model)
    else:
        reranker = ApiReranker(
            base_url=args.rerank_base_url or base_url,
            api_key=args.rerank_api_key or api_key,
            model=args.rerank_model,
            batch_size=args.rerank_batch_size,
        )
    result = retrieve_l2_path_beam_visited_aware_rerank(
        retriever=retriever,
        query=args.query,
        reranker=reranker,
        top_k_anchor=args.top_k_anchor,
        expansion_top_k=args.expansion_top_k,
        path_beam_size=args.path_beam_size,
        top_k_final=args.top_k_final,
        max_iter=args.max_iter,
        candidate_cap=args.candidate_cap if args.candidate_cap > 0 else None,
    )
    if args.include_summary:
        result["path_count_summary"] = summarize_path_counts(result)
    if not args.retrieval_only:
        answer_data = answer_with_retrieval_context(
            client=client,
            llm_model=llm_model,
            question=args.query,
            retrieval_result=result,
            answer_style=args.answer_style,
        )
        result["answer"] = answer_data["answer"]
        result["generation"] = answer_data["raw_generation"]
        result["answer_style"] = args.answer_style
        result["llm_model"] = llm_model
    result["embedding_model"] = embedding_model

    payload = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    run_cli()
