import asyncio
import json
import math
import re
from tqdm.asyncio import tqdm as tqdm_async

from tenacity import retry, stop_after_attempt, wait_none, retry_if_result, RetryError
import warnings
import random
from collections import Counter, defaultdict, deque
import copy
from itertools import product
from typing import Any, Union, List, Optional, Dict, Iterable
import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    parse_llm_result_into_tuples,
    parse_llm_result_into_lists,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    safety_margin = global_config.get("llm_input_token_safety_margin", 512)
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]

    prompt_token_budget = max(llm_max_tokens - summary_max_tokens - safety_margin, 1)

    def _build_summary_prompt(description_text: str) -> str:
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description_text.split(GRAPH_FIELD_SEP),
            language=language,
        )
        return prompt_template.format(**context_base)

    use_description = decode_tokens_by_tiktoken(
        tokens[: min(len(tokens), prompt_token_budget)],
        model_name=tiktoken_model_name,
    )
    use_prompt = _build_summary_prompt(use_description)
    prompt_tokens = len(
        encode_string_by_tiktoken(use_prompt, model_name=tiktoken_model_name)
    )
    if prompt_tokens > prompt_token_budget:
        low = 0
        high = min(len(tokens), prompt_token_budget)
        best_prompt = _build_summary_prompt("")
        while low <= high:
            mid = (low + high) // 2
            candidate_description = decode_tokens_by_tiktoken(
                tokens[:mid], model_name=tiktoken_model_name
            )
            candidate_prompt = _build_summary_prompt(candidate_description)
            candidate_prompt_tokens = len(
                encode_string_by_tiktoken(
                    candidate_prompt, model_name=tiktoken_model_name
                )
            )
            if candidate_prompt_tokens <= prompt_token_budget:
                best_prompt = candidate_prompt
                low = mid + 1
            else:
                high = mid - 1
        use_prompt = best_prompt

    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )


def _normalize_argument_role(raw_role: str | None) -> str:
    role = clean_str(str(raw_role or "").strip())
    if not role:
        return "Unrecognized"
    role = role.replace("-", "_").replace(" ", "_")
    role = re.sub(r"[^A-Za-z0-9_]", "", role)
    if not role:
        return "Unrecognized"
    return role[:1].upper() + role[1:]


def _canonical_entity_key(raw_entity: Any) -> str:
    text = clean_str(str(raw_entity or "").strip())
    # Graph candidates may be stored as '"DISPLAY"', while the LLM often returns DISPLAY.
    # Compare entities on a quote-insensitive canonical form, then restore the graph entity name.
    text = text.replace('\"', '"').strip()
    text = text.strip('"').strip("'").strip()
    return clean_str(text.upper())


def _normalize_fact_core(value: Any) -> str:
    text = clean_str(str(value or "").strip())
    text = text.replace('\\"', '"').strip()
    text = text.strip('"').strip("'").strip()
    return clean_str(text)


def _normalize_argument_roles(argument_roles: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in argument_roles or []:
        if not isinstance(item, dict):
            continue
        entity_name = clean_str(str(item.get("entity", "")).upper())
        if not entity_name:
            continue
        role = _normalize_argument_role(item.get("role"))
        key = (entity_name, role)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"entity": entity_name, "role": role})
    return normalized


def _load_argument_roles(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return _normalize_argument_roles(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return _normalize_argument_roles(json.loads(text))
        except Exception:
            return []
    return []


def _dump_argument_roles(argument_roles: list[dict]) -> str:
    return json.dumps(_normalize_argument_roles(argument_roles), ensure_ascii=False)


def _merge_argument_roles(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for item in _normalize_argument_roles(group):
            key = (item["entity"], item["role"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _primary_argument_role(argument_roles: list[dict]) -> str:
    unique_roles = []
    seen = set()
    for item in _normalize_argument_roles(argument_roles):
        role = item["role"]
        if role in seen:
            continue
        seen.add(role)
        unique_roles.append(role)
    if not unique_roles:
        return "Unrecognized"
    if len(unique_roles) == 1:
        return unique_roles[0]
    return GRAPH_FIELD_SEP.join(unique_roles)


def _format_hyperedge_embedding_content(
    hyperedge_name: str,
    fact_core: str | None = None,
    argument_roles: list[dict] | None = None,
) -> str:
    core = _normalize_fact_core(fact_core or _strip_hyperedge_prefix(hyperedge_name))
    arguments = _normalize_argument_roles(argument_roles)
    if not arguments:
        return core or _strip_hyperedge_prefix(hyperedge_name)
    rendered_arguments = "\n".join(
        f"{item['role']}: {item['entity']}" for item in arguments
    )
    return f"{core}\nArguments:\n{rendered_arguments}".strip()


def _build_hyperedge_entity_map_from_records(
    maybe_nodes: dict[str, list[dict]],
) -> dict[str, list[str]]:
    hyperedge_to_entities: dict[str, set[str]] = defaultdict(set)
    for entity_name, records in maybe_nodes.items():
        for record in records:
            hyper_relation = record.get("hyper_relation")
            if not hyper_relation:
                continue
            hyperedge_to_entities[hyper_relation].add(entity_name)
    return {
        hyperedge_name: sorted(entity_names)
        for hyperedge_name, entity_names in hyperedge_to_entities.items()
        if entity_names
    }


async def _assign_argument_roles_for_hyperedges(
    *,
    hyperedge_entity_map: dict[str, list[str]],
    support_text: str,
    global_config: dict,
    batch_size: int = 12,
) -> dict[str, dict]:
    if not hyperedge_entity_map:
        return {}

    use_llm_func: callable = global_config["llm_model_func"]
    hyperedge_names = list(hyperedge_entity_map.keys())
    results: dict[str, dict] = {}

    for batch_start in range(0, len(hyperedge_names), max(batch_size, 1)):
        batch_names = hyperedge_names[batch_start : batch_start + max(batch_size, 1)]
        rendered = []
        for idx, hyperedge_name in enumerate(batch_names, start=1):
            entity_list = ", ".join(hyperedge_entity_map.get(hyperedge_name, []))
            rendered.append(
                "\n".join(
                    [
                        f"HYPEREDGE_ID: H{idx}",
                        f"HYPEREDGE_TEXT: {_strip_hyperedge_prefix(hyperedge_name)}",
                        f"CANDIDATE_ENTITIES: {entity_list}",
                    ]
                )
            )
        prompt = f"""You are assigning semantic roles to candidate entities for each event-like hyperedge.

Use the support text and the hyperedge sentence to decide the role each candidate entity plays in that hyperedge.
Role style examples: Subject, Object, Condition, Cause, Mechanism, Result, Metric, Time, Location, Material, Actor, Target, Device, Measurement, Attribute, Component, Method, Finding.
You may choose any concise role label if needed. Prefer the most specific role supported by the text.
Do not use generic roles such as Participant unless the entity is truly only a generic participant.

Return a JSON array only. Each item must contain:
- hyperedge_id: string
- fact_core: concise factual sentence for that hyperedge
- arguments: array of objects with keys entity and role

Rules:
1. Only use candidate entities listed for that hyperedge.
2. Keep the original entity wording.
3. Do not invent new entities.
4. Do not collapse all entities to the same generic role when their semantic functions differ.
5. If an entity's role truly cannot be determined from the text, use Unrecognized.
6. Return one item for every hyperedge_id.

Support text:
{support_text}

Hyperedges:
{chr(10).join(rendered)}
"""
        try:
            response = await use_llm_func(prompt)
            parsed = json.loads(_extract_json_block(response))
        except Exception as exc:
            logger.warning(f"Argument role assignment failed for hyperedge batch, falling back to defaults: {exc}")
            parsed = []

        parsed_by_id = {}
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                parsed_by_id[str(item.get("hyperedge_id", "")).strip()] = item

        for idx, hyperedge_name in enumerate(batch_names, start=1):
            payload = parsed_by_id.get(f"H{idx}", {})
            fact_core = clean_str(str(payload.get("fact_core", "")).strip()) or _strip_hyperedge_prefix(hyperedge_name)
            candidate_entities = list(hyperedge_entity_map.get(hyperedge_name, []))
            candidate_lookup = {
                _canonical_entity_key(entity_name): entity_name
                for entity_name in candidate_entities
                if _canonical_entity_key(entity_name)
            }
            normalized_arguments = []
            for item in _normalize_argument_roles(payload.get("arguments", [])):
                canonical_entity = candidate_lookup.get(_canonical_entity_key(item.get("entity")))
                if canonical_entity:
                    normalized_arguments.append({"entity": canonical_entity, "role": item["role"]})
            normalized_arguments = _merge_argument_roles(normalized_arguments)
            if not normalized_arguments:
                normalized_arguments = [
                    {"entity": entity_name, "role": "Unrecognized"}
                    for entity_name in candidate_entities
                ]
            results[hyperedge_name] = {
                "fact_core": fact_core,
                "arguments": normalized_arguments,
            }

    return results


def _apply_argument_roles_to_records(
    maybe_nodes: dict[str, list[dict]],
    maybe_edges: dict[str, list[dict]],
    hyperedge_roles: dict[str, dict],
) -> None:
    if not hyperedge_roles:
        return

    role_lookup = {
        hyperedge_name: {
            item["entity"]: item["role"]
            for item in payload.get("arguments", [])
        }
        for hyperedge_name, payload in hyperedge_roles.items()
    }

    for hyperedge_name, payload in hyperedge_roles.items():
        for edge_record in maybe_edges.get(hyperedge_name, []):
            edge_record["fact_core"] = payload.get("fact_core", _strip_hyperedge_prefix(hyperedge_name))
            edge_record["argument_roles"] = payload.get("arguments", [])

    for entity_name, records in maybe_nodes.items():
        for record in records:
            hyper_relation = record.get("hyper_relation")
            if not hyper_relation:
                continue
            role = role_lookup.get(hyper_relation, {}).get(entity_name)
            if role:
                record["argument_role"] = role

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    existing_argument_roles: list[dict] = []
    existing_fact_cores: list[str] = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )
        existing_argument_roles = _load_argument_roles(
            already_hyperedge.get("argument_roles_json")
        )
        existing_fact_cores = split_string_by_multi_markers(
            already_hyperedge.get("fact_core", ""), [GRAPH_FIELD_SEP]
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    argument_roles = _merge_argument_roles(
        existing_argument_roles,
        *[dp.get("argument_roles", []) for dp in nodes_data],
    )
    fact_cores = sorted(
        {
            normalized
            for value in (
                existing_fact_cores
                + [dp.get("fact_core", "") for dp in nodes_data]
                + [_strip_hyperedge_prefix(hyperedge_name)]
            )
            for normalized in [_normalize_fact_core(value)]
            if normalized
        }
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
        hyperedge_level=0,
        fact_core=GRAPH_FIELD_SEP.join(fact_cores),
        argument_roles_json=_dump_argument_roles(argument_roles),
        argument_role_count=len(argument_roles),
    )
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    node_data["argument_roles"] = argument_roles
    node_data["embedding_content"] = _format_hyperedge_embedding_content(
        hyperedge_name=hyperedge_name,
        fact_core=fact_cores[0] if fact_cores else None,
        argument_roles=argument_roles,
    )
    return node_data






async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data




async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        argument_role = _normalize_argument_role(node.get("argument_role"))
        
        already_weights = []
        already_source_ids = []
        existing_argument_roles: list[dict] = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            existing_argument_roles = _load_argument_roles(
                already_edge.get("argument_roles_json")
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )
        merged_argument_roles = _merge_argument_roles(
            existing_argument_roles,
            [{"entity": entity_name, "role": argument_role}],
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
                role="link",
                argument_role=_primary_argument_role(merged_argument_roles),
                argument_roles_json=_dump_argument_roles(merged_argument_roles),
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
            argument_role=_primary_argument_role(merged_argument_roles),
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]



    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break


        # seperate llm response into records (hyper-relation + entities)
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            # if record is hyper-relation
            # dict(
            # hyper_relation="<hyperedge>"+knowledge_fragment,
            # weight=weight,
            # source_id=edge_source_id,)
            if_relation = await _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                # start processing a new hyper-relation
                now_hyper_relation = if_relation["hyper_relation"]


            # if record is entity
            # dict(
            # entity_name=entity_name,
            # entity_type=entity_type,
            # description=entity_description,
            # weight=weight,
            # hyper_relation=hyper_relation,
            # source_id=entity_source_id,
            # )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
        hyperedge_entity_map = _build_hyperedge_entity_map_from_records(maybe_nodes)
        if hyperedge_entity_map:
            hyperedge_roles = await _assign_argument_roles_for_hyperedges(
                hyperedge_entity_map=hyperedge_entity_map,
                support_text=content,
                global_config=global_config,
                batch_size=int(global_config.get("role_assignment_batch_size", 12)),
            )
            _apply_argument_roles_to_records(
                maybe_nodes=maybe_nodes,
                maybe_edges=maybe_edges,
                hyperedge_roles=hyperedge_roles,
            )
        # number of chunks processed    
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # # without concurrency limit
    # results = []
    # for result in tqdm_async(
    #     asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
    #     total=len(ordered_chunks),
    #     desc="Extracting entities from chunks",
    #     unit="chunk",
    # ):
    #     results.append(await result)

    ################
    # concurrency limit
    max_concurrency = global_config["max_concurrency"]
    # max_concurrency = 4

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_single_content_with_semaphore(chunk_key_dp):
        async with semaphore:
            return await _process_single_content(chunk_key_dp)

    tasks = [_process_single_content_with_semaphore(c) for c in ordered_chunks]
    results = []
    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)
    #################
    

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # for a given hyperedge id ("<hyperedge>"+knowledge_fragment), 
                # if it comes from multiple source chunks or if it is already in the graph, 
                # sum all its weights and concat all source chunck ids 
                _merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # for a given entity id (entity_name), 
                # if it comes from multiple knowledge segments or if it is already in the graph, 
                # only keep the most frequent entity_type,
                # concat all descriptions, concat all source chunck ids 
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # for a given entity id - hyperedge id pair, 
                # if it comes from multiple knowledge segments or if it is already in the graph, 
                # sum all weights, concat all source chunck ids 
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            # id in vdb = md5 hash the hyperedge name then add a prefix 
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp.get("embedding_content", dp["hyperedge_name"]), # embedding content
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            # id in vdb = md5 hash the entity name then add a prefix 
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"], # embedding content
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst



def _strip_hyperedge_prefix(hyperedge_name: str) -> str:
    return re.sub(r"^<hyperedge[^>]*>", "", hyperedge_name).strip()


def _extract_tagged_text(content: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in content and end_tag in content:
        return content.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return content.strip()


def _extract_json_block(content: str) -> str:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _get_hyperedge_level(node_data: dict | None) -> int:
    if not node_data:
        return 0
    try:
        return int(node_data.get("hyperedge_level", 0))
    except (TypeError, ValueError):
        return 0


async def _get_absorbed_hyperedges(
    knowledge_graph_inst: BaseGraphStorage,
) -> set[str]:
    absorbed_hyperedges = set()
    for source_node_id, target_node_id, edge_data in knowledge_graph_inst.iter_edges():
        if edge_data.get("role") != "contains_hyperedge":
            continue
        source_data = await knowledge_graph_inst.get_node(source_node_id)
        target_data = await knowledge_graph_inst.get_node(target_node_id)
        if not source_data or not target_data:
            continue
        if source_data.get("role") != "hyperedge" or target_data.get("role") != "hyperedge":
            continue

        source_level = _get_hyperedge_level(source_data)
        target_level = _get_hyperedge_level(target_data)
        if source_level > target_level:
            absorbed_hyperedges.add(target_node_id)
        elif target_level > source_level:
            absorbed_hyperedges.add(source_node_id)
    return absorbed_hyperedges


async def _get_higher_order_input_pool(
    knowledge_graph_inst: BaseGraphStorage,
    level: int,
) -> list[str]:
    if level <= 0:
        return []

    prev_level_hyperedges = [
        node_id
        for node_id, node_data in knowledge_graph_inst.iter_hyperedge_nodes()
        if _get_hyperedge_level(node_data) == level - 1
    ]
    return sorted(set(prev_level_hyperedges))


def _collect_member_source_ids(member_node_data: dict | None) -> set[str]:
    if not member_node_data:
        return set()

    if member_node_data.get("member_source_ids"):
        return set(
            split_string_by_multi_markers(
                member_node_data["member_source_ids"], [GRAPH_FIELD_SEP]
            )
        )

    source_id = member_node_data.get("source_id", "")
    if not source_id:
        return set()
    return set(split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP]))


async def _build_higher_order_candidate_clusters(
    knowledge_graph_inst: BaseGraphStorage,
    input_hyperedges: list[str],
    hyperedge_vdb: BaseVectorStorage,
    params: dict,
) -> list[list[str]]:
    if hyperedge_vdb is None:
        logger.warning("Hyperedge vector DB is not initialized, skipping higher-order clustering.")
        return []

    min_cluster_size = max(int(params.get("min_cluster_size", 2)), 2)
    max_cluster_size_raw = params.get("max_cluster_size", 100)
    max_cluster_size = (
        None
        if max_cluster_size_raw is None
        else max(int(max_cluster_size_raw), min_cluster_size)
    )
    max_summary_clusters_raw = params.get("max_summary_clusters")
    if max_summary_clusters_raw is None:
        max_summary_clusters = None
    else:
        try:
            max_summary_clusters = int(max_summary_clusters_raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid max_summary_clusters={max_summary_clusters_raw}, disabling cluster cap."
            )
            max_summary_clusters = None
        else:
            if max_summary_clusters <= 0:
                max_summary_clusters = None

    names: list[str] = []
    vectors: list[np.ndarray] = []
    for hyperedge_name in input_hyperedges:
        hyperedge_id = compute_mdhash_id(hyperedge_name, prefix="rel-")
        vector = await hyperedge_vdb.get_vector_by_id(hyperedge_id)
        if vector is None:
            continue
        names.append(hyperedge_name)
        vectors.append(np.asarray(vector, dtype=float))

    if len(names) < min_cluster_size:
        return []

    strategy = str(params.get("cluster_strategy", "semantic_hdbscan")).strip().lower()
    if strategy == "mixed_graph_community":
        return await _build_higher_order_candidate_clusters_mixed_graph(
            knowledge_graph_inst=knowledge_graph_inst,
            names=names,
            vectors=vectors,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            max_summary_clusters=max_summary_clusters,
            params=params,
        )
    if strategy == "semantic_hnsw_community":
        return await _build_higher_order_candidate_clusters_semantic_hnsw(
            names=names,
            vectors=vectors,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            max_summary_clusters=max_summary_clusters,
            params=params,
        )
    return _build_higher_order_candidate_clusters_hdbscan(
        names=names,
        vectors=vectors,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_summary_clusters=max_summary_clusters,
        params=params,
    )


def _build_higher_order_candidate_clusters_hdbscan(
    names: list[str],
    vectors: list[np.ndarray],
    min_cluster_size: int,
    max_cluster_size: int | None,
    max_summary_clusters: int | None,
    params: dict,
) -> list[list[str]]:
    min_samples_raw = params.get("min_samples", 1)
    min_samples = None if min_samples_raw is None else max(int(min_samples_raw), 1)
    cluster_metric = params.get("cluster_metric", "cosine")
    cluster_selection_epsilon = float(params.get("cluster_selection_epsilon", 0.0))
    matrix = np.vstack(vectors).astype(np.float32)
    if params.get("semantic_pca_dim") is not None:
        matrix = _project_matrix_for_hnsw_search(
            matrix,
            pca_dim=params.get("semantic_pca_dim"),
        )
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=cluster_metric,
        copy=True,
    )
    labels = clusterer.fit_predict(matrix)

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        grouped_indices[int(label)].append(idx)

    def _truncate_cluster_members(indices: list[int]) -> list[int]:
        if max_cluster_size is None or len(indices) <= max_cluster_size:
            return indices
        cluster_vectors = matrix[indices]
        centroid = np.mean(cluster_vectors, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return indices[:max_cluster_size]
        centroid = centroid / centroid_norm
        scored_indices = []
        for idx in indices:
            vector = matrix[idx]
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0:
                score = -1.0
            else:
                score = float(np.dot(vector / vector_norm, centroid))
            scored_indices.append((score, names[idx], idx))
        scored_indices.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, _, idx in scored_indices[:max_cluster_size]]

    clusters: list[list[str]] = []
    for _, indices in sorted(grouped_indices.items(), key=lambda item: (-len(item[1]), item[0])):
        capped_indices = _truncate_cluster_members(indices)
        member_names = sorted(names[idx] for idx in capped_indices)
        if len(member_names) < min_cluster_size:
            continue
        clusters.append(member_names)
        if max_summary_clusters is not None and len(clusters) >= max_summary_clusters:
            break

    return clusters


async def _build_higher_order_candidate_clusters_mixed_graph(
    knowledge_graph_inst: BaseGraphStorage,
    names: list[str],
    vectors: list[np.ndarray],
    min_cluster_size: int,
    max_cluster_size: int | None,
    max_summary_clusters: int | None,
    params: dict,
) -> list[list[str]]:
    vector_map = {
        name: _normalize_vector(vector)
        for name, vector in zip(names, vectors)
    }
    hyperedge_entities = await _collect_hyperedge_entities(
        knowledge_graph_inst=knowledge_graph_inst,
        hyperedge_names=names,
    )
    topology_graph = _build_hyperedge_topology_graph(hyperedge_entities)
    candidate_map = _build_mixed_graph_candidate_map(
        names=names,
        vector_map=vector_map,
        topology_graph=topology_graph,
        params=params,
    )
    sparse_graph = _build_sparse_mixed_similarity_graph(
        candidate_map=candidate_map,
        vector_map=vector_map,
        hyperedge_entities=hyperedge_entities,
        topology_graph=topology_graph,
        params=params,
    )
    return _extract_clusters_from_sparse_graph(
        sparse_graph=sparse_graph,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_summary_clusters=max_summary_clusters,
    )


async def _build_higher_order_candidate_clusters_semantic_hnsw(
    names: list[str],
    vectors: list[np.ndarray],
    min_cluster_size: int,
    max_cluster_size: int | None,
    max_summary_clusters: int | None,
    params: dict,
) -> list[list[str]]:
    vector_map = {
        name: _normalize_vector(vector)
        for name, vector in zip(names, vectors)
    }
    semantic_top_k = max(int(params.get("semantic_top_k", 20)), 1)
    ef_search = max(int(params.get("semantic_ef_search", semantic_top_k * 4)), semantic_top_k + 1)
    hnsw_m = max(int(params.get("semantic_hnsw_m", 16)), 4)
    semantic_threshold = float(
        params.get(
            "semantic_similarity_threshold",
            params.get("mixed_similarity_threshold", 0.5),
        )
    )

    candidate_map = _build_semantic_candidate_map_hnsw(
        names=names,
        vector_map=vector_map,
        top_k=semantic_top_k,
        ef_search=ef_search,
        m=hnsw_m,
        pca_dim=params.get("semantic_pca_dim"),
    )

    sparse_graph = nx.Graph()
    sparse_graph.add_nodes_from(names)
    processed_pairs: set[tuple[str, str]] = set()
    for left_name, neighbors in candidate_map.items():
        for right_name in neighbors:
            if left_name == right_name:
                continue
            pair = tuple(sorted((left_name, right_name)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            semantic_sim = float(np.dot(vector_map[left_name], vector_map[right_name]))
            if semantic_sim < semantic_threshold:
                continue
            sparse_graph.add_edge(
                left_name,
                right_name,
                mixed_similarity=semantic_sim,
                semantic_similarity=semantic_sim,
                topology_similarity=0.0,
            )

    return _extract_clusters_from_sparse_graph(
        sparse_graph=sparse_graph,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_summary_clusters=max_summary_clusters,
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(normalized)
    if norm <= 1e-12:
        return normalized
    return normalized / norm


def _normalize_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return normalized / norms


def _project_matrix_for_hnsw_search(
    matrix: np.ndarray,
    pca_dim: int | None,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return matrix.astype(np.float32, copy=False)

    if pca_dim is None:
        return matrix.astype(np.float32, copy=False)

    try:
        requested_dim = int(pca_dim)
    except (TypeError, ValueError):
        return matrix.astype(np.float32, copy=False)

    if requested_dim <= 0:
        return matrix.astype(np.float32, copy=False)

    max_components = min(matrix.shape[0] - 1, matrix.shape[1])
    if max_components <= 1:
        return matrix.astype(np.float32, copy=False)

    target_dim = min(requested_dim, max_components)
    if target_dim >= matrix.shape[1]:
        return matrix.astype(np.float32, copy=False)

    try:
        reduced = PCA(
            n_components=target_dim,
            svd_solver="randomized",
            random_state=0,
        ).fit_transform(matrix)
        logger.info(
            f"Reduced semantic search vectors with PCA from dim={matrix.shape[1]} to dim={target_dim}."
        )
        return _normalize_matrix_rows(reduced)
    except Exception as exc:  # pragma: no cover - runtime fallback
        logger.warning(f"PCA projection failed before HNSW build, using original vectors: {exc}")
        return matrix.astype(np.float32, copy=False)


async def _collect_hyperedge_entities(
    knowledge_graph_inst: BaseGraphStorage,
    hyperedge_names: list[str],
) -> dict[str, set[str]]:
    hyperedge_set = set(hyperedge_names)
    result = {name: set() for name in hyperedge_names}
    for source_node_id, target_node_id, edge_data in knowledge_graph_inst.iter_edges():
        if edge_data.get("role") != "link":
            continue
        source_data = await knowledge_graph_inst.get_node(source_node_id)
        target_data = await knowledge_graph_inst.get_node(target_node_id)
        if not source_data or not target_data:
            continue
        if (
            source_data.get("role") == "hyperedge"
            and target_data.get("role") == "entity"
            and source_node_id in hyperedge_set
        ):
            result[source_node_id].add(target_node_id)
        elif (
            target_data.get("role") == "hyperedge"
            and source_data.get("role") == "entity"
            and target_node_id in hyperedge_set
        ):
            result[target_node_id].add(source_node_id)
    return result


async def _collect_hyperedge_argument_roles(
    knowledge_graph_inst: BaseGraphStorage,
    hyperedge_names: list[str],
) -> dict[str, list[dict]]:
    hyperedge_set = set(hyperedge_names)
    result = {name: [] for name in hyperedge_names}
    for source_node_id, target_node_id, edge_data in knowledge_graph_inst.iter_edges():
        if edge_data.get("role") != "link":
            continue
        source_data = await knowledge_graph_inst.get_node(source_node_id)
        target_data = await knowledge_graph_inst.get_node(target_node_id)
        if not source_data or not target_data:
            continue
        if (
            source_data.get("role") == "hyperedge"
            and target_data.get("role") == "entity"
            and source_node_id in hyperedge_set
        ):
            result[source_node_id].extend(
                _load_argument_roles(edge_data.get("argument_roles_json"))
                or [{"entity": target_node_id, "role": edge_data.get("argument_role", "Unrecognized")}]
            )
        elif (
            target_data.get("role") == "hyperedge"
            and source_data.get("role") == "entity"
            and target_node_id in hyperedge_set
        ):
            result[target_node_id].extend(
                _load_argument_roles(edge_data.get("argument_roles_json"))
                or [{"entity": source_node_id, "role": edge_data.get("argument_role", "Unrecognized")}]
            )
    return {
        hyperedge_name: _normalize_argument_roles(argument_roles)
        for hyperedge_name, argument_roles in result.items()
    }


def _build_hyperedge_topology_graph(
    hyperedge_entities: dict[str, set[str]],
) -> nx.Graph:
    topology_graph = nx.Graph()
    topology_graph.add_nodes_from(hyperedge_entities.keys())

    entity_to_hyperedges: dict[str, list[str]] = defaultdict(list)
    for hyperedge_name, entities in hyperedge_entities.items():
        for entity_name in entities:
            entity_to_hyperedges[entity_name].append(hyperedge_name)

    for member_names in entity_to_hyperedges.values():
        unique_names = sorted(set(member_names))
        for left_idx, left_name in enumerate(unique_names):
            for right_name in unique_names[left_idx + 1 :]:
                if topology_graph.has_edge(left_name, right_name):
                    topology_graph[left_name][right_name]["shared_entities"] += 1
                else:
                    topology_graph.add_edge(left_name, right_name, shared_entities=1)
    return topology_graph


def _build_mixed_graph_candidate_map(
    names: list[str],
    vector_map: dict[str, np.ndarray],
    topology_graph: nx.Graph,
    params: dict,
) -> dict[str, set[str]]:
    semantic_top_k = max(int(params.get("semantic_top_k", 20)), 1)
    topology_hops = max(int(params.get("topology_hops", 2)), 1)
    candidate_map = {name: set() for name in names}

    semantic_neighbors = _build_semantic_candidate_map_hnsw(
        names=names,
        vector_map=vector_map,
        top_k=semantic_top_k,
        ef_search=max(int(params.get("semantic_ef_search", semantic_top_k * 4)), semantic_top_k + 1),
        m=max(int(params.get("semantic_hnsw_m", 16)), 4),
        pca_dim=params.get("semantic_pca_dim"),
    )
    for hyperedge_name, neighbors in semantic_neighbors.items():
        candidate_map[hyperedge_name].update(neighbors)

    for hyperedge_name in names:
        if hyperedge_name not in topology_graph:
            continue
        path_lengths = nx.single_source_shortest_path_length(
            topology_graph,
            hyperedge_name,
            cutoff=topology_hops,
        )
        for neighbor_name, distance in path_lengths.items():
            if neighbor_name == hyperedge_name or distance <= 0:
                continue
            candidate_map[hyperedge_name].add(neighbor_name)

    for left_name, neighbors in candidate_map.items():
        for right_name in list(neighbors):
            if right_name == left_name:
                continue
            candidate_map.setdefault(right_name, set()).add(left_name)
    return candidate_map


def _build_semantic_candidate_map_hnsw(
    names: list[str],
    vector_map: dict[str, np.ndarray],
    top_k: int,
    ef_search: int,
    m: int,
    pca_dim: int | None = None,
) -> dict[str, set[str]]:
    candidate_map = {name: set() for name in names}
    if len(names) <= 1:
        return candidate_map

    matrix = np.vstack([vector_map[name] for name in names]).astype(np.float32)
    search_matrix = _project_matrix_for_hnsw_search(matrix, pca_dim=pca_dim)
    try:
        import hnswlib

        index = hnswlib.Index(space="cosine", dim=search_matrix.shape[1])
        index.init_index(max_elements=len(names), ef_construction=max(ef_search, 100), M=m)
        index.add_items(search_matrix, np.arange(len(names)))
        index.set_ef(max(ef_search, top_k + 1))
        labels, _ = index.knn_query(search_matrix, k=min(top_k + 1, len(names)))
        for row_idx, neighbor_row in enumerate(labels):
            source_name = names[row_idx]
            for neighbor_idx in neighbor_row:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx == row_idx:
                    continue
                candidate_map[source_name].add(names[neighbor_idx])
        return candidate_map
    except Exception as exc:  # pragma: no cover - runtime fallback
        logger.warning(f"HNSW candidate build failed, fallback to exact cosine KNN: {exc}")

    similarity = search_matrix @ search_matrix.T
    for row_idx, source_name in enumerate(names):
        order = np.argsort(similarity[row_idx])[::-1]
        for neighbor_idx in order:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx == row_idx:
                continue
            candidate_map[source_name].add(names[neighbor_idx])
            if len(candidate_map[source_name]) >= top_k:
                break
    return candidate_map


def _build_sparse_mixed_similarity_graph(
    candidate_map: dict[str, set[str]],
    vector_map: dict[str, np.ndarray],
    hyperedge_entities: dict[str, set[str]],
    topology_graph: nx.Graph,
    params: dict,
) -> nx.Graph:
    semantic_weight = float(params.get("semantic_weight", 0.7))
    topology_weight = float(params.get("topology_weight", 0.3))
    direct_weight = float(params.get("topology_direct_weight", 0.5))
    path_weight = float(params.get("topology_path_weight", 0.5))
    threshold = float(params.get("mixed_similarity_threshold", 0.6))
    topology_hops = max(int(params.get("topology_hops", 2)), 1)
    topology_path_decay = float(params.get("topology_path_decay", 0.7))

    graph = nx.Graph()
    graph.add_nodes_from(candidate_map.keys())

    processed_pairs: set[tuple[str, str]] = set()
    for left_name, neighbors in candidate_map.items():
        for right_name in neighbors:
            if left_name == right_name:
                continue
            pair = tuple(sorted((left_name, right_name)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            semantic_sim = float(np.dot(vector_map[left_name], vector_map[right_name]))
            topology_sim = _compute_topology_similarity(
                left_name=left_name,
                right_name=right_name,
                hyperedge_entities=hyperedge_entities,
                topology_graph=topology_graph,
                topology_hops=topology_hops,
                topology_path_decay=topology_path_decay,
                direct_weight=direct_weight,
                path_weight=path_weight,
            )
            mixed_sim = semantic_weight * semantic_sim + topology_weight * topology_sim
            if mixed_sim < threshold:
                continue
            graph.add_edge(
                left_name,
                right_name,
                mixed_similarity=float(mixed_sim),
                semantic_similarity=float(semantic_sim),
                topology_similarity=float(topology_sim),
            )
    return graph


def _compute_topology_similarity(
    left_name: str,
    right_name: str,
    hyperedge_entities: dict[str, set[str]],
    topology_graph: nx.Graph,
    topology_hops: int,
    topology_path_decay: float,
    direct_weight: float,
    path_weight: float,
) -> float:
    left_entities = hyperedge_entities.get(left_name, set())
    right_entities = hyperedge_entities.get(right_name, set())
    direct_overlap = _jaccard_similarity(left_entities, right_entities)

    path_score = 0.0
    if topology_graph.has_node(left_name) and topology_graph.has_node(right_name):
        try:
            distance = nx.shortest_path_length(
                topology_graph,
                left_name,
                right_name,
            )
        except nx.NetworkXNoPath:
            distance = None
        if distance is not None and 1 <= distance <= topology_hops:
            path_score = math.exp(-topology_path_decay * (distance - 1))

    total_weight = max(direct_weight + path_weight, 1e-12)
    return float((direct_weight * direct_overlap + path_weight * path_score) / total_weight)


def _extract_clusters_from_sparse_graph(
    sparse_graph: nx.Graph,
    min_cluster_size: int,
    max_cluster_size: int | None,
    max_summary_clusters: int | None,
) -> list[list[str]]:
    communities: list[list[str]] = []
    if sparse_graph.number_of_edges() == 0:
        communities = [
            [node_name]
            for node_name in sorted(sparse_graph.nodes())
        ]
    else:
        try:
            raw_communities = nx.community.greedy_modularity_communities(
                sparse_graph,
                weight="mixed_similarity",
            )
            communities = [
                sorted(community)
                for community in raw_communities
            ]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Greedy modularity communities failed, fallback to connected components: {exc}")
            communities = [
                sorted(component)
                for component in nx.connected_components(sparse_graph)
            ]
    communities = [
        community
        for community in communities
        if len(community) >= min_cluster_size
    ]
    communities.sort(key=lambda community: (-len(community), community))
    clusters: list[list[str]] = []
    for component in communities:
        truncated_component = _truncate_component_by_weighted_degree(
            sparse_graph=sparse_graph,
            component=component,
            max_cluster_size=max_cluster_size,
        )
        if len(truncated_component) < min_cluster_size:
            continue
        clusters.append(truncated_component)
        if max_summary_clusters is not None and len(clusters) >= max_summary_clusters:
            break
    return clusters


def _truncate_component_by_weighted_degree(
    sparse_graph: nx.Graph,
    component: list[str],
    max_cluster_size: int | None,
) -> list[str]:
    if max_cluster_size is None or len(component) <= max_cluster_size:
        return sorted(component)
    scored_nodes: list[tuple[float, str]] = []
    for node_name in component:
        weighted_degree = 0.0
        for _, _, edge_data in sparse_graph.edges(node_name, data=True):
            weighted_degree += float(edge_data.get("mixed_similarity", 0.0))
        scored_nodes.append((weighted_degree, node_name))
    scored_nodes.sort(key=lambda item: (-item[0], item[1]))
    kept_nodes = [name for _, name in scored_nodes[:max_cluster_size]]
    return sorted(kept_nodes)


async def _generate_higher_order_summary(
    member_hyperedges: list[str],
    global_config: dict,
    level: int,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    member_descriptions = []
    for idx, hyperedge_name in enumerate(member_hyperedges, start=1):
        member_descriptions.append(f"{idx}. {_strip_hyperedge_prefix(hyperedge_name)}")

    prompt = f"""-Goal-
You are given several semantically overlapping member hyperedge descriptions.
Write exactly one higher-order hyperedge description for level {level}.

-Requirements-
1. Output exactly one factual sentence, not a title or a topic label.
2. Summarize only the shared semantics across the input descriptions.
3. Preserve important conditions, participants, and constraints when they are consistently supported.
4. Do not introduce new entities, events, causes, or conclusions that are absent from the inputs.
5. Keep the sentence concise but specific.

Output format:
<summary>
your sentence here
</summary>

Low-level hyperedge descriptions:
{chr(10).join(member_descriptions)}
"""
    summary_result = await use_llm_func(prompt)
    summary_text = clean_str(_extract_tagged_text(summary_result, "summary"))
    return summary_text


async def _generate_higher_order_batch_outputs(
    cluster_payloads: list[dict],
    global_config: dict,
    level: int,
) -> dict[str, dict]:
    if not cluster_payloads:
        return {}

    use_llm_func: callable = global_config["llm_model_func"]
    rendered_clusters = []
    for payload in cluster_payloads:
        descriptions = [
            f"{idx}. {_strip_hyperedge_prefix(hyperedge_name)}"
            for idx, hyperedge_name in enumerate(payload["member_hyperedges"], start=1)
        ]
        rendered_clusters.append(
            "\n".join(
                [
                    f"CLUSTER_ID: {payload['cluster_id']}",
                    "MEMBER_HYPEREDGES:",
                    *descriptions,
                ]
            )
        )

    prompt = f"""You are given multiple clusters of semantically overlapping low-level hyperedge descriptions.
For each cluster, produce exactly one higher-order hyperedge summary and the entities explicitly supported by that summary, with one semantic role for each entity.

Return a JSON array only. Each item must have:
- cluster_id: string
- summary: one factual sentence for level {level}
- entities: array of objects with keys name, entity_type, description, role, weight

Requirements:
1. One output item per input cluster_id.
2. Each summary must capture only the shared meaning of that cluster.
3. Do not mix information across clusters.
4. Each entity must be grounded in the summary sentence for its own cluster.
5. Assign each entity one concise semantic role such as Subject, Condition, Cause, Mechanism, Result, Metric, Material, Actor, Target, Device, Measurement, Attribute, Component, Method, or Finding.
6. Keep entities concise and factual.

Input clusters:
{chr(10).join(rendered_clusters)}
"""
    response = await use_llm_func(prompt)
    try:
        parsed = json.loads(_extract_json_block(response))
    except Exception as exc:
        logger.warning(f"Batch higher-order generation failed to parse JSON, fallback to per-cluster generation: {exc}")
        return {}

    result = {}
    if not isinstance(parsed, list):
        return result
    for item in parsed:
        if not isinstance(item, dict):
            continue
        cluster_id = str(item.get("cluster_id", "")).strip()
        summary = clean_str(str(item.get("summary", "")).strip())
        entities = item.get("entities", [])
        if not cluster_id or not summary or not isinstance(entities, list):
            continue
        result[cluster_id] = {
            "summary": summary,
            "entities": entities,
        }
    return result


async def _upsert_entities_for_higher_order_from_records(
    summary_text: str,
    chunk_key: str,
    hyperedge_name: str,
    entity_records: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> tuple[list[str], list[dict]]:
    maybe_nodes = defaultdict(list)
    summary_text_upper = summary_text.upper()

    for record in entity_records:
        if not isinstance(record, dict):
            continue
        entity_name = clean_str(str(record.get("name", "")).upper())
        if not entity_name:
            continue
        entity_surface = entity_name.replace('"', "").strip()
        if entity_surface and entity_surface not in summary_text_upper:
            continue
        entity_type = clean_str(str(record.get("entity_type", "UNKNOWN")).upper()) or "UNKNOWN"
        description = clean_str(str(record.get("description", "")).strip()) or entity_surface
        weight_raw = record.get("weight", 50.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            weight = 50.0
        argument_role = _normalize_argument_role(record.get("role"))
        maybe_nodes[entity_name].append(
            dict(
                entity_name=entity_name,
                entity_type=entity_type,
                description=description,
                weight=weight,
                hyper_relation=hyperedge_name,
                source_id=chunk_key,
                argument_role=argument_role,
            )
        )

    if not maybe_nodes:
        return [], []

    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc=f"Extracting higher-order entities for {hyperedge_name}",
        unit="entity",
    ):
        all_entities_data.append(await result)

    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc=f"Linking higher-order entities for {hyperedge_name}",
        unit="entity",
    ):
        await result

    return [dp["entity_name"] for dp in all_entities_data], all_entities_data


async def _extract_entities_for_fixed_hyperedge(
    summary_text: str,
    chunk_key: str,
    hyperedge_name: str,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> list[str]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    examples = examples.format(**example_context_base)
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(**context_base, input_text=summary_text)

    final_result = await use_llm_func(hint_prompt)
    history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
    for now_glean_index in range(entity_extract_max_gleaning):
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
        final_result += glean_result
        if now_glean_index == entity_extract_max_gleaning - 1:
            break
        if_loop_result: str = await use_llm_func(
            if_loop_prompt, history_messages=history
        )
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result != "yes":
            break

    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    maybe_nodes = defaultdict(list)
    summary_text_upper = summary_text.upper()
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        if_entities = await _handle_single_entity_extraction(
            record_attributes, chunk_key, hyperedge_name
        )
        if if_entities is not None:
            entity_surface = if_entities["entity_name"].replace('"', "").strip()
            if entity_surface and entity_surface not in summary_text_upper:
                continue
            maybe_nodes[if_entities["entity_name"]].append(if_entities)

    if not maybe_nodes:
        return []

    role_payload = await _assign_argument_roles_for_hyperedges(
        hyperedge_entity_map={
            hyperedge_name: sorted(maybe_nodes.keys())
        },
        support_text=summary_text,
        global_config=global_config,
        batch_size=1,
    )
    _apply_argument_roles_to_records(
        maybe_nodes=maybe_nodes,
        maybe_edges=defaultdict(list, {hyperedge_name: []}),
        hyperedge_roles=role_payload,
    )

    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc=f"Extracting higher-order entities for {hyperedge_name}",
        unit="entity",
    ):
        all_entities_data.append(await result)

    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc=f"Linking higher-order entities for {hyperedge_name}",
        unit="entity",
    ):
        await result

    if entity_vdb is not None and all_entities_data:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return [dp["entity_name"] for dp in all_entities_data]


async def build_higher_order_hyperedges(
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> BaseGraphStorage:
    params = global_config.get("higher_order_params", {})
    if not params.get("enabled", False):
        logger.info("Higher-order hyperedge construction disabled.")
        return knowledge_graph_inst

    max_level = int(params.get("max_level", 1))
    if max_level < 1:
        logger.info("Higher-order hyperedge max_level < 1, skipping.")
        return knowledge_graph_inst

    existing_higher_nodes = [
        node_id
        for node_id, node_data in knowledge_graph_inst.iter_hyperedge_nodes()
        if int(node_data.get("hyperedge_level", 0)) >= 1
    ]
    if existing_higher_nodes and params.get("skip_if_exists", True):
        logger.info(
            f"Higher-order hyperedges already exist ({len(existing_higher_nodes)} nodes), skipping rebuild."
        )
        return knowledge_graph_inst

    min_cluster_size = max(int(params.get("min_cluster_size", 2)), 2)
    promote_singletons = bool(params.get("promote_singletons", False))
    batch_size = max(int(params.get("higher_order_batch_size", 1)), 1)
    created_count = 0

    for level in range(1, max_level + 1):
        input_pool = await _get_higher_order_input_pool(
            knowledge_graph_inst=knowledge_graph_inst,
            level=level,
        )
        if len(input_pool) < min_cluster_size:
            logger.info(
                f"Level {level}: input pool has only {len(input_pool)} hyperedge(s), skipping."
            )
            continue

        clusters = await _build_higher_order_candidate_clusters(
            knowledge_graph_inst=knowledge_graph_inst,
            input_hyperedges=input_pool,
            hyperedge_vdb=hyperedge_vdb,
            params=params,
        )
        if promote_singletons:
            clustered_members = {member for cluster in clusters for member in cluster}
            singleton_clusters = [
                [hyperedge_name]
                for hyperedge_name in input_pool
                if hyperedge_name not in clustered_members
            ]
            if singleton_clusters:
                logger.info(
                    f"Level {level}: promoting {len(singleton_clusters)} unclustered hyperedge(s) as singleton higher-order nodes."
                )
                clusters.extend(singleton_clusters)
        if not clusters:
            logger.info(f"Level {level}: no qualified overlapping clusters found.")
            continue

        logger.info(
            f"Level {level}: building {len(clusters)} higher-order hyperedge cluster(s) from {len(input_pool)} candidate hyperedges."
        )

        higher_order_chunks: dict[str, TextChunkSchema] = {}
        higher_order_vdb_data: dict[str, dict] = {}
        higher_order_vdb_vectors: list[dict[str, Any]] = []
        higher_order_entity_vdb_data: dict[str, dict] = {}
        cluster_payloads = [
            {
                "cluster_idx": cluster_idx,
                "cluster_id": compute_mdhash_id(
                    "|".join(sorted(member_hyperedges)),
                    prefix=f"ho-cluster-l{level}-",
                ),
                "member_hyperedges": member_hyperedges,
            }
            for cluster_idx, member_hyperedges in enumerate(clusters, start=1)
        ]
        batch_outputs: dict[str, dict] = {}
        batched_payloads = [
            payload
            for payload in cluster_payloads
            if len(payload["member_hyperedges"]) > 1
        ]
        total_batches = (
            (len(batched_payloads) + batch_size - 1) // batch_size
            if batched_payloads
            else 0
        )
        completed_batches = 0
        completed_clusters = 0
        for batch_start in range(0, len(batched_payloads), batch_size):
            batch = batched_payloads[batch_start : batch_start + batch_size]
            generated = await _generate_higher_order_batch_outputs(
                cluster_payloads=batch,
                global_config=global_config,
                level=level,
            )
            batch_outputs.update(generated)
            completed_batches += 1
            completed_clusters += len(batch)
            logger.info(
                f"Level {level}: completed batch {completed_batches}/{total_batches}; "
                f"completed clusters {completed_clusters}/{len(batched_payloads)}."
            )
        singleton_entity_map = (
            await _collect_hyperedge_entities(
                knowledge_graph_inst=knowledge_graph_inst,
                hyperedge_names=input_pool,
            )
            if any(len(cluster) == 1 for cluster in clusters)
            else {}
        )
        singleton_argument_role_map = (
            await _collect_hyperedge_argument_roles(
                knowledge_graph_inst=knowledge_graph_inst,
                hyperedge_names=input_pool,
            )
            if any(len(cluster) == 1 for cluster in clusters)
            else {}
        )

        for payload in cluster_payloads:
            cluster_idx = payload["cluster_idx"]
            cluster_id = payload["cluster_id"]
            member_hyperedges = payload["member_hyperedges"]
            if len(member_hyperedges) == 1:
                summary_text = _strip_hyperedge_prefix(member_hyperedges[0])
                batch_entity_records = None
            else:
                generated = batch_outputs.get(cluster_id)
                if generated:
                    summary_text = generated["summary"]
                    batch_entity_records = generated.get("entities", [])
                else:
                    summary_text = await _generate_higher_order_summary(
                        member_hyperedges=member_hyperedges,
                        global_config=global_config,
                        level=level,
                    )
                    batch_entity_records = None
            if not summary_text:
                logger.warning(
                    f"Level {level}: empty higher-order summary generated for cluster {cluster_idx}, skipping."
                )
                continue

            high_hyperedge_name = f"<hyperedge L{level}>{summary_text}"
            summary_chunk_id = compute_mdhash_id(
                high_hyperedge_name + cluster_id,
                prefix=f"chunk-ho-l{level}-",
            )
            argument_roles: list[dict] = []

            member_source_ids = set()
            for member_hyperedge in member_hyperedges:
                member_data = await knowledge_graph_inst.get_node(member_hyperedge) or {}
                member_source_ids.update(_collect_member_source_ids(member_data))

            chunk_tokens = len(
                encode_string_by_tiktoken(
                    summary_text,
                    model_name=global_config["tiktoken_model_name"],
                )
            )
            higher_order_chunks[summary_chunk_id] = {
                "tokens": chunk_tokens,
                "content": summary_text,
                "full_doc_id": f"high_order_level_{level}::{cluster_id}",
                "chunk_order_index": cluster_idx - 1,
            }

            await knowledge_graph_inst.upsert_node(
                high_hyperedge_name,
                node_data=dict(
                    role="hyperedge",
                    weight=float(len(member_hyperedges)),
                    source_id=summary_chunk_id,
                    hyperedge_level=level,
                    cluster_id=cluster_id,
                    member_count=len(member_hyperedges),
                    member_source_ids=GRAPH_FIELD_SEP.join(sorted(member_source_ids)),
                    fact_core=summary_text,
                    argument_roles_json="[]",
                    argument_role_count=0,
                ),
            )

            if len(member_hyperedges) == 1:
                generated_entities = sorted(
                    singleton_entity_map.get(member_hyperedges[0], set())
                )
                argument_roles = singleton_argument_role_map.get(member_hyperedges[0], [])
                for entity_name in generated_entities:
                    inherited_roles = [
                        item for item in argument_roles if item["entity"] == entity_name
                    ] or [{"entity": entity_name, "role": "Unrecognized"}]
                    await knowledge_graph_inst.upsert_edge(
                        high_hyperedge_name,
                        entity_name,
                        edge_data=dict(
                            weight=1.0,
                            source_id=summary_chunk_id,
                            role="link",
                            argument_role=_primary_argument_role(inherited_roles),
                            argument_roles_json=_dump_argument_roles(inherited_roles),
                        ),
                    )
            elif batch_entity_records is not None:
                generated_entities, all_entities_data = await _upsert_entities_for_higher_order_from_records(
                    summary_text=summary_text,
                    chunk_key=summary_chunk_id,
                    hyperedge_name=high_hyperedge_name,
                    entity_records=batch_entity_records,
                    knowledge_graph_inst=knowledge_graph_inst,
                    global_config=global_config,
                )
                argument_roles = _normalize_argument_roles(
                    [
                        {
                            "entity": dp["entity_name"],
                            "role": dp.get("argument_role", "Unrecognized"),
                        }
                        for dp in all_entities_data
                    ]
                )
                for dp in all_entities_data:
                    higher_order_entity_vdb_data[
                        compute_mdhash_id(dp["entity_name"], prefix="ent-")
                    ] = {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
            else:
                generated_entities = await _extract_entities_for_fixed_hyperedge(
                    summary_text=summary_text,
                    chunk_key=summary_chunk_id,
                    hyperedge_name=high_hyperedge_name,
                    knowledge_graph_inst=knowledge_graph_inst,
                    entity_vdb=entity_vdb,
                    global_config=global_config,
                )
                argument_roles = [
                    {
                        "entity": entity_name,
                        "role": (
                            (await knowledge_graph_inst.get_edge(high_hyperedge_name, entity_name)) or {}
                        ).get("argument_role", "Unrecognized"),
                    }
                    for entity_name in generated_entities
                ]

            await knowledge_graph_inst.upsert_node(
                high_hyperedge_name,
                node_data=dict(
                    role="hyperedge",
                    weight=float(len(member_hyperedges)),
                    source_id=summary_chunk_id,
                    hyperedge_level=level,
                    cluster_id=cluster_id,
                    member_count=len(member_hyperedges),
                    generated_entities=GRAPH_FIELD_SEP.join(sorted(generated_entities)),
                    member_source_ids=GRAPH_FIELD_SEP.join(sorted(member_source_ids)),
                    fact_core=summary_text,
                    argument_roles_json=_dump_argument_roles(argument_roles),
                    argument_role_count=len(_normalize_argument_roles(argument_roles)),
                ),
            )

            for member_hyperedge in member_hyperedges:
                await knowledge_graph_inst.upsert_edge(
                    high_hyperedge_name,
                    member_hyperedge,
                    edge_data=dict(
                        weight=1.0,
                        source_id=summary_chunk_id,
                        role="contains_hyperedge",
                    ),
                )

            higher_hyperedge_id = compute_mdhash_id(high_hyperedge_name, prefix="rel-")
            if len(member_hyperedges) == 1 and hyperedge_vdb is not None and hasattr(hyperedge_vdb, "_client"):
                member_hyperedge_id = compute_mdhash_id(member_hyperedges[0], prefix="rel-")
                member_vector = await hyperedge_vdb.get_vector_by_id(member_hyperedge_id)
                if member_vector is not None:
                    higher_order_vdb_vectors.append(
                        {
                            "__id__": higher_hyperedge_id,
                            "hyperedge_name": high_hyperedge_name,
                            "__vector__": np.asarray(member_vector, dtype=np.float32),
                        }
                    )
                else:
                    higher_order_vdb_data[higher_hyperedge_id] = {
                        "content": _format_hyperedge_embedding_content(
                            hyperedge_name=high_hyperedge_name,
                            fact_core=summary_text,
                            argument_roles=argument_roles,
                        ),
                        "hyperedge_name": high_hyperedge_name,
                    }
            else:
                higher_order_vdb_data[higher_hyperedge_id] = {
                    "content": _format_hyperedge_embedding_content(
                        hyperedge_name=high_hyperedge_name,
                        fact_core=summary_text,
                        argument_roles=argument_roles,
                    ),
                    "hyperedge_name": high_hyperedge_name,
                }
            created_count += 1

        if text_chunks_db is not None and higher_order_chunks:
            await text_chunks_db.upsert(higher_order_chunks)
        if hyperedge_vdb is not None and higher_order_vdb_data:
            await hyperedge_vdb.upsert(higher_order_vdb_data)
        if hyperedge_vdb is not None and higher_order_vdb_vectors and hasattr(hyperedge_vdb, "_client"):
            hyperedge_vdb._client.upsert(datas=higher_order_vdb_vectors)
        if entity_vdb is not None and higher_order_entity_vdb_data:
            logger.info(
                f"Level {level}: bulk upserting {len(higher_order_entity_vdb_data)} higher-order entity vectors."
            )
            await entity_vdb.upsert(higher_order_entity_vdb_data)

    logger.info(f"Created {created_count} higher-order hyperedges across levels 1-{max_level}.")
    return knowledge_graph_inst



async def _group_synonyms(
    representative: str,
    synonyms: set[str],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):

    synonyms.add(representative)
    logger.debug(f"synonyms: {synonyms}")
    if len(synonyms) == 1:
        # if there is only one synonym, just return the representative
        logger.debug("Skipping synonym group of one")
        return representative

    group_name = "<synonyms>" + GRAPH_FIELD_SEP.join(
        sorted(synonyms)
    )

    await knowledge_graph_inst.upsert_node(
        group_name,
        node_data=dict(
        role = "synonyms",
    ),
    )
    
    for synonym in synonyms:
        await knowledge_graph_inst.upsert_edge(
        group_name,
        synonym,
        edge_data={
            "role": "synonym",
        }
    )

    return group_name

async def merge_synonym_entities(
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
    max_compare_batch_size: int = 20,
    max_shuffle_attempts: int = 3,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )


    components = knowledge_graph_inst.get_similar_components()
    if not components:
        logger.warning("No similar entities found in the knowledge graph.")
        return None
    
    logger.info(f"Found {len(components)} synonym entity components to merge.")
    
    merge_synonym_prompt = PROMPTS["merge_synonym"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        language=language,
    )
    merge_synonym_prompt = merge_synonym_prompt.format(
        **context_base, input_text="{input_text}"
    )
    
    async def _merge_single_synonym_component(component:set[str]):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(entity) for entity in component]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
            return None
        entity_to_desc = {}
        for entity, node_data in zip(component, node_datas):
            entity_desc = node_data.get('description', None)
            if entity_desc is None:
                logger.warning(f"Entity {entity} has no description, skipping.")
                continue
            entity_to_desc[entity] = entity_desc.split(GRAPH_FIELD_SEP)
        

        
        entity_to_group = {entity: {entity} for entity in entity_to_desc.keys()}

        attempt = 0
        while attempt < max_shuffle_attempts:
            merged = False
            current_entities = list(entity_to_group.keys())
            
            # logger.debug(f"current_entities: {current_entities}")
            random.shuffle(current_entities)
            new_entity_to_group = {}
            used = set()
            for i in range(0, len(current_entities), max_compare_batch_size):
                batch_entity_names = current_entities[i:i+max_compare_batch_size]
                # batch = {entity_name: entity_to_data[entity_name] for entity_name in batch_entity_names}

                entity_input_text = ""
                for entity_name in batch_entity_names:
                    entity_input_text += f"Entity name: {entity_name}, descriptions: {entity_to_desc[entity_name]}\n"
                
                hint_prompt = merge_synonym_prompt.format(input_text=entity_input_text)

                # logger.debug(f"Merging synonym entities prompt: {hint_prompt}")

                llm_output = await use_llm_func(hint_prompt)
                # logger.debug(f"Merging synonym entities response: {llm_output}")


                # seperate llm response into records (hyper-relation + entities)
                groups = split_string_by_multi_markers(
                    llm_output,
                    [context_base["record_delimiter"], context_base["completion_delimiter"]],
                )
                # logger.debug(f"groups: {groups}")
                for group in groups:
                    # logger.debug(f"group: {group}")
                    group = re.search(r"\((.*)\)", group)
                    if group is None:
                        # logger.debug(f"regex search failed for group: {group}")
                        continue
                    group = group.group(1)
                    group = split_string_by_multi_markers(
                        group, [context_base["tuple_delimiter"]]
                    )
                    
                    synonyms = set(group)
                    # intersect with batch_entity_names
                    synonyms = synonyms.intersection(set(batch_entity_names))
                    # logger.debug(f"synonyms: {synonyms}")
                    if not synonyms:
                        # logger.debug(f"No synonyms found in group: {group}")
                        continue
                    if len(synonyms) > 1:
                        merged = True
                    group = list(synonyms)
                    rep = group[0]
                    used.update(synonyms)
                    combined = set()
                    for synonym in synonyms:
                        if synonym in entity_to_group:
                            combined.update(entity_to_group[synonym])
                    new_entity_to_group[rep] = combined
                    # logger.debug(f"add mapping: {rep}: {combined}")
                for entity_name in batch_entity_names:
                    if entity_name not in used:
                        new_entity_to_group[entity_name] = entity_to_group[entity_name]
            entity_to_group = new_entity_to_group
            # logger.debug(f"New entity to group mapping: {entity_to_group}")
            if merged:
                attempt = 0  # restart attempts
            else:
                attempt += 1
            if len(current_entities) <= max_compare_batch_size:
                attempt = max_shuffle_attempts
        return entity_to_group
    
    ################
    # concurrency limit
    max_concurrency = global_config["max_concurrency"]
    # max_concurrency = 4

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _merge_single_synonym_component_with_semaphore(chunk_key_dp):
        async with semaphore:
            return await _merge_single_synonym_component(chunk_key_dp)

    tasks = [_merge_single_synonym_component_with_semaphore(c) for c in components]
    results = []
    for result in tqdm_async(
        asyncio.as_completed(tasks),
        total=len(components),
        desc="Merging synonym components",
        unit="component",
    ):
        results.append(await result)
    #################
    # logger.debug(f"Total Merging synonym entities results: {results}")
    
    entity_to_group = defaultdict(set)

    for result in results:
        if result is None:
            continue
        for k, v in result.items():
            entity_to_group[k].update(v)

            
    logger.info("Inserting synonym edges into storage...")


    all_represents = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # for a given entity id (entity_name), 
                # if it comes from multiple knowledge segments or if it is already in the graph, 
                # only keep the most frequent entity_type,
                # concat all descriptions, concat all source chunck ids 
                # _merge_synonyms_then_upsert(k, v, knowledge_graph_inst, global_config)
                # for k, v in entity_to_group.items()
                _group_synonyms(k, v, knowledge_graph_inst, global_config)
                for k, v in entity_to_group.items()
            ]
        ),
        total=len(entity_to_group),
        desc="Merge synonyms",
        unit="synonym group",
    ):
        all_represents.append(await result)

    return knowledge_graph_inst


class Hypergraph:
    def __init__(
            self, 
            knowledge_graph_inst, 
            entity_names_vdb, 
            entities_vdb, 
            hyperedges_vdb,
            query_param: QueryParam, 
            ):
        self.base = knowledge_graph_inst
        self.entity_names_vdb = entity_names_vdb
        self.entities_vdb = entities_vdb
        self.hyperedges_vdb = hyperedges_vdb
        self.entities = []
        self.hyperedges = []
        self.entity_to_id = {}
        self.hyperedge_to_id = {}
        self.entity_to_hyperedges = []
        self.hyperedge_to_entities = []
        self.tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
        self.ent_desc_embs = []
        self.query_param = query_param

        self.model = None
        if query_param.emb_model == "SBERT":
            self.model = SentenceTransformer('all-MiniLM-L6-v2')



    
    async def he_match(self, queries, he_top_k = 5, he_threshold = 0.6):
        linked_hes = []
        for i, query in enumerate(queries):
            hes = await self.hyperedges_vdb.query(query, top_k=he_top_k, better_than_threshold=he_threshold)
            if not query:
                logger.debug(f"Can not a match for '{query}' in knowledge hypergraph.")
                continue
            raw_he_names = [
                h["hyperedge_name"]
                for h in hes
                if self.get_hyperedge_id(h["hyperedge_name"]) >= 0
            ]
            he_names = await _filter_l0_hyperedge_names(raw_he_names, self.base)
            linked_hes.extend(he_names)
            logger.info(f"Query '{query}' is matched with: ")
            logger.info(', '.join(he_names))
        
        return linked_hes


    async def topic_link(self, key_entities, topic_top_k = 1, topic_threshold = 0.9):
        topic_entities = []
        for i, key in enumerate(key_entities):
            topics = await self.entity_names_vdb.query(key, top_k=topic_top_k, better_than_threshold=topic_threshold)
            if not topics:
                logger.warning(f"Can not link key entity '{key}' with knowledge hypergraph.")
                continue
            # keep name only
            topic_entity_names= [t["entity_name"] for t in topics if self.get_entity_id(t["entity_name"]) >= 0]
            topic_entities.extend(topic_entity_names)
            logger.info(f"Key entity '{key}' is linked to: ")
            logger.info(', '.join(topic_entity_names))
        
        return topic_entities

    async def add_entity(self, entity):
        if entity not in self.entity_to_id:
            entity_id = len(self.entities)
            synonyms = []
            if self.query_param.get_synonyms:
                synonyms = await self.base.get_entity_synonyms(entity)
            synonyms.insert(0, entity)  # Ensure the original entity is included at the beginning
        

            entity_data_list = []
            ent_desc_emb_list = []
            ent_desc_list = []
            
            for synonym in synonyms:
                self.entity_to_id[synonym] = entity_id
                synonym_data = await self.base.get_node(synonym)
                if synonym_data is None:
                    continue

                synonym_description = synonym_data['description']

                entity_data_list.append({
                    "entity_name": synonym,
                    "description": synonym_description
                })
   
                id = compute_mdhash_id(synonym, prefix="ent-") 
                entity_desc_emb = await self.entities_vdb.get_vector_by_id(id)
                if entity_desc_emb is None:
                    logger.warning(f"Entity description embedding not found for entity '{synonym}' with id '{id}'")
                    continue
                ent_desc_emb_list.append(entity_desc_emb)
                ent_desc_list.append(synonym + synonym_description)
            
            if self.query_param.emb_model == "SBERT" and self.model is not None and ent_desc_list:
                # Compute embedding for the entire description using SBERT
                ent_desc_emb_list = self.model.encode(
                    ent_desc_list,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                ent_desc_emb_list = [emb.cpu().numpy() for emb in ent_desc_emb_list]
            
            self.entities.append(entity_data_list)
            self.ent_desc_embs.append(ent_desc_emb_list)

            self.entity_to_hyperedges.append([])           

        return self.entity_to_id[entity]
    
    async def add_hyperedge(self, hyperedge):
        if hyperedge not in self.hyperedge_to_id:
            hyperedge_id = len(self.hyperedges)
            self.hyperedges.append(hyperedge)
            self.hyperedge_to_id[hyperedge] = hyperedge_id
            self.hyperedge_to_entities.append([])
            entities = await self.base.get_hyperedge_entities(hyperedge, data = False, get_synonyms = False)
            for entity in entities:
                entity_id = await self.add_entity(entity)
                self.entity_to_hyperedges[entity_id].append(hyperedge_id)
                self.hyperedge_to_entities[hyperedge_id].append(entity_id)
        return self.hyperedge_to_id[hyperedge]
    
    def get_entity_id(self, entity):
        """Get the ID of an entity, adding it if it doesn't exist."""
        if entity not in self.entity_to_id:
            return -1
        return self.entity_to_id[entity]
    
    def get_hyperedge_id(self, hyperedge):
        """Get the ID of a hyperedge, adding it if it doesn't exist."""
        if hyperedge not in self.hyperedge_to_id:
            return -1
        return self.hyperedge_to_id[hyperedge]
    
    def get_hyperedge_entities_id(self, hyperedge_id :int) -> List[int]:
        """Get the entity IDs of a hyperedge by its ID."""
        if hyperedge_id < 0 or hyperedge_id >= len(self.hyperedge_to_entities):
            return []
        return self.hyperedge_to_entities[hyperedge_id]
    
    def get_entity_hyperedges_id(self, entity_id: int) -> List[int]:
        """Get the hyperedge IDs of an entity by its ID."""
        if entity_id < 0 or entity_id >= len(self.entity_to_hyperedges):
            return []
        return self.entity_to_hyperedges[entity_id]
    
    def get_entity_name(self, entity_id: int) -> Union[str, None]:
        """Get the name of an entity by its ID."""
        if entity_id < 0 or entity_id >= len(self.entities):
            return None
        return self.entities[entity_id][0]["entity_name"]
    
    def get_hyperedge_name(self, hyperedge_id: int) -> Union[str, None]:
        """Get the name of a hyperedge by its ID."""
        if hyperedge_id < 0 or hyperedge_id >= len(self.hyperedges):
            return None
        return self.hyperedges[hyperedge_id]
    
    def get_entity_description(self, entity_id: int) -> Union[str, None]:
        """Get the description of an entity by its ID."""
        if entity_id < 0 or entity_id >= len(self.entities):
            return None
        

        entity_data_list = self.entities[entity_id]
        entity = entity_data_list[0]["entity_name"]
        synonyms = [edata["entity_name"] for edata in entity_data_list[1:]]
        description = f"({entity}{self.tuple_delimiter}"
        if synonyms:
            description += f"{entity} has synonyms: " + ", ".join(synonyms) + "."

        for entity_data in entity_data_list:
            description += "\n" + entity_data["description"]
        description += f"){self.record_delimiter}"
        return description

    
    def print_entities(self) -> str:
        result = []

        for entity_id in range(len(self.entities)):
            description = self.get_entity_description(entity_id)
            if description:
                result.append(description)

        return "\n".join(result)
    

    def print_hyperedges(self) -> str:
        return "\n".join(self.hyperedges)
    

    def score_entity_relevance_by_id(self, query_emb, entity_id):
        if entity_id < 0 or entity_id >= len(self.ent_desc_embs):
            return 0.0
        ent_embs = self.ent_desc_embs[entity_id]
        if not ent_embs:
            return 0.0
        query_embedding = np.array(query_emb)
        ent_embs = [np.array(emb) for emb in ent_embs]
        # Compute cosine similarity between query embedding and each entity description embeddings
        scores = [np.dot(query_embedding, ent_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(ent_emb)) for ent_emb in ent_embs]
        return float(max(scores))
    

    
class ReasoningDAG:
    def __init__(self, orig_question, plan_llm_answer, global_topic_entity_names):
        
        self.orig_question = orig_question
        self.global_topic_entity_names = global_topic_entity_names

        raw_subquestions, dag_edge_list = self.extract_dag_from_llm_answer(plan_llm_answer, global_topic_entity_names)

        if not raw_subquestions:
            raise ValueError(f"Failed to extract subquestions from LLM answer. {plan_llm_answer}")

        
        subq_topic_set = set()
        for subq_oid, subq, topics_in_sq in raw_subquestions:
            subq_topic_set.update(topics_in_sq)
        if subq_topic_set != set(global_topic_entity_names):
            raise ValueError(f"Subquestion topics {subq_topic_set} do not match global topics {set(global_topic_entity_names)}.")
        


        self.n = len(raw_subquestions)
        self.dag_edges = dag_edge_list

        node_levels, node_order = self.topo_levels_from_edges(len(raw_subquestions), dag_edge_list)
        self.subquestions = []
        self.subquestion_to_id = {}
        self.levels = node_levels
        

        for subq_oid, subq, topics_in_sq in raw_subquestions:
            if subq_oid in node_order:
                level = node_order[subq_oid]
                self.subquestions.append({"id": subq_oid, "subquestion": subq, "topics": topics_in_sq, "level": level, "answer": None, "path": []})
                self.subquestion_to_id[subq] = subq_oid
            else:
                logger.warning(f"Subquestion ID {subq_oid} not found in DAG nodes.")

        self.completed_sq_oids = []
        self.completed_level = -1

    @staticmethod
    def extract_dag_from_llm_answer(llm_answer, global_topic_entity_names) -> tuple[list[tuple[int,str,list[str]]], list[tuple[int,int]]] | tuple[None,None]:
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        
        subquestions_section = llm_answer.split("<subquestions>")[1].split("</subquestions>")[0].strip()
        subquestions = parse_llm_result_into_lists(
            content=subquestions_section, 
            tuple_delimiter=tuple_delimiter, 
            record_delimiter=record_delimiter, 
            completion_delimiter=completion_delimiter
        )
        logger.debug(f"Extracted subquestions: {subquestions}.")

        for i, subq_data in enumerate(subquestions):
            if len(subq_data) < 2:
                logger.warning("Invalid subquestion format.")
                return [], []
            if len(subq_data) == 2:
                subq_oid, subq = subq_data
                topics_str = ""
            else:
                subq_oid, subq, topics_str = subq_data
            subq_data = []


            subq_oid = int(subq_oid.strip())
            if subq_oid < 0 or subq_oid >= len(subquestions):
                logger.warning("Subquestion id not in range.")
                return [], []
            subq_data.append(subq_oid)
            
            
            subq = subq.strip()
            if not subq:
                logger.warning("Invalid subquestion format.")
                return [], []
            subq_data.append(subq)
        
            topics_in_sq = []

            if topics_str.strip():
                # find all substrings enclosed in quotes
                matches = re.findall(r'"(.*?)"', topics_str.strip())
                for m in matches:
                    real = m.strip().upper()
                    if real:
                        topics_in_sq.append(f'"{real}"')

            # check if all topic_strs are in topic_entities
            for t in topics_in_sq:
                if t not in global_topic_entity_names:              
                    logger.warning(f"Subquestion topic entity '{t}' not in topic entities {global_topic_entity_names}.")
                    return [], []
            subq_data.append(topics_in_sq)



            subquestions[i] = subq_data

        dag_section = llm_answer.split("<dag>")[1].split("</dag>")[0].strip()
        
        dag_edge_list = parse_llm_result_into_tuples(
            content=dag_section, 
            tuple_delimiter=tuple_delimiter, 
            record_delimiter=record_delimiter, 
            completion_delimiter=completion_delimiter
        )

        logger.debug(f"Extracted DAG edges {dag_edge_list} from LLM answer.")

        # check if dag_edge_list is valid
        for i, (u, v) in enumerate(dag_edge_list):
            u = int(u.strip())
            v = int(v.strip())
            if u == v or u < 0 or v < 0 or u >= len(subquestions) or v >= len(subquestions):
                logger.warning("Invalid DAG edge.")
                return [], []
            dag_edge_list[i] = (u, v)
        return sorted(subquestions), dag_edge_list
        
    
    # @staticmethod
    # def topo_levels_from_edges(
    #     edges: list[tuple[str,str]],
    # ) -> tuple[list[list[str]], dict[str, int]]:
    def topo_levels_from_edges(
        self,
        nb_of_nodes: int,
        edges: list[tuple[str,str]],
    ) -> tuple[list[list[str]], dict[str, int]]:
        """
        Deterministic topological layering for a DAG.
        Input:
        edges: directed edges (u, v) meaning u -> v.
        
        Output:
            levels: list of layers; nodes in the same inner list share the same level.
            node_order:  node -> level index.

        Raises:
            ValueError if the graph has a cycle.
        """
        nodes = list(range(nb_of_nodes))

        adj_list = defaultdict(list)
        
        for u, v in edges:
            adj_list[u].append(v)
        for u in adj_list.keys():
            adj_list[u] = sorted(adj_list[u])

        
        in_degr = {u: 0 for u in nodes}
        for u in adj_list.keys():
            for v in adj_list[u]:
                in_degr[v] += 1
        
        frontier = sorted([u for u, d in in_degr.items() if d == 0])
        levels = []
        node_order = {}

        
        processed = 0
        level = 0

        while frontier:
            # Emit current level in deterministic order
            levels.append(frontier[:])
            for u in frontier:
                node_order[u] = level
            processed += len(frontier)

            # Collect next frontier; accumulate and then sort once
            next_frontier = []
            for u in frontier:
                for v in adj_list[u]:
                    in_degr[v] -= 1
                    if in_degr[v] == 0:
                        next_frontier.append(v)

            frontier = sorted(next_frontier)
            level += 1

        if processed != len(in_degr):
            raise ValueError("Graph has a cycle; topological levels undefined.")

        return levels, node_order
    
    
    def print_dag(self) -> str:
        dag_str = f"Orig Question: {self.orig_question}\n"
        dag_str += f"Global Topic Entities: {self.global_topic_entity_names}\n\n"
        for i, level in enumerate(self.levels): 
            level_str = f"Level {i}: " 
            if i <= self.completed_level:
                level_str += f"(completed)\n"
            else:
                level_str += "\n"
            for node_id in level:
                subq_data = self.subquestions[node_id]
                subq_str = f"Subquestion{node_id}: {subq_data['subquestion']}\n"
                if subq_data['answer'] is not None:
                    subq_str += f"Answer: {subq_data['answer']}\n"
                if subq_data['path'] is not None:
                    subq_str += f"Path: {subq_data['path']}\n"
                level_str += subq_str
            dag_str += level_str + "\n"
        return dag_str

    
    def copy(self):
        return copy.deepcopy(self)
    
    def refine(self, llm_answer):
        """Refine the current dag based on the llm answer, that is, add new subquestions and edges"""

        logger.debug(f"Refining DAG with LLM answer: {llm_answer}")
        new_raw_subquestions, new_dag_edge_list = self.extract_dag_from_llm_answer(llm_answer, self.global_topic_entity_names)

        logger.debug(f"Extracted subquestions: {new_raw_subquestions}.")
        logger.debug(f"Extracted DAG edges {new_dag_edge_list} from LLM answer.")

        if not new_raw_subquestions and not new_dag_edge_list:
            logger.warning("Failed to refine DAG from LLM answer.")
            return
        
        combined_subquestions = []
        combined_subquestion_to_id = {}
        for sq_oid in self.completed_sq_oids:
            sq_data = self.subquestions[sq_oid]
            combined_subquestions.append(sq_data)
            combined_subquestion_to_id[sq_data["subquestion"]] = sq_oid

        logger.debug(f"Current completed subquestions: {self.completed_sq_oids}")
        logger.debug(f"Current completed level: {self.completed_level}")
        logger.debug(f"Current completed(combined) subquestions data: {combined_subquestions}")
        logger.debug(f"Current completed(combined) subquestion to id map: {combined_subquestion_to_id}")

        

        cur_level_sqs = self.levels[self.completed_level]

        logger.debug(f"Current level subquestions: {cur_level_sqs}")
        combined_dag_edge_list = []
        for u, v in self.dag_edges:
            if u in self.completed_sq_oids and v in self.completed_sq_oids:
                combined_dag_edge_list.append((u, v))
        logger.debug(f"Current completed(combined) DAG edges: {combined_dag_edge_list}")
        for u, v in new_dag_edge_list:
            if u in self.completed_sq_oids:
                if v in self.completed_sq_oids and (u, v) not in combined_dag_edge_list:
                    logger.warning("Invalid attempt to modify DAG with edge ({u},{v}).")
                    return
                if v not in self.completed_sq_oids and u not in cur_level_sqs:
                    logger.warning("Invalid attempt to modify DAG with edge ({u},{v}).")
                    return
            combined_dag_edge_list.append((u, v))
        

        total_nb_of_nodes = len(self.completed_sq_oids)
        for subq_oid, _, _ in new_raw_subquestions:
            if subq_oid not in self.completed_sq_oids:
                total_nb_of_nodes += 1
        
        combined_node_levels, combined_node_order = self.topo_levels_from_edges(total_nb_of_nodes, combined_dag_edge_list)

        logger.debug(f"Combined DAG levels (drafted)): {combined_node_levels}")
        logger.debug(f"Combined DAG node order (drafted): {combined_node_order}")

        for subq_oid, subq, topics_in_sq in new_raw_subquestions:

            for t in topics_in_sq:
                if t not in self.global_topic_entity_names:
                    self.global_topic_entity_names.append(t)

            if subq_oid in self.completed_sq_oids:
                continue
            if subq_oid in combined_node_order:
                level = combined_node_order[subq_oid]
                combined_subquestions.append({"id": subq_oid, "subquestion": subq, "topics": topics_in_sq, "level": level, "answer": None, "path": []})
                combined_subquestion_to_id[subq] = subq_oid
            else:
                logger.warning(f"Subquestion ID {subq_oid} not found in DAG nodes.")
        logger.debug(f"Combined(completed + new) subquestions data: {combined_subquestions}")
        logger.debug(f"Combined(completed + new) subquestion to id map: {combined_subquestion_to_id}")
        logger.debug(f"Combined(completed + new) DAG edges: {combined_dag_edge_list}")
        
        self.n = len(combined_subquestions)
        self.dag_edges = combined_dag_edge_list
        self.subquestions = combined_subquestions
        self.subquestion_to_id = combined_subquestion_to_id
        self.levels = combined_node_levels

class DAGFrontier:
    def __init__(self, init_dags: list[ReasoningDAG], query_param: QueryParam):
        self.dags = []
        self.query_param = query_param
        
        if self.query_param.search_tree_mode == "BFS":
            self.dags = deque(init_dags[:])
        else:
            self.dags = init_dags[:]
        
    def push(self, dag: ReasoningDAG):
        self.dags.append(dag)
    
    def pop(self) -> ReasoningDAG | None:
        if self.dags:
            if self.query_param.search_tree_mode == "BFS":
                return self.dags.popleft()
            else:
                return self.dags.pop()
        return None
    
    def extend(self, dags: list[ReasoningDAG]):
        self.dags.extend(dags)
    
    def is_empty(self) -> bool:
        return len(self.dags) == 0
    
    def __len__(self): 
        return len(self.dags)

    
 
class ReasoningDAGAgent:
    def __init__(self, 
                 upperbound_graph: Hypergraph, 
                 dags: list[ReasoningDAG], 
                 global_config: dict, 
                 query_param: QueryParam, 
                 text_chunks_db: BaseKVStorage[TextChunkSchema],
    ):
        self.upperbound_graph = upperbound_graph
        self.init_dags = dags
        self.answered_subquestions = set()
        self.global_config = global_config
        self.text_chunks_db = text_chunks_db
        self.query_param = query_param
        self.use_model_func = global_config["llm_model_func"]

        self.tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        self.completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        self.max_nb_of_answers = query_param.max_nb_of_answers
        self.peak_tree_width = 0
        self.peak_tree_depth = -1 # root level is 0
        self.total_visited_states = 0
        self.total_seen_states = 0
        self.max_seen_dag_level = -1 # root level is 0

        self.token_usage = defaultdict(int) 
        self.graph_search_depth_stats = defaultdict(int)
    

    async def reason(self):
        """Perform reasoning based on the current DAG and Knowledge Hypergraph in a dfs style, i.e. each subquestion node may have multiple answers. the reasoning should process subquestions level by level. and the multiple answers are like branches. The branches will be processed in a dfs style"""
        # given current dag from the stack, check the processing level
        # process all subquestions in the current level
        # for each subquestion, if it can not be answered abandon this branch
        # if there is an answer to each subquestion, feed the answers with the current dag into the llm and asking for new dag
        # that is, the missing part of some next level subquestions may be filled by the answers of the current level
        # or the llm may generate new subquestions
        # if there are multiple answers for one subquestion, the process will branch, that is, one new dag for each combination of answers 
        # push the new dag(s) on to the stack
        
        finished = []

        dag_frontier = DAGFrontier(init_dags=self.init_dags, query_param=self.query_param)
        self.peak_tree_width = max(self.peak_tree_width, len(dag_frontier))
        self.total_seen_states += len(dag_frontier)
        self.max_seen_dag_level = max(self.max_seen_dag_level, max([len(d.levels) for d in self.init_dags]) - 1)
        
        while not dag_frontier.is_empty() and len(finished) < self.max_nb_of_answers:
            cur_dag = dag_frontier.pop()
            self.total_visited_states += 1
            logger.debug(f"Current reasoning DAG:\n{cur_dag.print_dag()}")
            # process the current dag
            cur_level = cur_dag.completed_level + 1

            
            if cur_level >= len(cur_dag.levels):
                finished.append(cur_dag)
                # return finished
                continue

            self.peak_tree_depth = max(self.peak_tree_depth, cur_level)
            current_level_sqs = []
            for sq_oid in cur_dag.levels[cur_level]:
                sq_data = cur_dag.subquestions[sq_oid]
                current_level_sqs.append(sq_data)
            
            cur_level_answer_with_paths = {}
            abandon = False

            for subq_data in current_level_sqs:

                step = ReasoningStep(
                    upperbound_graph=self.upperbound_graph, 
                    subquestion_data=subq_data,
                    global_config=self.global_config,
                    query_param = self.query_param,
                    text_chunks_db=self.text_chunks_db
                    )
                await step.async_init()
                logger.debug(f"Answering subquestion {subq_data['id']}: {subq_data['subquestion']}")
                # answers = step.generate_answers(self.use_model_func)
                answer_path_pairs = await step.retrieve_path_and_answer(self.use_model_func)
                self.update_token_usage(step.token_usage)
                self.graph_search_depth_stats[step.depth] += 1
                if answer_path_pairs:
                    cur_level_answer_with_paths[subq_data["id"]] = answer_path_pairs
                else:
                    # abandon this branch
                    abandon = True
                    break
            if abandon:
                continue
            # generate new dags based on the current dag and the answers
            new_dags = await self.generate_new_dags(cur_dag, cur_level_answer_with_paths)
            dag_frontier.extend(new_dags)
            self.peak_tree_width = max(self.peak_tree_width, len(dag_frontier))
            self.total_seen_states += len(new_dags)
            self.max_seen_dag_level = max(self.max_seen_dag_level, max([len(d.levels) for d in new_dags]) - 1)
            logger.debug(f"Pushed {len(new_dags)} new DAGs to the frontier. Current frontier size: {len(dag_frontier)}")



        for dag in finished:
            logger.debug(f"Finished reasoning DAG:\n{dag.print_dag()}")
            
            print(dag.print_dag())

        tree_record = {
            "peak_tree_width": self.peak_tree_width,
            "peak_tree_depth": self.peak_tree_depth,
            "total_visited_states": self.total_visited_states,
            "total_seen_states": self.total_seen_states,
            "max_seen_dag_level": self.max_seen_dag_level,
        }
        return finished, tree_record


    def update_token_usage(self, token_usage: dict):
        for k, v in token_usage.items():
            self.token_usage[k] += v
        
    def format_reasoning_plan_progress(self, dag: ReasoningDAG) -> str:
        result = []
        for i, level in enumerate(dag.levels):  
            level_str = f"Level {i}: "
            for node_id in level:
                q, a = dag.subquestions[node_id]['subquestion'], dag.subquestions[node_id]['answer']
                level_str += f"({node_id}{self.tuple_delimiter}{q}{self.tuple_delimiter}{','.join(dag.subquestions[node_id]['topics'])}){self.record_delimiter}\n"
                if a:
                    level_str += f"Answer: {a}\n"
            result.append(level_str)
        return "\n".join(result)

    async def generate_new_dags(self, cur_dag: ReasoningDAG, cur_level_answer_with_paths: dict[int, list[tuple[str,list[int]]]]) -> list[ReasoningDAG]:
        """Generate new dags based on the current dag and the answers of the current level"""
        
        use_model_func = self.global_config["llm_model_func"]
        
        dag_refinement_prompt_temp = PROMPTS["dag_refinement"]

        dag_refinement_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )


        logger.debug(f"global_topic_entity_names: {cur_dag.global_topic_entity_names}")

        new_dags = []
        # for each combination of answers, generate a new dag
        cur_sq_ids = list(cur_level_answer_with_paths.keys())
        cur_answer_with_paths = [cur_level_answer_with_paths[sq_id] for sq_id in cur_sq_ids]
        cur_answer_combinations = list(product(*cur_answer_with_paths))
        for answer_comb in cur_answer_combinations:
            # create a new dag based on the current dag
            new_dag = cur_dag.copy()
            logger.debug(f"Processing answer combination: {answer_comb}")
            
            for sq_oid, (answer, path) in zip(cur_sq_ids, answer_comb):
                new_dag.subquestions[sq_oid]["answer"] = answer
                new_dag.subquestions[sq_oid]["path"] = path
                new_dag.completed_sq_oids.append(sq_oid)
            new_dag.completed_level = cur_dag.completed_level + 1
            logger.debug(f"Answer added DAG:\n{new_dag.print_dag()}")

            if new_dag.completed_level + 1 >= len(new_dag.levels):
                new_dags.append(new_dag)
                continue

            reasoning_plan_progress = self.format_reasoning_plan_progress(new_dag)
            dag_refinement_prompt = dag_refinement_prompt_temp.format(
                **dag_refinement_prompt_base, 
                input_question=cur_dag.orig_question,
                topic_entities=', '.join(cur_dag.global_topic_entity_names),
                progress=reasoning_plan_progress
            )
            
            dag_refinement_llm_answer, token_usage =  await use_model_func(dag_refinement_prompt, count_token=True)

            self.token_usage["DAG_refine"] += token_usage["total_tokens"]


            logger.debug(f"DAG refinement LLM answer:\n{dag_refinement_llm_answer}")
            try:
                new_dag.refine(dag_refinement_llm_answer)
                logger.debug(f"Refined reasoning DAG:\n{new_dag.print_dag()}")

            except ValueError as e:
                logger.warning(f"Failed to refine DAG: {e}")
                continue

            logger.debug(f"Refined reasoning DAG:\n{new_dag.print_dag()}")

            new_dags.append(new_dag)
        return new_dags
            


class ReasoningStep:
    def __init__(
            self, 
            upperbound_graph: Hypergraph, 
            subquestion_data: dict,
            global_config: dict, 
            query_param: QueryParam,
            text_chunks_db: BaseKVStorage[TextChunkSchema],
        ):
        self.upperbound_graph = upperbound_graph
        self.subq_data = subquestion_data
        self.subq_emb = None
        self.global_config = global_config
        self.text_chunks_db = text_chunks_db
        self.query_param = query_param
        self.use_model_func = global_config["llm_model_func"]
        self.embedding_func = global_config["embedding_func"]
        self.max_paths = query_param.max_paths
        self.max_depth = query_param.max_depth
        self.max_width = query_param.max_width
        self.draft_width = query_param.draft_width
        self.he_score_threshold = query_param.he_score_threshold
        self.subquestion_id = subquestion_data['id']
        self.subquestion = subquestion_data['subquestion']

        self.topics = []
        self.topic_ent_ids = []
        self.linked_hes = []
        self.linked_he_ids = set()
        
        self.entity_scores: dict[int, float] = {}
        self.path_dict = dict()
        self.frontier = dict()

        self.tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        self.completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        self.token_usage = defaultdict(int)
        self.depth = 0
    
    async def async_init(self):
        if self.query_param.emb_model == "OPENAI":
            self.subq_emb = await self.embedding_func(self.subquestion)
        else:
            subq_emb = self.upperbound_graph.model.encode(
                [self.subquestion], 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            self.subq_emb = subq_emb.cpu().numpy()

        logger.debug(f"Re-extract subquestion topics for '{self.subquestion}'")

        self.topics= await self.topic_reinitialisation(
            query=self.subquestion, 
            # entity_names_vdb=self.upperbound_graph.entity_names_vdb,
            global_config=self.global_config
        )

        self.subq_data['topics'] = self.topics
        self.topic_ent_ids = [self.upperbound_graph.get_entity_id(ent) for ent in self.topics if self.upperbound_graph.get_entity_id(ent) != -1]
        if self.query_param.with_target_hyperedges:
            self.linked_hes = await self.upperbound_graph.he_match([self.subquestion])
            self.linked_he_ids = set(self.upperbound_graph.get_hyperedge_id(he) for he in self.linked_hes)


    

    async def topic_reinitialisation(
            self,
            query: str, 
            global_config: dict,
        ) -> list[str]:
        """
        This function is used to extract topic entities and subquestions from the query.
        It uses a language model to process the query and extract the relevant information.
        """

        use_model_func = global_config["llm_model_func"]

        topic_initialisation_prompt_temp = PROMPTS["topic_initialisation"]
        topic_initialisation_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )
        
        topic_initialisation_prompt = topic_initialisation_prompt_temp.format(
            **topic_initialisation_prompt_base, input_question=query
        )


        topic_entities: list[str] = []
        try_count = 0
        max_tries = 3
        while len(topic_entities) < 1 and try_count < max_tries:
            try_count += 1

            topic_initialisation_result, token_usage = await use_model_func(topic_initialisation_prompt, count_token=True)
            self.token_usage["topic_reinit"] += token_usage["total_tokens"]

            logger.info("Topic initialisation LLM result:")
            logger.info(f"{topic_initialisation_result}")

            try:
                key_entities_section = topic_initialisation_result.split("<entities>")[1].split("</entities>")[0].strip()
                key_entities = parse_llm_result_into_lists(
                    content=key_entities_section, 
                    tuple_delimiter=topic_initialisation_prompt_base["tuple_delimiter"], 
                    record_delimiter=topic_initialisation_prompt_base["record_delimiter"], 
                    completion_delimiter=topic_initialisation_prompt_base["completion_delimiter"]
                )
                key_entities = key_entities[0]
                key_entities = [k.strip().upper() for k in key_entities if k.strip()]
            except Exception as e:
                logger.warning(f"Failed to parse key entities from LLM result: {e}")
                key_entities = []


            topic_entities = await self.upperbound_graph.topic_link(key_entities)

        return topic_entities

    async def get_entity_scores(self, entity_ids: list[int]) -> dict[int, float]:
        if not entity_ids:
            return {}
        scores: dict[int, float] = {}
        missing = []
        for ent_id in entity_ids:
            if ent_id in self.entity_scores:
                scores[ent_id] = self.entity_scores[ent_id]
            else:
                missing.append(ent_id)
        
        if self.query_param.score_mode != "LLM":

            further = []
            
            for ent_id in missing:
                score = self.upperbound_graph.score_entity_relevance_by_id(self.subq_emb, ent_id)
                if self.query_param.score_mode == "EMB":
                    scores[ent_id] = score
                    self.entity_scores[ent_id] = score
                else:
                    if score >= self.query_param.emb_filter_threshold:
                        further.append(ent_id)
                    else:
                        scores[ent_id] = 0.0
                        self.entity_scores[ent_id] = 0.0
           
            missing = further
        if not missing:
            return scores
  
        entity_eva_prompt_temp = PROMPTS["entity_evaluation"]

        entity_eva_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )
        
        max_rounds = 3
        cur_round = 0
        max_batch_size = 20

        while missing and cur_round < max_rounds:
            left = []

            for i in range(0, len(missing), max_batch_size):
                batch = missing[i : i + max_batch_size]

                entity_descriptions = [self.upperbound_graph.get_entity_description(ent_id) for ent_id in batch]
               
                entity_eva_prompt = entity_eva_prompt_temp.format(
                    **entity_eva_prompt_base, 
                    entity_descriptions="\n".join(entity_descriptions), 
                    question=self.subquestion
                )
                logger.debug(f"Scoring entities: {batch}")

                llm_answer, token_usage = await self.use_model_func(entity_eva_prompt, count_token=True)
                self.token_usage["entity_score"] += token_usage["total_tokens"]
                logger.debug(f"LLM answer:\n{llm_answer}")

                try:
                    ent_id_scores = self.parse_entity_scores(llm_answer)
                except Exception as e:
                    logger.warning(f"Failed to parse entity scores from LLM answer: {e}")
                    ent_id_scores = {}

                logger.debug(f"Entity scores: {ent_id_scores}")

                
                scores.update(ent_id_scores)
                self.entity_scores.update(ent_id_scores)
                for ent_id in batch:
                    if ent_id not in ent_id_scores:
                        left.append(ent_id)
            missing = left
            cur_round += 1
        for ent_id in missing:
            scores[ent_id] = 0.0
            self.entity_scores[ent_id] = 0.0
        if missing:
            logger.warning(f"Failed to get scores for entities {[self.upperbound_graph.get_entity_name(eid) for eid in missing]} after {max_rounds} rounds.")
        return scores

  
    def parse_entity_scores(self, llm_answer: str) -> dict[int, float]:
        ent_scores = {}

        score_section = llm_answer.split("<entity_scores>")[1].split("</entity_scores>")[0].strip()
        tuples = parse_llm_result_into_tuples(
            content=score_section, 
            tuple_delimiter=self.tuple_delimiter, 
            record_delimiter=self.record_delimiter, 
            completion_delimiter=self.completion_delimiter
        )
        for ent, score in tuples:
            try:
                score = int(score.strip()) / 10.0
                # score = float(score.strip())
                ent = ent.strip().upper()     
                ent_id = self.upperbound_graph.get_entity_id(ent)
                if ent_id != -1:
                    ent_scores[ent_id] = score
            except ValueError:
                logger.warning(f"Invalid entity score format: {ent}, {score}.")
                continue
        return ent_scores


    async def form_paths_from_path_dict(self, fall_back=False) -> list[list[int]]:
        """
        # if only one topic, use every path
        # if multiple topics, use only the paths that connect topics in sequence
        # fall back mode: if cannot connect topics, use unconnected partial paths
        """
        async def _simple_score_path(path: list[int]) -> float:
            path_entity_ids = set()
            for he_id in path:
                he_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(he_id)
                path_entity_ids.update(he_entity_ids)
            path_entity_scores = await self.get_entity_scores(list(path_entity_ids))
            path_entity_scores = list(path_entity_scores.values())
            return sum(path_entity_scores)
        
        async def _best_path_single_src(src_name: str) -> list[int]:
            if src_name not in self.path_dict:
                return []
            src_path_list = []
            for he_id, path_list in self.path_dict[src_name].items():
                src_path_list.extend(path_list)
            if not src_path_list:
                return []
            path_scores = await asyncio.gather(*[_simple_score_path(path) for path in src_path_list])
            path_score_pairs = list(zip(path_scores, src_path_list))
            path_score_pairs = sorted(path_score_pairs, key=lambda x: x[0], reverse=True)
            best_path = path_score_pairs[0][1]
            return best_path
        
        def _merge_paths(a: list[int], b: list[int]) -> list[int]:
            if not a:
                return b[::-1] if b else []
            if not b:
                return a[:]

            b_rev = b[::-1]
            if a[-1] == b_rev[0]:
                return a + b_rev[1:]
            return a + b_rev
            
        paths = []
        if len(self.topics) == 1:
            for src in self.topics:
                for he_id, path_list in self.path_dict[src].items():
                    for path in path_list:
                        paths.append(path)
        else:
            for i in range(len(self.topics)-1):
                partial_paths = []
                src = self.topics[i]
                dest = self.topics[i+1]
                logger.debug(f"Forming paths from {src} to {dest}.")
                

                for mid_he_id in self.path_dict[src].keys():
                    if mid_he_id not in self.path_dict[dest].keys():
                        continue
                    src_path_list = self.path_dict[src][mid_he_id]
                    for src_path in src_path_list:
                        # if len(src_path) < depth - 1:
                        #     continue
                        dest_path_list = self.path_dict[dest][mid_he_id]
                        for dest_path in dest_path_list:
                            merged_path = _merge_paths(src_path, dest_path)
                            partial_paths.append(merged_path)
                logger.debug(f"Found partial paths: {partial_paths}.")

                if not partial_paths and fall_back:
                    # use unconnected partial paths
                    src_best_path = await _best_path_single_src(src)
                    dest_best_path = await _best_path_single_src(dest)
                    partial_paths = [_merge_paths(src_best_path, dest_best_path)]
                    logger.debug(f"Fall back to unconnected partial paths: {partial_paths}.")
                
                # limit the number of partial paths to avoid combinatorial explosion
                max_branch = 10
                if len(partial_paths) > max_branch:
                    path_scores = await asyncio.gather(*[_simple_score_path(path) for path in partial_paths])
                    path_score_pairs = list(zip(path_scores, partial_paths))
                    path_score_pairs = sorted(path_score_pairs, key=lambda x: x[0], reverse=True)
                    path_sorted = [p for s, p in path_score_pairs]
                    partial_paths = path_sorted[:max_branch]
                
                paths.append(partial_paths)
            # combine the partial paths
            if paths:
                combined_paths = list(product(*paths))
                final_paths = []
                for comb in combined_paths:
                    # logger.debug(f"Combining partial paths: {comb}.")
                    final_path = []
                    for part in comb:
                        final_path.extend(part)
                    # logger.debug(f"Combined path: {final_path}.")
                    final_paths.append(final_path)

                paths = final_paths
            else:
                paths = []
        
        for linked_he in self.linked_hes:
            for he_id, path_list in self.path_dict[linked_he].items():
                for path in path_list:
                    paths.append(path)

        path_infos = []
        for path in paths:
    
            he_match = False
            path_entity_ids = set()
            for he_id in path:
                if he_id in self.linked_he_ids:
                    he_match = True
                he_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(he_id)
                path_entity_ids.update(he_entity_ids)
            
            path_entity_scores = await self.get_entity_scores(list(path_entity_ids))
            path_entity_scores = list(path_entity_scores.values())
            path_score = sum(path_entity_scores)
            if he_match:
                path_infos.append((1, path_score, path))
            else:
                path_infos.append((0, path_score, path))

        if self.query_param.form_path_selection == "RAND":
            path_infos = random.sample(path_infos, min(len(path_infos), self.max_paths))
        else:
            path_infos = sorted(path_infos, key=lambda x: (x[0], x[1]), reverse=True)
            path_infos = path_infos[:self.max_paths]
        return path_infos
    

    def add_context_to_paths(self, paths: list[list[int]]) -> list[tuple[str, str]]:
        paths_with_context = []


        for path_info in paths:
            _, _, path = path_info

            ent_ids = set()


            he_names = []
            for he_id in path:
                he_name = self.upperbound_graph.get_hyperedge_name(he_id)
                he_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(he_id)
                ent_ids.update(he_entity_ids)
                he_names.append(he_name)

            entity_descriptions = [self.upperbound_graph.get_entity_description(ent_id) for ent_id in ent_ids]
            he_str = " -> ".join(he_names)
            ent_str = "\n".join(entity_descriptions)
            paths_with_context.append((path, he_str, ent_str))
        return paths_with_context
    

    async def retrieve_chunks_for_path(self, path: list[int]) -> str:
        # retrieve text chunks for the hyperedges in the path
        
        he_names = []
        for he_id in path:
            he_name = self.upperbound_graph.get_hyperedge_name(he_id)
            he_names.append(he_name)
        
        hyperedge_datas = [await self.upperbound_graph.base.get_hyperedge(he_name) for he_name in he_names]
        text_unit_ids = []
        for hd in hyperedge_datas:
            units = split_string_by_multi_markers(hd["source_id"], [GRAPH_FIELD_SEP])
            for unit in units:
                if unit not in text_unit_ids:
                    text_unit_ids.append(unit)
        
        text_contexts = []

        for unit_id in text_unit_ids:
            chunk_data = await self.text_chunks_db.get_by_id(unit_id)
            if chunk_data is not None and "content" in chunk_data:
                text_contexts.append(chunk_data["content"])
        
        return '\n'.join(text_contexts)
    

    async def llm_select_paths(self, paths_with_context: list[tuple[list[int], str, str]]) -> tuple[list[int], int]:
        path_selection_prompt_temp = PROMPTS["final_path_selection"]

        path_selection_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

        path_records = []
        for i, (path, he_str, ent_str) in enumerate(paths_with_context):

            record = f"Path {i}: {he_str}"
            path_records.append(record)
        path_selection_prompt = path_selection_prompt_temp.format(
            **path_selection_prompt_base, 
            paths="\n".join(path_records), 
            question=self.subquestion
        )

        llm_answer, token_usage = await self.use_model_func(path_selection_prompt, count_token=True)
        self.token_usage["path_selection"] += token_usage["total_tokens"]
        logger.debug(f"Path selection LLM answer:\n{llm_answer}")


        flag = llm_answer.split("<flag>")[1].split("</flag>")[0].strip().lower()
        
        selected_ids = []
        if flag != "yes":
            return selected_ids
        
        id_section = llm_answer.split("<id>")[1].split("</id>")[0].strip()


        ids = parse_llm_result_into_lists(
            content=id_section, 
            tuple_delimiter=path_selection_prompt_base["tuple_delimiter"], 
            record_delimiter=path_selection_prompt_base["record_delimiter"], 
            completion_delimiter=path_selection_prompt_base["completion_delimiter"]
            )

        logger.debug(f"Selected path IDs: {ids}")

        if not ids:
            return selected_ids
        
        ids = ids[0]
        
        for id in ids:
            try:
                id = int(id.strip())
                if id >= 0 and id < len(paths_with_context):
                    selected_ids.append(id)
            except ValueError:
                continue
        return selected_ids

    

    async def llm_select_directions(self, drafted_paths: list[tuple[float, float, int, str, list[int]]]) -> list[tuple[int, str, list[int]]]:

        dirs = []
        for _, _, he_id, src, new_path in drafted_paths:
            dirs.append((he_id, src, new_path))
        
        if len(dirs) <= self.max_width:
            return dirs
        
        
        dirs_with_context = self.add_context_to_paths(dirs)


        dir_records = []
        for i, (path, he_str, ent_str) in enumerate(dirs_with_context):

            record = f"Direction {i}: {he_str}"
            dir_records.append(record)
        
        dir_selection_prompt_temp = PROMPTS["direction_selection"]

        dir_selection_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

        dir_selection_prompt = dir_selection_prompt_temp.format(
            **dir_selection_prompt_base, 
            width = self.max_width,
            directions="\n".join(dir_records), 
            question=self.subquestion
        )

        llm_answer, token_usage = await self.use_model_func(dir_selection_prompt, count_token=True)
        self.token_usage["dir_selection"] += token_usage["total_tokens"]
        logger.debug(f"Direction selection LLM answer:\n{llm_answer}")


        selected_ids = []
        
        id_section = llm_answer.split("<id>")[1].split("</id>")[0].strip()


        ids = parse_llm_result_into_lists(
            content=id_section, 
            tuple_delimiter=dir_selection_prompt_base["tuple_delimiter"], 
            record_delimiter=dir_selection_prompt_base["record_delimiter"], 
            completion_delimiter=dir_selection_prompt_base["completion_delimiter"]
            )

        logger.debug(f"Selected direction IDs: {ids}")
        
        if not ids:
            return dirs[:self.max_width]
        
        ids = ids[0]

        selected_dirs = []
        selected_ids = []
        for id in ids:
            try:
                id = int(id.strip())
                if id >= 0 and id < len(dirs):
                    selected_dirs.append(dirs[id])
                    selected_ids.append(id)
            except ValueError:
                continue
        if len(selected_dirs) < self.max_width:
            for i, path in enumerate(dirs):
                if i not in selected_ids:
                    selected_dirs.append(path)
                if len(selected_dirs) >= self.max_width:
                    break
        return selected_dirs


    async def BFS_with_pruning(self):
    
        new_frontier = dict()

        cur_entity_set = set()
        for cur_he_id in self.frontier.keys():
            cur_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(cur_he_id)
            cur_entity_set.update(cur_entity_ids)
        cur_entities_scores = await self.get_entity_scores(cur_entity_set)
        max_ent_score = max(cur_entities_scores.values()) if cur_entities_scores else 0.0

        draft_next_hes_with_paths = []
        for cur_he_id in self.frontier.keys():
            cur_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(cur_he_id)

            next_he_scores = defaultdict(list)
            for cur_ent_id in cur_entity_ids:
               
                cur_ent_hyperedges = self.upperbound_graph.get_entity_hyperedges_id(cur_ent_id)
                for he in cur_ent_hyperedges:
                    if he != cur_he_id:
                        next_he_scores[he].append(cur_entities_scores[cur_ent_id])
            
            
            for he, he_scores in next_he_scores.items():
                he_max_score = max(he_scores)
                he_avg_score = sum(he_scores) / len(he_scores)

                if he_max_score == max_ent_score or he_avg_score > self.he_score_threshold:

                    for src, path in self.frontier[cur_he_id]:
                        if he in path:
                            continue
                        new_path = path + [he]
                        draft_next_hes_with_paths.append((he_max_score, he_avg_score, he, src, new_path))


        if self.query_param.dir_selection == "RAND":
            draft_next_hes_with_paths = random.sample(draft_next_hes_with_paths, min(len(draft_next_hes_with_paths), self.max_width))
        elif self.query_param.dir_selection == "EOW":
            draft_next_hes_with_paths.sort(key=lambda x: (x[0], x[1]), reverse=True)
            draft_next_hes_with_paths = draft_next_hes_with_paths[:self.max_width]
        else: # LLM
            draft_next_hes_with_paths.sort(key=lambda x: (x[0], x[1]), reverse=True)
            draft_next_hes_with_paths = draft_next_hes_with_paths[:self.draft_width]
        selected_next_hes_with_paths = await self.llm_select_directions(draft_next_hes_with_paths)


        for he_id, src, new_path in selected_next_hes_with_paths:
            if he_id not in self.path_dict[src]:
                self.path_dict[src][he_id] = []
            self.path_dict[src][he_id].append(new_path)
            if he_id not in new_frontier:
                new_frontier[he_id] = []
            new_frontier[he_id].append((src, new_path))

        self.frontier = new_frontier



    async def generate_step_answer(self, he_str: str, ent_str: str) -> str:
        step_answer_prompt_temp = PROMPTS["step_answer_generation"]

        step_answer_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

        step_answer_prompt = step_answer_prompt_temp.format(
            **step_answer_prompt_base, 
            path=he_str, 
            entity_descriptions=ent_str, 
            question=self.subquestion
        )

        llm_answer, token_usage = await self.use_model_func(step_answer_prompt, count_token=True)
        self.token_usage["step_answer"] += token_usage["total_tokens"]
        logger.debug(f"Step answer LLM answer:\n{llm_answer}")
        answer = llm_answer.split("<answer>")[1].split("</answer>")[0].strip()
        return answer


    async def retrieve_path_and_answer(self, use_model_func) -> list[tuple[str, list[int]]]:      
        one_hop_ent_ids = set()

        for topic in self.topics:
            self.path_dict[topic] = {}
            t_ent_id = self.upperbound_graph.get_entity_id(topic)
            t_ent_hes = self.upperbound_graph.get_entity_hyperedges_id(t_ent_id)
            for he_id in t_ent_hes:
                self.frontier[he_id] = [(topic, [he_id])]
                self.path_dict[topic][he_id] = [[he_id]] # Initialize with 1 hop path
                he_ent_ids = self.upperbound_graph.get_hyperedge_entities_id(he_id)
                one_hop_ent_ids.update(he_ent_ids)
        

        for linked_he in self.linked_hes:
            self.path_dict[linked_he] = {}
            linked_he_id = self.upperbound_graph.get_hyperedge_id(linked_he)
            self.frontier[linked_he_id] = [(linked_he, [linked_he_id])]
            self.path_dict[linked_he][linked_he_id] = [[linked_he_id]]
            he_ent_ids = self.upperbound_graph.get_hyperedge_entities_id(linked_he_id)
            one_hop_ent_ids.update(he_ent_ids)

        
        await self.get_entity_scores(list(one_hop_ent_ids))

        for depth in range(1, self.max_depth+1):
            self.depth = depth
            logger.debug(f"Searching paths with depth {depth} for subquestion {self.subquestion_id}: {self.subquestion}")
            if depth > 1:
                await self.BFS_with_pruning()
            paths = await self.form_paths_from_path_dict()
            logger.debug(f"Found {len(paths)} paths with depth {depth} for subquestion {self.subquestion_id}: {self.subquestion}")
            paths_with_context = self.add_context_to_paths(paths)
            logger.debug(f"Paths with context: {paths_with_context}")
            if paths_with_context:
                selected_path_ids = await self.llm_select_paths(paths_with_context)
                if selected_path_ids:
                    answer_path_pairs = []
                    for selected_id in selected_path_ids:
                        selected_path, he_str, ent_str = paths_with_context[selected_id]
                        if self.query_param.with_src_chunks:
                            chunk_str = await self.retrieve_chunks_for_path(selected_path)
                            he_str += "\n" + chunk_str
                        answer = await self.generate_step_answer(he_str, ent_str)
                        answer_path_pairs.append((answer, selected_path))

                    logger.debug(f"Answer path pairs: {answer_path_pairs}")
                    return answer_path_pairs
        
        
        
        logger.debug(f"Reached max depth {self.max_depth} without finding a valid path, using the best available path if any.")
        answer_path_pairs = []
        paths = await self.form_paths_from_path_dict(fall_back=True)
        paths_with_context = self.add_context_to_paths(paths[:1]) 
        if paths_with_context:
            selected_path, he_str, ent_str = paths_with_context[0]
        else:
            selected_path, he_str, ent_str = [], "", ""
        answer = await self.generate_step_answer(he_str, ent_str)
        answer_path_pairs.append((answer, selected_path))
        return answer_path_pairs
            

   
async def topic_initialisation(
    query: str, 
    entity_names_vdb: BaseVectorStorage,
    global_config: dict,
    topic_top_k: int = 1,
    topic_threshold: float = 0.6
    ) -> list[str]:
    """
    This function is used to extract topic entities and subquestions from the query.
    It uses a language model to process the query and extract the relevant information.
    """

    start = time.time()
    use_model_func = global_config["llm_model_func"]

    topic_initialisation_prompt_temp = PROMPTS["topic_initialisation"]
    topic_initialisation_prompt_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],

    )
    
    topic_initialisation_prompt = topic_initialisation_prompt_temp.format(
        **topic_initialisation_prompt_base, input_question=query
    )

    topic_entities: list[str] = []
    
    max_time = 200
    token_usage_dict = defaultdict(int)
    while len(topic_entities) < 1 and (time.time() - start < max_time):
  

        topic_initialisation_result, token_usage = await use_model_func(topic_initialisation_prompt, count_token=True)

        token_usage_dict["topic_init"] += token_usage["total_tokens"]

        logger.debug("Topic initialisation LLM result:")
        logger.debug(f"{topic_initialisation_result}")

        try:
            key_entities_section = topic_initialisation_result.split("<entities>")[1].split("</entities>")[0].strip()
            key_entities = parse_llm_result_into_lists(
                content=key_entities_section, 
                tuple_delimiter=topic_initialisation_prompt_base["tuple_delimiter"], 
                record_delimiter=topic_initialisation_prompt_base["record_delimiter"], 
                completion_delimiter=topic_initialisation_prompt_base["completion_delimiter"]
            )
            key_entities = key_entities[0]
            key_entities = [k.strip().upper() for k in key_entities if k.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse key entities from LLM result: {e}")
            key_entities = []


        for key in key_entities:
            hits = await entity_names_vdb.query(key, top_k=topic_top_k, better_than_threshold=topic_threshold)
            if not hits:
                logger.warning(f"Can not link key entity '{key}' with knowledge hypergraph.")
                continue
        
            hit_names =  [t["entity_name"] for t in hits]
            topic_entities.extend(hit_names)
            logger.info(f"Key entity '{key}' is linked to: ")
            logger.info(', '.join(hit_names))

    if not topic_entities:
        logger.warning(f"Fallback to link topics directly from the query.")
        hits = await entity_names_vdb.query(query, top_k=topic_top_k*5, better_than_threshold=topic_threshold)
        if not hits:
            logger.warning(f"Can not link '{query}' with any entities in knowledge hypergraph.")
        else:
            hit_names =  [t["entity_name"] for t in hits]
            topic_entities.extend(hit_names)
            logger.debug(f"Link '{query}' with entities: {', '.join(topic_entities)}")
    return topic_entities, token_usage_dict
    
 
def _is_l0_hyperedge_data(hyperedge_data: Union[dict, None]) -> bool:
    if hyperedge_data is None:
        return False
    if hyperedge_data.get("role") != "hyperedge":
        return False
    return int(hyperedge_data.get("hyperedge_level", 0) or 0) == 0


async def _filter_l0_hyperedge_names(
    hyperedge_names: list[str],
    knowledge_graph_inst: BaseGraphStorage,
) -> list[str]:
    filtered_names = []
    seen_names = set()
    for he_name in hyperedge_names:
        if he_name in seen_names:
            continue
        he_data = await knowledge_graph_inst.get_hyperedge(he_name)
        if not _is_l0_hyperedge_data(he_data):
            continue
        filtered_names.append(he_name)
        seen_names.add(he_name)
    return filtered_names


async def target_hyperedge_matching(
    query: str, 
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    # global_config: dict,
    he_top_k: int = 5,
    he_threshold: float = 0.6
    ) -> list[str]:
    """
    This function is used to match query with hyperedges.
    """
    hes = await hyperedges_vdb.query(query, top_k=he_top_k, better_than_threshold=he_threshold)
    if not query:
        logger.debug(f"Can not a match for '{query}' in knowledge hypergraph.")
        return []
    raw_he_names = [he["hyperedge_name"] for he in hes]
    he_names = await _filter_l0_hyperedge_names(raw_he_names, knowledge_graph_inst)

    logger.info(f"Query '{query}' is matched with: ")
    logger.info(', '.join(he_names))
    return he_names
        

async def plan_context_graph_construction(
    knowledge_graph_inst: BaseGraphStorage,
    entity_names_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    global_config: dict,
    query_param: QueryParam,
    query: str,
    topic_entities: list[str],
    target_hyperedges: list[str],
) -> Hypergraph:

    plan_context_graph = Hypergraph(
        query_param = query_param,
        knowledge_graph_inst=knowledge_graph_inst, 
        entity_names_vdb=entity_names_vdb, 
        entities_vdb=entities_vdb,
        hyperedges_vdb=hyperedges_vdb
        )
    
    if query_param.emb_model == "OPENAI":
        embedding_func = global_config["embedding_func"]
        query_emb = await embedding_func(query)
    else:
        query_emb = plan_context_graph.model.encode(
            [query], 
            convert_to_tensor=True, 
            normalize_embeddings=True
            )
        query_emb = query_emb.cpu().numpy()

    for t in topic_entities:
        await plan_context_graph.add_entity(t)
    for he in target_hyperedges:
        await plan_context_graph.add_hyperedge(he)


    src_he_set = set(target_hyperedges)
    for t in topic_entities:
        hyperedges = await knowledge_graph_inst.get_entity_hyperedges(t, get_synonyms=query_param.get_synonyms)
        if hyperedges:
            src_he_set.update(await _filter_l0_hyperedge_names(hyperedges, knowledge_graph_inst))

    queue = []
    for he in src_he_set:
        queue.append((he, 1))
    queue = deque(queue)
   

    while queue:
        cur_he_name, cur_depth = queue.popleft()
        if plan_context_graph.get_hyperedge_id(cur_he_name) > -1:
            continue
        await plan_context_graph.add_hyperedge(cur_he_name)

        if cur_depth >= query_param.max_plan_depth:
            continue
        cur_entities = await knowledge_graph_inst.get_hyperedge_entities(cur_he_name) or []

        cur_entities_scores = {}
        for cur_ent in cur_entities:
            cur_ent_id = plan_context_graph.get_entity_id(cur_ent)
            cur_ent_score = plan_context_graph.score_entity_relevance_by_id(query_emb=query_emb, entity_id=cur_ent_id)
            cur_entities_scores[cur_ent] = cur_ent_score

        next_he_scores = defaultdict(list)
        for cur_ent in cur_entities:
            cur_ent_score = cur_entities_scores[cur_ent]
            cur_ent_hyperedges = await knowledge_graph_inst.get_entity_hyperedges(cur_ent, get_synonyms=query_param.get_synonyms) or []
            cur_ent_hyperedges = await _filter_l0_hyperedge_names(cur_ent_hyperedges, knowledge_graph_inst)
            for he in cur_ent_hyperedges:
                if he != cur_he_name:
                    next_he_scores[he].append(cur_ent_score)

        max_ent_score = max(cur_entities_scores.values())
        qualified_next_hes = []
        for he, he_scores in next_he_scores.items():
            if not he_scores:
                continue
            he_max_score = max(he_scores)
            he_avg_score = sum(he_scores) / len(he_scores)

            if he_max_score == max_ent_score or he_avg_score > query_param.he_score_threshold:
                qualified_next_hes.append((he, he_max_score, he_avg_score))

        qualified_next_hes = sorted(qualified_next_hes, key=lambda x: (x[1], x[2]), reverse=True)

        count = 0
        for next_he_name, _,_ in qualified_next_hes:
            
            if plan_context_graph.get_hyperedge_id(next_he_name) > -1:
                continue

            queue.append((next_he_name, cur_depth + 1))
            await plan_context_graph.add_hyperedge(next_he_name)
            count += 1
            if count >= query_param.max_width:
                break

    return plan_context_graph





async def upperbound_graph_construction(
    knowledge_graph_inst: BaseGraphStorage,
    entity_names_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    query_param: QueryParam,
    topic_entities: list[str],
    target_hyperedges: list[str]
) -> Hypergraph:

    upperbound_graph = Hypergraph(
        knowledge_graph_inst=knowledge_graph_inst, 
        entity_names_vdb=entity_names_vdb, 
        entities_vdb=entities_vdb,
        hyperedges_vdb=hyperedges_vdb,
        query_param = query_param
        )
    for t in topic_entities:
        await upperbound_graph.add_entity(t)
    for he in target_hyperedges:
        await upperbound_graph.add_hyperedge(he)


    src_he_set = set(target_hyperedges)
    for t in topic_entities:
        hyperedges = await knowledge_graph_inst.get_entity_hyperedges(t, get_synonyms=query_param.get_synonyms)
        if hyperedges:
            src_he_set.update(await _filter_l0_hyperedge_names(hyperedges, knowledge_graph_inst))

    queue = []
    for he in src_he_set:
        queue.append((he, 1))
    queue = deque(queue)
   
    while queue:
        cur_he_name, cur_depth = queue.popleft()
        if upperbound_graph.get_hyperedge_id(cur_he_name) > -1:
            continue
        await upperbound_graph.add_hyperedge(cur_he_name)
        if cur_depth >= query_param.max_depth:
            continue
        cur_entities = await knowledge_graph_inst.get_hyperedge_entities(cur_he_name)
        next_he_set = set()

        for cur_ent in cur_entities:
            cur_ent_hyperedges = await knowledge_graph_inst.get_entity_hyperedges(cur_ent, get_synonyms=query_param.get_synonyms)
            cur_ent_hyperedges = await _filter_l0_hyperedge_names(cur_ent_hyperedges or [], knowledge_graph_inst)
            for he in cur_ent_hyperedges:
                if he != cur_he_name:
                    next_he_set.add(he)


        for next_he_name in next_he_set:
            if upperbound_graph.get_hyperedge_id(next_he_name) > -1:
                continue
            queue.append((next_he_name, cur_depth + 1))
            await upperbound_graph.add_hyperedge(next_he_name)

    return upperbound_graph


async def plan_initialisation(
        query: str,
        topic_entities: list[str],
        plan_context_graph: Hypergraph,
        global_config: dict,
        query_param: QueryParam
) -> list[ReasoningDAG]:

    start = time.time()
    topic_entity_names = [t.upper() for t in topic_entities]


    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
    
    
    token_usage_dict = defaultdict(int)
    init_dags = []

    if not query_param.with_planning:
        orig_query_plan = f"""<subquestions>(0{tuple_delimiter}{query}{tuple_delimiter}{','.join(topic_entity_names)}){record_delimiter}</subquestions><dag></dag>"""
        logger.debug(f"Original query plan: {orig_query_plan}")
        init_dags = [ReasoningDAG(orig_question=query, plan_llm_answer=orig_query_plan, global_topic_entity_names=topic_entity_names)]
        return init_dags, token_usage_dict

    
    use_model_func = global_config["llm_model_func"]
    plan_initialisation_prompt_temp = PROMPTS["plan_initialisation"]
    plan_initialisation_prompt_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )
    plan_context = ""
    if query_param.with_plan_context:
        plan_context = plan_context_graph.print_hyperedges()

    plan_initialisation_prompt = plan_initialisation_prompt_temp.format(
        **plan_initialisation_prompt_base, 
        input_question=query,
        topic_entities=", ".join(topic_entity_names),
        plan_context=plan_context
    )


    max_time = 200  

    while len(init_dags) < query_param.nb_of_init_plans and (time.time() - start) < max_time:

        try:
            plan_initialisation_result, token_usage = await use_model_func(plan_initialisation_prompt, count_token=True)
            
            token_usage_dict["plan_init"] = token_usage["total_tokens"]

            logger.info("Plan initialisation LLM result:")
            logger.info(f"{plan_initialisation_result}")

            init_dag = ReasoningDAG(orig_question=query, plan_llm_answer=plan_initialisation_result, global_topic_entity_names=topic_entity_names)
            init_dags.insert(0, init_dag)
        except ValueError as e:
            logger.warning(f"Failed to parse reasoning DAG with {e}: {plan_initialisation_result}")

    if not init_dags:
        orig_query_plan = f"""<subquestions>(0{tuple_delimiter}{query}{tuple_delimiter}{','.join(topic_entity_names)}){record_delimiter}</subquestions><dag></dag>"""
        logger.warning(f"Fallback to original query plan: {orig_query_plan}")
        init_dags = [ReasoningDAG(orig_question=query, plan_llm_answer=orig_query_plan, global_topic_entity_names=topic_entity_names)]
    return init_dags, token_usage_dict

class AnsweringAgent:
    def __init__(self, upperbound_graph: Hypergraph, dags: ReasoningDAG, global_config: dict, text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam):
        self.upperbound_graph = upperbound_graph
        self.dags = dags
        self.global_config = global_config
        self.text_chunks_db = text_chunks_db
        self.query_param = query_param
        self.use_model_func = global_config["llm_model_func"]
        self.tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        self.completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

        self.orig_question = dags[0].orig_question
        self.answer_dicts = []
        self.token_usage = defaultdict(int)

    

    async def extract_context_from_dags(self, dag: ReasoningDAG) -> str:


        context = []
        for i, level in enumerate(dag.levels):  
            for node_id in level:
                subq_data = dag.subquestions[node_id]
                path = dag.subquestions[node_id]['path']
                he_names = [self.upperbound_graph.get_hyperedge_name(he_id) for he_id in path]
                subq_data['reasoning_path'] = ' -> '.join(he_names)

                
                entity_descriptions = []
                ent_ids = set()
                for he_id in path:
                    he_entity_ids = self.upperbound_graph.get_hyperedge_entities_id(he_id)
                    ent_ids.update(he_entity_ids)
                entity_descriptions = [self.upperbound_graph.get_entity_description(ent_id) for ent_id in ent_ids]
                subq_data['entity_descriptions'] = '\n'.join(entity_descriptions)


                subq_data['src_text_chunks'] = ""
                if self.query_param.with_src_chunks:
                    he_names = []
                    for he_id in path:

                        he_name = self.upperbound_graph.get_hyperedge_name(he_id)
                        he_names.append(he_name)
                    
                    hyperedge_datas = [await self.upperbound_graph.base.get_hyperedge(he_name) for he_name in he_names]

                    text_unit_ids = []
                    for hd in hyperedge_datas:
                        units = split_string_by_multi_markers(hd["source_id"], [GRAPH_FIELD_SEP])
                        for unit in units:
                            if unit not in text_unit_ids:
                                text_unit_ids.append(unit)
                    
                    text_contexts = []

                    for unit_id in text_unit_ids:
                        chunk_data = await self.text_chunks_db.get_by_id(unit_id)
                        if chunk_data is not None and "content" in chunk_data:
                            text_contexts.append(chunk_data["content"])
                    
                    subq_data['src_text_chunks'] = '\n'.join(text_contexts)
                context.append(subq_data)

        return context


    def format_context_from_dag(self, context: list[dict]) -> str:
        context_strs = []
        for subq_data in context:
            subq_str = ""
            if self.query_param.subquestion_guided:
                subq_str += f"Subquestion{subq_data['id']}: {subq_data['subquestion']}\n"
                subq_str += f"Answer: {subq_data['answer']}\n"

            subq_str += f"Reasoning Path: {subq_data['reasoning_path']}\n"

            if self.query_param.with_src_chunks:
                subq_str += f"Source Text Chunks: {subq_data['src_text_chunks']}\n"
           
            subq_str += f"Entity Descriptions:\n{subq_data['entity_descriptions']}\n"
            context_strs.append(subq_str)
        full_context = "\n".join(context_strs)
        return full_context
    
    async def get_answers(self) -> list[dict]:

        if self.query_param.subquestion_guided:
            final_answer_prompt_temp = PROMPTS["final_answer_generation_guided"]
        else:
            final_answer_prompt_temp = PROMPTS["final_answer_generation"]
        final_answer_prompt_base = dict(
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
        )

        
        logger.debug(f"Starting to get answers for {len(self.dags)} DAGs, timestamp: {time.time()}")
        for dag in self.dags:
            context = await self.extract_context_from_dags(dag)
            logger.debug(f"Extracted context for DAG, timestamp: {time.time()}")
            full_context = self.format_context_from_dag(context)
            logger.debug(f"format context for DAG, timestamp: {time.time()}")
            final_answer_prompt = final_answer_prompt_temp.format(**final_answer_prompt_base, question=dag.orig_question, context=full_context)
            llm_answer, token_usage = await self.use_model_func(final_answer_prompt, count_token=True)
            self.token_usage["final_answer"] += token_usage["total_tokens"]

            logger.debug(f"Final answer LLM answer:\n{llm_answer}")
            logger.debug(f"Generated answer for DAG, timestamp: {time.time()}")
            gen_answer = llm_answer.split("<answer>")[1].split("</answer>")[0].strip()
            self.answer_dicts.append(
                {"gen_answer": gen_answer, "generation": llm_answer, "dag": dag.print_dag(), "retrieved": context}
            )
            logger.debug(f"Appended answer dict, timestamp: {time.time()}")

    

    async def llm_select_final_answer(self) -> dict:
        answer_selection_prompt_temp = PROMPTS["final_answer_selection"]

        answer_selection_prompt_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

        answer_records = []
        for i, answer_dict in enumerate(self.answer_dicts):
            answer = answer_dict["gen_answer"]
            context = answer_dict["retrieved"]
            path_strs = [subq_data['reasoning_path'] for subq_data in context]
            context_str = "\n".join(path_strs)
            answer_records.append(f"Answer {i}:\n{answer}\nContext {i}: {context_str}")


        answer_selection_prompt = answer_selection_prompt_temp.format(
            **answer_selection_prompt_base, 
            answers="\n".join(answer_records), 
            question=self.orig_question
        )

        llm_answer, token_usage = await self.use_model_func(answer_selection_prompt, count_token=True)
        self.token_usage["final_answer_select"] += token_usage["total_tokens"]
        logger.debug(f"Final answer selection LLM response:\n{llm_answer}")

   
        id_section = llm_answer.split("<id>")[1].split("</id>")[0].strip()
        id = parse_llm_result_into_lists(
            content=id_section, 
            tuple_delimiter=self.tuple_delimiter, 
            record_delimiter=self.record_delimiter, 
            completion_delimiter=self.completion_delimiter
            )
        id = id[0][0]
        id = int(id.strip())

        logger.debug(f"Select final answer: {answer_records[id]}")

    
        return self.answer_dicts[id]



    async def get_best_answer(self) -> dict:
        if not self.answer_dicts:
            await self.get_answers()
        if len(self.answer_dicts) == 1:
            return self.answer_dicts[0]
        logger.debug(f"Multiple answers generated: {len(self.answer_dicts)}. Starting selection process, timestamp: {time.time()}")
        best_answer_dict = await self.llm_select_final_answer()
        logger.debug(f"Selected best answer, timestamp: {time.time()}")
        if not best_answer_dict:
            return self.answer_dicts[0]
        return best_answer_dict

    




async def kg_query_with_reasoning(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_names_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    
    fail_answer_dict = {"gen_answer": "", "generation": PROMPTS["fail_response"], "dag": "", "retrieved": []}

    token_usage = defaultdict(int)
        
    topic_entities, ti_token_usage = await topic_initialisation(query=query, entity_names_vdb=entity_names_vdb, global_config=global_config)
    token_usage.update(ti_token_usage)
    logger.debug(f"Topic entities: {topic_entities}")

    if not topic_entities:
        logger.warning("No key entities found in the topic initialisation result.")
        return fail_answer_dict
    

    target_hyperedges = []
    if query_param.with_target_hyperedges:
        target_hyperedges = await target_hyperedge_matching(query=query, knowledge_graph_inst=knowledge_graph_inst, hyperedges_vdb=hyperedges_vdb)
        if not target_hyperedges:
            logger.warning("No target hyperedges matched for the query.")

    logger.debug(f"Target hyperedges: {target_hyperedges}")

    logger.debug("Constructing plan context graph...")

    plan_context_graph = await plan_context_graph_construction(
        knowledge_graph_inst=knowledge_graph_inst,
        entity_names_vdb = entity_names_vdb,
        entities_vdb = entities_vdb,
        hyperedges_vdb = hyperedges_vdb,
        global_config=global_config,
        query_param = query_param,
        query=query,
        topic_entities=topic_entities,
        target_hyperedges=target_hyperedges,
    )
    logger.debug("Plan context graph constructed.")




    init_dags, id_token_usage = await plan_initialisation(
        query=query,
        topic_entities=topic_entities,
        plan_context_graph=plan_context_graph,
        global_config=global_config,
        query_param = query_param
    )
        
    token_usage.update(id_token_usage)


    if not init_dags:
        logger.warning("Failed to generate a valid reasoning DAG.")
        return fail_answer_dict
    
    logger.debug("Initial reasoning DAGs:")
    for i, init_dag in enumerate(init_dags):
        logger.debug(f"Initial reasoning DAG {i}:") 
        logger.debug(init_dag.print_dag())


    
    upperbound_graph = await upperbound_graph_construction(
        knowledge_graph_inst=knowledge_graph_inst,
        entity_names_vdb = entity_names_vdb,
        entities_vdb = entities_vdb,
        hyperedges_vdb = hyperedges_vdb,
        query_param = query_param,
        topic_entities=topic_entities,
        target_hyperedges=target_hyperedges
    )


    rs_agent = ReasoningDAGAgent(upperbound_graph=upperbound_graph, dags=init_dags, global_config=global_config, query_param=query_param, text_chunks_db=text_chunks_db)
    finished_dags, tree_record = await rs_agent.reason()

    token_usage.update(rs_agent.token_usage)


    if not finished_dags:
        logger.info("Cannot finish reasoning.")
        return fail_answer_dict
    logger.info(f"Finished {len(finished_dags)} reasoning DAG(s).")

    
    as_agent = AnsweringAgent(upperbound_graph=upperbound_graph, dags=finished_dags, global_config=global_config, query_param=query_param, text_chunks_db=text_chunks_db)
    final_answer_dict = await as_agent.get_best_answer()

    token_usage.update(as_agent.token_usage)
    logger.info("Selected Final Answer DAG:")
    logger.info(f"Answer: {final_answer_dict['gen_answer']}")
    logger.info(f"Reason: {final_answer_dict['generation']}")
    logger.info(final_answer_dict["dag"])

    final_answer_dict["tree_record"] = tree_record
    final_answer_dict["token_usage"] = token_usage
    final_answer_dict["graph_search_depth_stats"] = rs_agent.graph_search_depth_stats



    return final_answer_dict






    
