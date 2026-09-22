#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

PROMPT = """You are a technical assistant answering questions using retrieved evidence.

Generate a clear, evidence-grounded long-form answer to the user's question.

Use the organized evidence below. It has three parts:
1. Core evidence: facts most directly related to the question.
2. Supporting evidence: additional facts that may clarify mechanisms, conditions, or implications.
3. Selected source sentences: compact original-source sentences selected for relevance.

Requirements:
- Ground the answer in the provided evidence.
- Prioritize the core evidence; use supporting evidence only when it helps answer the question.
- Explain the mechanism, conditions, consequences, and practical implications when they are relevant.
- Keep the answer readable and concise: usually 3-6 short paragraphs, with short markdown headings only if useful.
- Prefer concrete mechanisms, quantities, conditions, and named components from the evidence.
- Avoid broad textbook background unless it directly helps answer the question.
- Avoid unsupported claims and avoid inventing details not supported by the evidence.
- If evidence is insufficient for part of the question, state the limitation briefly.

Organized evidence:
{context}
"""

STOPWORDS = {
    'the','a','an','and','or','of','to','in','on','for','with','by','as','is','are','was','were','be','been','being',
    'this','that','these','those','it','its','their','they','them','from','at','into','during','when','why','how','what',
    'which','can','may','must','should','would','could','about','than','then','also','not','more','less','such','using',
    'use','used','cell','cells','battery','batteries','bms','li','ion','pack','system','systems'
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='results/eval_g_bms_long_v2_expansion_variants_gpt4omini_fresh_20260528/semantic_weak_structure/eval_g_results.json')
    p.add_argument('--output-dir', default='results/bms_long_v2_semantic_weak_structure_compact_synthesis_prompt_20260601')
    p.add_argument('--config', default='runtime_config.json')
    p.add_argument('--concurrency', type=int, default=12)
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--core-n', type=int, default=8)
    p.add_argument('--support-n', type=int, default=8)
    p.add_argument('--mechanism-top-m', type=int, default=50)
    p.add_argument('--mechanism-final-k', type=int, default=0, help='0 means core_n + support_n')
    p.add_argument('--mechanism-alpha', type=float, default=0.55)
    p.add_argument('--mechanism-beta', type=float, default=0.20)
    p.add_argument('--mechanism-gamma', type=float, default=0.20)
    p.add_argument('--mechanism-delta', type=float, default=0.05)
    p.add_argument('--skip-final-evidence-selection', action='store_true')
    p.add_argument('--source-sentences-per-evidence', type=int, default=2)
    p.add_argument('--max-source-total-chars', type=int, default=4500)
    p.add_argument('--max-context-chars', type=int, default=9000)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    x = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(x, list):
        return x
    for key in ('questions', 'results', 'data', 'items'):
        v = x.get(key)
        if isinstance(v, list):
            return v
    raise ValueError(f'cannot load rows from {path}')


def load_config(path: Path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    llm = cfg['llm']
    return llm['base_url'], llm['api_key'], llm['model']


def words(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?", str(text).lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def normalize_phrase(text: str) -> str:
    text = re.sub(r'[^a-z0-9+\-/ ]+', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip(' -/')


def normalize_mechanism(value: Any) -> str:
    if isinstance(value, dict):
        parts = [
            str(value.get(key) or '').strip()
            for key in ('condition', 'cause', 'process', 'effect')
            if str(value.get(key) or '').strip()
        ]
        raw = ' -> '.join(parts) if parts else str(value.get('mechanism') or value.get('relation') or '')
    else:
        raw = str(value or '')
    raw = re.sub(r'\s+', ' ', raw).strip()
    if not raw:
        return ''
    return raw[:320]


def evidence_text(item: dict[str, Any]) -> str:
    return ' '.join(
        str(item.get(key) or '')
        for key in ('fact_core', 'text', 'source_text')
        if item.get(key)
    )


def evidence_id(item: dict[str, Any], fallback: int = 0) -> str:
    return str(item.get('id') or item.get('hyperedge_id') or item.get('chunk_id') or f'evidence_{fallback}')


def reranker_score(item: dict[str, Any]) -> float:
    for key in ('reranker_score', 'final_score', 'score'):
        try:
            return float(item.get(key))
        except (TypeError, ValueError):
            pass
    return 0.0


def source_id(item: dict[str, Any]) -> str:
    for key in ('source_id', 'chunk_id', 'doc_id'):
        value = item.get(key)
        if value:
            return str(value)
    source = str(item.get('source_text') or '')
    if source:
        return 'source:' + hashlib.md5(source[:800].encode('utf-8', errors='ignore')).hexdigest()[:12]
    return ''


def vector_from_embedding(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def lexical_vector(item: dict[str, Any]) -> Counter:
    return Counter(words(evidence_text(item)))


def cosine_dense(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def evidence_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    avec = vector_from_embedding(a.get('embedding'))
    bvec = vector_from_embedding(b.get('embedding'))
    if avec is not None and bvec is not None:
        return max(0.0, min(1.0, cosine_dense(avec, bvec)))
    return max(0.0, min(1.0, cosine_counter(lexical_vector(a), lexical_vector(b))))


def novelty(item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 1.0
    return 1.0 - max(evidence_similarity(item, other) for other in selected)


def evidence_mechanisms(item: dict[str, Any]) -> set[str]:
    raw = item.get('_mechanisms')
    if raw is None:
        raw = item.get('mechanisms')
    if raw is None:
        raw = item.get('mechanism_relations')
    if not isinstance(raw, list):
        return set()
    out = {normalize_mechanism(value) for value in raw}
    return {value for value in out if value}


def mechanism_coverage_gain(item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    mechanisms = evidence_mechanisms(item)
    if not mechanisms:
        return 0.0
    covered: set[str] = set()
    for other in selected:
        covered.update(evidence_mechanisms(other))
    new_mechanisms = mechanisms - covered
    return len(new_mechanisms) / max(1, len(mechanisms))


def source_redundancy_penalty(item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    sid = source_id(item)
    if not sid or not selected:
        return 0.0
    same = sum(1 for other in selected if source_id(other) == sid)
    return same / max(1, len(selected))


def normalize_reranker_scores(pool: list[dict[str, Any]]) -> dict[str, float]:
    raw = [reranker_score(item) for item in pool]
    lo = min(raw) if raw else 0.0
    hi = max(raw) if raw else 0.0
    out = {}
    for idx, item in enumerate(pool):
        eid = evidence_id(item, idx)
        if hi > lo:
            out[eid] = (reranker_score(item) - lo) / (hi - lo)
        else:
            out[eid] = 1.0 if raw else 0.0
    return out


def mechanism_chain_aware_select(
    candidates: list[dict[str, Any]],
    top_m: int = 50,
    final_k: int = 8,
    alpha: float = 0.70,
    beta: float = 0.20,
    gamma: float = 0.08,
    delta: float = 0.02,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [item for item in candidates if isinstance(item, dict)]
    sorted_valid = sorted(valid, key=reranker_score, reverse=True)
    candidate_pool = sorted_valid if top_m <= 0 else sorted_valid[:max(1, top_m)]
    if not candidate_pool or final_k <= 0:
        return [], {
            'strategy': 'mechanism_chain_aware_selection',
            'candidate_count': len(valid),
            'candidate_pool_count': len(candidate_pool),
            'selected_count': 0,
        }

    norm_scores = normalize_reranker_scores(candidate_pool)
    selected = [candidate_pool[0]]
    remaining = candidate_pool[1:]
    debug = []

    first = selected[0]
    first_rel = norm_scores.get(evidence_id(first, 0), 0.0)
    first_gain_raw = mechanism_coverage_gain(first, [])
    debug.append(
        {
            'id': evidence_id(first, 0),
            'reranker_score': reranker_score(first),
            'reranker_score_normalized': first_rel,
            'novelty': 1.0,
            'mechanism_coverage_gain_raw': first_gain_raw,
            'mechanism_coverage_gain': first_gain_raw * first_rel,
            'source_penalty': 0.0,
            'final_score': None,
            'mechanisms': sorted(evidence_mechanisms(first)),
            'selection_step': 1,
            'selection_reason': 'highest_reranker_score',
        }
    )

    while remaining and len(selected) < final_k:
        best_item = None
        best_debug = None
        best_score = -float('inf')
        for item in remaining:
            eid = evidence_id(item)
            rel = norm_scores.get(eid, 0.0)
            nov = novelty(item, selected)
            gain_raw = mechanism_coverage_gain(item, selected)
            gain = gain_raw * rel
            penalty = source_redundancy_penalty(item, selected)
            final_score = alpha * rel + beta * nov + gamma * gain - delta * penalty
            if final_score > best_score:
                best_score = final_score
                best_item = item
                best_debug = {
                    'id': eid,
                    'reranker_score': reranker_score(item),
                    'reranker_score_normalized': rel,
                    'novelty': nov,
                    'mechanism_coverage_gain_raw': gain_raw,
                    'mechanism_coverage_gain': gain,
                    'source_penalty': penalty,
                    'final_score': final_score,
                    'mechanisms': sorted(evidence_mechanisms(item)),
                    'selection_step': len(selected) + 1,
                }
        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)
        debug.append(best_debug or {})

    meta = {
        'strategy': 'mechanism_chain_aware_selection',
        'candidate_count': len(valid),
        'candidate_pool_count': len(candidate_pool),
        'selected_count': len(selected),
        'top_m': top_m,
        'final_k': final_k,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'selected_evidence': debug,
    }
    return selected, meta


def split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?;])\s+', text)
    out = []
    for s in parts:
        s = s.strip()
        if 40 <= len(s) <= 420:
            out.append(s)
    return out


def score_sentence(sentence: str, query_terms: Counter, evidence_terms: Counter) -> float:
    sw = Counter(words(sentence))
    if not sw:
        return 0.0
    score = 0.0
    for t, c in sw.items():
        score += min(c, 2) * (3.0 * query_terms.get(t, 0) + 1.0 * evidence_terms.get(t, 0))
    # mild preference for information-dense but not huge sentences
    score = score / math.sqrt(max(len(sw), 1))
    if re.search(r'\d|%|V|A|SOC|DOD|OCV|CAN|RS232|CCCV', sentence):
        score += 1.5
    return score


def select_source_sentences(question: str, evidence_text: str, source_text: str, n: int) -> list[str]:
    query_terms = Counter(words(question))
    evidence_terms = Counter(words(evidence_text))
    candidates = split_sentences(source_text)
    ranked = sorted(candidates, key=lambda s: score_sentence(s, query_terms, evidence_terms), reverse=True)
    selected = []
    seen_norm = set()
    for s in ranked:
        norm = re.sub(r'\W+', ' ', s.lower()).strip()[:180]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        selected.append(s)
        if len(selected) >= n:
            break
    return selected


def clamp(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + '\n...[truncated]'


def build_context(row: dict[str, Any], args) -> tuple[str, dict[str, Any]]:
    question = str(row.get('question') or '')
    retrieval = row.get('retrieval') or {}
    top = retrieval.get('top_l0_hyperedges') or retrieval.get('selected_hyperedges') or []
    top = [x for x in top if isinstance(x, dict)]
    final_k = args.mechanism_final_k or (args.core_n + args.support_n)
    if args.skip_final_evidence_selection:
        selection_meta = {
            'strategy': 'skip_final_evidence_selection_use_input_order',
            'candidate_count': len(top),
            'candidate_pool_count': len(top),
            'selected_count': len(top),
            'top_m': len(top),
            'final_k': len(top),
        }
    else:
        top, selection_meta = mechanism_chain_aware_select(
            top,
            top_m=args.mechanism_top_m,
            final_k=final_k,
            alpha=args.mechanism_alpha,
            beta=args.mechanism_beta,
            gamma=args.mechanism_gamma,
            delta=args.mechanism_delta,
        )

    core = top[:args.core_n]
    support = top[args.core_n:] if args.skip_final_evidence_selection else top[args.core_n:args.core_n + args.support_n]

    lines = ['## Core evidence']
    for i, item in enumerate(core, start=1):
        hid = str(item.get('hyperedge_id') or f'core_{i}')
        text = re.sub(r'\s+', ' ', str(item.get('text') or '')).strip()
        lines.append(f'{i}. {hid}\n{text}')

    lines.append('\n## Supporting evidence')
    for i, item in enumerate(support, start=1):
        hid = str(item.get('hyperedge_id') or f'support_{i}')
        text = re.sub(r'\s+', ' ', str(item.get('text') or '')).strip()
        lines.append(f'{i}. {hid}\n{text}')

    source_blocks = []
    used = 0
    seen_sentence = set()
    for item in core + support:
        evidence_text = str(item.get('text') or '')
        source_text = str(item.get('source_text') or '')
        hid = str(item.get('hyperedge_id') or '')
        selected = select_source_sentences(question, evidence_text, source_text, args.source_sentences_per_evidence)
        kept = []
        for sent in selected:
            norm = re.sub(r'\W+', ' ', sent.lower()).strip()[:220]
            if norm in seen_sentence:
                continue
            seen_sentence.add(norm)
            kept.append(sent)
        if not kept:
            continue
        block = f'- {hid}: ' + ' '.join(kept)
        if used + len(block) > args.max_source_total_chars:
            break
        source_blocks.append(block)
        used += len(block)

    lines.append('\n## Selected source sentences')
    lines.extend(source_blocks if source_blocks else ['No additional source sentences selected.'])

    context = clamp('\n\n'.join(lines), args.max_context_chars)
    meta = {
        'core_count': len(core),
        'support_count': len(support),
        'source_sentence_blocks': len(source_blocks),
        'context_chars': len(context),
        'selection': selection_meta,
    }
    return context, meta


async def call_one(client: AsyncOpenAI, model: str, question: str, context: str, timeout: float):
    system_prompt = PROMPT.format(context=context)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question},
        ],
        temperature=0,
        timeout=timeout,
    )
    return response.choices[0].message.content or ''


async def main():
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'compact_synthesis_results.jsonl'
    json_path = out_dir / 'compact_synthesis_results.json'
    err_path = out_dir / 'compact_synthesis_errors.jsonl'

    rows = load_rows(source)
    completed = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    completed.add(int(json.loads(line)['idx']))
                except Exception:
                    pass

    base_url, api_key, model = load_config(Path(args.config))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)
    print(f'[config] rows={len(rows)} completed={len(completed)} model={model} concurrency={args.concurrency}', flush=True)

    async def run_one(pos: int, row: dict[str, Any]):
        idx = int(row.get('question_index', row.get('idx', pos - 1)))
        if idx in completed:
            return None
        async with sem:
            question = str(row.get('question') or '')
            context, meta = build_context(row, args)
            start = time.time()
            try:
                print(f'[gen] {pos}/{len(rows)} idx={idx} {question[:100]}', flush=True)
                result = await call_one(client, model, question, context, args.timeout)
                return True, {
                    'idx': idx,
                    'question_index': idx,
                    'question': question,
                    'result': result,
                    'answer': result,
                    'reference_answer': row.get('golden_answers'),
                    'source_result_path': str(source),
                    'prompt_style': 'ES-HiR2 mechanism-chain-aware compact evidence synthesis prompt',
                    'selection_strategy': 'mechanism_chain_aware_selection',
                    'retrieval_summary': row.get('retrieval_summary') or row.get('path_summary'),
                    'context_meta': meta,
                    'elapsed_sec': round(time.time() - start, 3),
                }
            except Exception as exc:
                return False, {'idx': idx, 'question': question, 'error': repr(exc), 'elapsed_sec': round(time.time() - start, 3)}

    tasks = [asyncio.create_task(run_one(pos, row)) for pos, row in enumerate(rows, start=1) if int(row.get('question_index', row.get('idx', pos - 1))) not in completed]
    done = len(completed)
    with jsonl_path.open('a', encoding='utf-8') as out, err_path.open('a', encoding='utf-8') as err:
        for task in asyncio.as_completed(tasks):
            res = await task
            if res is None:
                continue
            ok, item = res
            done += 1
            if ok:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                out.flush()
                print(f'[saved] {done}/{len(rows)} idx={item["idx"]} len={len(item["result"])} ctx={item["context_meta"]["context_chars"]}', flush=True)
            else:
                err.write(json.dumps(item, ensure_ascii=False) + '\n')
                err.flush()
                print(f'[error] {done}/{len(rows)} idx={item["idx"]} {item["error"]}', flush=True)

    final_rows = []
    if jsonl_path.exists():
        final_rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        final_rows.sort(key=lambda x: int(x['idx']))
    json_path.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    lens = [len(str(r.get('result',''))) for r in final_rows]
    ctx_lens = [int((r.get('context_meta') or {}).get('context_chars') or 0) for r in final_rows]
    summary = {
        'count': len(final_rows),
        'mean_answer_len': sum(lens)/len(lens) if lens else None,
        'min_answer_len': min(lens) if lens else None,
        'max_answer_len': max(lens) if lens else None,
        'mean_context_len': sum(ctx_lens)/len(ctx_lens) if ctx_lens else None,
        'source': str(source),
        'output': str(json_path),
    }
    (out_dir / 'generation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
