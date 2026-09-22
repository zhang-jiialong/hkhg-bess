#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval.eval import cal_em, cal_f1
from eval_v2_mechanism_chain_aware import (
    gen_dimension_averages,
    load_questions_any,
    maybe_cal_gen_for_eval,
    metric_average,
    question_answers,
    question_context,
    question_text,
)
from hh_iterative_eval_runner import build_answer_prompt
from hh_iterative_l2_path_beam_rerank import build_context as build_standard_context
from hh_iterative_l2_path_beam_rerank import resolve_config
from layer_answer_compare import ask_llm
from strategy_eval_compare import maybe_cal_rsim, save_results


def load_experiment_defaults(path: str, section: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    values = payload.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"{section!r} in {config_path} must be a JSON object")
    return values


def parse_args() -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--experiment-config", default="configs/default.json")
    known, _ = probe.parse_known_args()
    defaults = load_experiment_defaults(known.experiment_config, "generation")

    parser = argparse.ArgumentParser(
        description="Generate <think>/<answer> outputs with one HKHG final-answer-generation call."
    )
    parser.add_argument("--experiment-config", default=known.experiment_config)
    parser.add_argument("--source", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="runtime_config.json")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--mode", choices=["long", "short"], required=True)
    parser.add_argument("--answer-language", required=True)
    parser.add_argument("--top-k-evidence", type=int, default=defaults.get("top_k_evidence", 20))
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 8))
    parser.add_argument("--gen-workers", type=int, default=defaults.get("gen_workers", 8))
    parser.add_argument("--timeout", type=float, default=defaults.get("timeout", 180.0))
    parser.add_argument("--skip-rsim", action="store_true", default=bool(defaults.get("skip_rsim", False)))
    parser.add_argument("--skip-gen", action="store_true", default=bool(defaults.get("skip_gen", False)))
    parser.add_argument("--gen-llm-only", action="store_true", default=bool(defaults.get("gen_llm_only", False)))
    parser.add_argument("--resume", action="store_true", default=bool(defaults.get("resume", False)))
    return parser.parse_args()


def load_retrieval_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    rows = [row for row in (rows or []) if isinstance(row, dict)]
    rows.sort(key=lambda row: int(row.get("question_index", 0)))
    return rows


def topk_retrieval_row(row: dict[str, Any], top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieval = dict(row.get("retrieval") or {})
    top = retrieval.get("top_l0_hyperedges") or retrieval.get("selected_hyperedges") or []
    top = [item for item in top if isinstance(item, dict)]
    if top_k > 0:
        top = top[:top_k]
    retrieval["top_l0_hyperedges"] = top
    retrieval["selected_hyperedges"] = top
    out = dict(row)
    out["retrieval"] = retrieval
    return out, top


def avg_selected(rows: list[dict[str, Any]]) -> float | None:
    vals = []
    for row in rows:
        v = (row.get("retrieval_summary") or {}).get("selected_count")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return round(statistics.mean(vals), 2) if vals else None


def write_summary(payload: dict[str, Any], output_dir: Path, mode: str) -> None:
    avg = payload.get("averages", {})
    dim = payload.get("gen_dimension_averages", {})
    lines = [
        f"# Existing Prompt Think/Answer {mode} Results",
        "",
        "| done | EM | F1 | R-S | G-E | selected_count |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(payload.get('questions', []))} | {avg.get('em')} | {avg.get('f1')} | {avg.get('rsim')} | {avg.get('gen')} | {avg.get('selected_count')} |",
        "",
        "## G-E Dimension Averages",
        "",
        json.dumps(dim, ensure_ascii=False, indent=2),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "existing_prompts_think_answer_results.json"

    client, _embedding_model, llm_model, _base_url, _api_key = resolve_config(args)
    retrieval_rows = load_retrieval_rows(source)
    question_rows = load_questions_any(Path(args.questions))
    by_idx = {int(row.get("question_index", i)): row for i, row in enumerate(retrieval_rows)}
    indices = sorted(by_idx)

    if args.resume and output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    else:
        payload = {
            "settings": {
                "source": str(source),
                "questions": str(args.questions),
                "mode": args.mode,
                "answer_language": args.answer_language,
                "prompt_style": "one HKHG final_answer_generation call for <reasoning> and <answer>",
                "top_k_evidence": args.top_k_evidence,
                "llm_model": llm_model,
                "workers": max(1, args.workers),
                "gen_workers": max(1, args.gen_workers),
            },
            "questions": [],
            "averages": {},
            "gen_dimension_averages": {},
        }
    save_results(output_json, payload)
    completed = {int(row["question_index"]) for row in payload.get("questions", [])}
    def run_one(pos: int, idx: int) -> dict[str, Any] | None:
        if idx in completed:
            print(f"[{pos}/{len(indices)}] skip completed Q{idx}", flush=True)
            return None
        raw = question_rows[idx]
        retrieval_row, top_evidence = topk_retrieval_row(by_idx[idx], args.top_k_evidence)
        retrieval = retrieval_row.get("retrieval") or {}
        question = question_text(raw) or str(retrieval_row.get("question") or "")
        golden_answers = question_answers(raw)
        gold_context = question_context(raw)
        print(f"[{pos}/{len(indices)}] existing-prompts-{args.mode} Q{idx}: {question[:100]}", flush=True)

        t0 = time.perf_counter()
        standard_context = build_standard_context(top_evidence)
        short_prompt = build_answer_prompt(
            question,
            standard_context,
            args.mode,
            answer_language=args.answer_language,
        )
        short_data = ask_llm(client, llm_model, short_prompt)
        short_reasoning = str(short_data.get("reasoning") or "").strip()
        short_answer = str(short_data.get("answer") or "").strip()
        short_raw_generation = str(short_data.get("raw_generation") or "").strip()
        generation = f"<think>\n{short_reasoning}\n</think>\n<answer>\n{short_answer}\n</answer>"

        scores = {"em": None, "f1": None, "rsim": None, "gen": None}
        score_errors = {"rsim": None, "gen": None}
        gen_details = None
        if args.mode == "short":
            scores["em"] = float(cal_em([golden_answers], [short_answer]))
            scores["f1"] = float(cal_f1([golden_answers], [short_answer]))
            rsim, rsim_error = maybe_cal_rsim(gold_context, standard_context, args.skip_rsim)
            gen, gen_error, gen_details = maybe_cal_gen_for_eval(
                question,
                golden_answers,
                generation,
                float(scores["f1"]),
                args.skip_gen,
                args.gen_llm_only,
            )
            scores["rsim"] = rsim
            scores["gen"] = gen
            score_errors = {"rsim": rsim_error, "gen": gen_error}

        return {
            "question_index": idx,
            "question": question,
            "golden_answers": golden_answers,
            "answer": generation,
            "reasoning": short_reasoning,
            "short_answer": short_answer,
            "short_raw_generation": short_raw_generation,
            "generation": generation,
            "scores": scores,
            "score_errors": score_errors,
            "gen_details": gen_details,
            "timings": {"total_seconds": round(time.perf_counter() - t0, 4)},
            "retrieval_summary": {
                **(retrieval_row.get("retrieval_summary") or {}),
                "selected_count": len(top_evidence),
                "top_k_evidence": args.top_k_evidence,
            },
            "context_meta": {"standard_context_chars": len(standard_context)},
            "retrieval": retrieval,
        }

    def safe_run_one(pos: int, idx: int) -> dict[str, Any] | None:
        try:
            return run_one(pos, idx)
        except Exception as exc:
            print(f"  [error] question_index={idx} failed: {type(exc).__name__}: {exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(safe_run_one, pos, idx) for pos, idx in enumerate(indices, start=1) if idx not in completed]
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            payload["questions"].append(row)
            payload["questions"].sort(key=lambda item: int(item["question_index"]))
            rows = payload["questions"]
            payload["averages"] = {
                "em": metric_average(rows, "em"),
                "f1": metric_average(rows, "f1"),
                "rsim": metric_average(rows, "rsim"),
                "gen": metric_average(rows, "gen"),
                "selected_count": avg_selected(rows),
            }
            payload["gen_dimension_averages"] = gen_dimension_averages(rows)
            save_results(output_json, payload)
            write_summary(payload, output_dir, args.mode)
            print(f"  saved -> {output_json} ({len(rows)}/{len(indices)})", flush=True)

    write_summary(payload, output_dir, args.mode)
    print(json.dumps({"output_json": str(output_json), "done": len(payload.get("questions", [])), "averages": payload.get("averages", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
