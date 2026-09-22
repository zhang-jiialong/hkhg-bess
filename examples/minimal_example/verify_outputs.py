#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "minimal_example"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


short_result = read_json(
    EXAMPLE / "output" / "generation_short" / "existing_prompts_think_answer_results.json"
)
long_result = read_json(
    EXAMPLE / "output" / "generation_long" / "existing_prompts_think_answer_results.json"
)
short_retrieval = read_json(
    EXAMPLE / "output" / "retrieval_short" / "selection_retrieval_results.json"
)
long_retrieval = read_json(
    EXAMPLE / "output" / "retrieval_long" / "selection_retrieval_results.json"
)
assert len(short_retrieval.get("questions", [])) == 1
assert len(long_retrieval.get("questions", [])) == 1
assert len(short_result.get("questions", [])) == 1
assert len(long_result.get("questions", [])) == 1
assert short_result["questions"][0].get("short_answer")
assert long_result["questions"][0].get("short_answer")
assert not short_result["questions"][0].get("score_errors", {}).get("gen")

gen_details = short_result["questions"][0].get("gen_details") or {}
for item in (gen_details.get("explanation") or {}).values():
    assert "Defaulted to score=5" not in str(item.get("explanation", ""))

summary = {
    "graph_constructed": (ROOT / "data" / "work" / "minimal_example" / "graph_chunk_entity_relation.graphml").exists(),
    "short_retrieval_completed": len(short_retrieval["questions"]) == 1,
    "long_retrieval_completed": len(long_retrieval["questions"]) == 1,
    "short_generation_completed": True,
    "long_generation_completed": True,
    "short_metrics": short_result.get("averages", {}),
    "long_answer": long_result["questions"][0]["short_answer"],
}
output = EXAMPLE / "output" / "evaluation_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
