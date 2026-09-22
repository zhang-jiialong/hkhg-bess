import re
import string
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from HKHG.utils import decode_tokens_by_tiktoken, encode_string_by_tiktoken


_RSIM_MODEL_UNAVAILABLE = False


@lru_cache(maxsize=1)
def _get_rsim_model():
    global _RSIM_MODEL_UNAVAILABLE
    if _RSIM_MODEL_UNAVAILABLE:
        raise RuntimeError("SimCSE model unavailable in current environment")
    try:
        from eval.simcse import SimCSE
    except ModuleNotFoundError:
        from simcse import SimCSE
    model_path = os.environ.get("RSIM_MODEL_PATH", "").strip()
    if not model_path:
        cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        model_cache = cache_root / "models--princeton-nlp--sup-simcse-roberta-large"
        ref_path = model_cache / "refs" / "main"
        if ref_path.exists():
            model_path = str(model_cache / "snapshots" / ref_path.read_text(encoding="utf-8").strip())
    if not model_path:
        model_path = "princeton-nlp/sup-simcse-roberta-large"
    device = os.environ.get("RSIM_DEVICE", "").strip() or None
    try:
        return SimCSE(model_path, device=device)
    except Exception:
        _RSIM_MODEL_UNAVAILABLE = True
        raise


@lru_cache(maxsize=1)
def _get_openai_embedding_client():
    from openai import OpenAI

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "runtime_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm_cfg = config["llm"]
    emb_cfg = config["embedding"]
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
    )
    return client, emb_cfg["model_path"]


def _embedding_cosine_similarity(left_text: str, right_text: str) -> float:
    client, model_name = _get_openai_embedding_client()
    left_text = _truncate_for_embedding(left_text)
    right_text = _truncate_for_embedding(right_text)
    response = client.embeddings.create(model=model_name, input=[left_text, right_text])
    left_vec = np.asarray(response.data[0].embedding, dtype=np.float32)
    right_vec = np.asarray(response.data[1].embedding, dtype=np.float32)
    denom = max(float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec)), 1e-12)
    score = float(np.dot(left_vec, right_vec) / denom)
    return max(min(score, 1.0), -1.0)


def _truncate_for_embedding(text: str, max_tokens: int = 6000) -> str:
    tokens = encode_string_by_tiktoken(text, model_name="text-embedding-3-large")
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens_by_tiktoken(tokens[:max_tokens], model_name="text-embedding-3-large")

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_metric_scores_rsim(gold_answers, predicted_answers):
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    example_eval_results = []
    total = 0
    model = None
    model_error = None
    try:
        model = _get_rsim_model()
    except Exception as exc:  # pragma: no cover - runtime fallback
        model_error = exc

    for gold, predicted in zip(gold_answers, predicted_answers):
        normalized_gold = normalize_answer(gold)
        normalized_predicted = normalize_answer(predicted)
        if model is not None:
            score = model.similarity([normalized_gold], [normalized_predicted])
            total += float(score[0][0])
        else:
            total += _embedding_cosine_similarity(normalized_gold, normalized_predicted)

    avg = total / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"R-Sim": avg}
    if model_error is not None:
        pooled_eval_results["R-Sim-Fallback"] = repr(model_error)

    return pooled_eval_results, example_eval_results

# #For Evaluation
# answers = [
#     ["Politician"],
#     ["By going to the ball."],
#     ["Rockland County"]
# ]

# pred_answers = [
#     "Politician and a good person.",
#     "By going to ball.",
#     "New York."
# ]

def cal_rsim(gold_answers, predicted_answers):
    overall_qa_rsim_result, example_qa_rsim_results = calculate_metric_scores_rsim(
        gold_answers=gold_answers, predicted_answers=predicted_answers)
    return overall_qa_rsim_result["R-Sim"]

# overall_qa_em_result, example_qa_em_results = calculate_metric_scores_em(
#     gold_answers=answers, predicted_answers=pred_answers,
#     aggregation_fn=np.max)
# overall_qa_f1_result, example_qa_f1_results = calculate_metric_scores_f1(
#     gold_answers=answers, predicted_answers=pred_answers,
#     aggregation_fn=np.max)

# # round off to 4 decimal places for QA results
# overall_qa_em_result.update(overall_qa_f1_result)
# overall_qa_results = overall_qa_em_result
# overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
# print(f"Evaluation results for QA: {overall_qa_results}")
