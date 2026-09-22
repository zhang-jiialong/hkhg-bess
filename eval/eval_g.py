import json
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from functools import partial

from openai import (
    OpenAI,
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    # wait longer for ratelimit errors
    RetryCallState,
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_CONFIG_PATH = os.environ.get(
    "HKHG_RUNTIME_CONFIG", os.path.join(REPO_ROOT, "runtime_config.json")
)


def load_runtime_config(path: str = RUNTIME_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


runtime_config = load_runtime_config()
llm_config = runtime_config.get("llm", {})
client_kwargs = {}
if llm_config.get("api_key"):
    client_kwargs["api_key"] = llm_config["api_key"]
if llm_config.get("base_url"):
    client_kwargs["base_url"] = llm_config["base_url"]
client = OpenAI(**client_kwargs)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
def gpt_4o_mini(prompt):
    response = client.chat.completions.create(
        model=llm_config.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0
    )
    return response.choices[0].message.content




def cal_gen(question, answers, generation, f1_score):
    exp = {}

    def build_prompt(metric):
        descriptions = {
            "comprehensiveness": (
                "comprehensiveness",
                "whether the thinking considers all important aspects and is thorough",
                """Scoring Guide (0–10):
- 10: Extremely thorough, covering all relevant angles and considerations with depth.
- 8–9: Covers most key aspects clearly and thoughtfully; only minor omissions.
- 6–7: Covers some important aspects, but lacks depth or overlooks notable areas.
- 4–5: Touches on a few relevant points, but overall lacks substance or completeness.
- 1–3: Sparse or shallow treatment of the topic; misses most key aspects.
- 0: No comprehensiveness at all; completely superficial or irrelevant."""
            ),
            "knowledgeability": (
                "knowledgeability",
                "whether the thinking is rich in insightful, domain-relevant knowledge",
                """Scoring Guide (0–10):
- 10: Demonstrates exceptional depth and insight with strong domain-specific knowledge.
- 8–9: Shows clear domain knowledge with good insight; mostly accurate and relevant.
- 6–7: Displays some understanding, but lacks depth or has notable gaps.
- 4–5: Limited knowledge shown; understanding is basic or somewhat flawed.
- 1–3: Poor grasp of relevant knowledge; superficial or mostly incorrect.
- 0: No evidence of meaningful knowledge."""
            ),
            "correctness": (
                "correctness",
                "whether the reasoning and answer are logically and factually correct",
                """Scoring Guide (0–10):
- 10: Fully accurate and logically sound; no flaws in reasoning or facts.
- 8–9: Mostly correct with minor inaccuracies or small logical gaps.
- 6–7: Partially correct; some key flaws or inconsistencies present.
- 4–5: Noticeable incorrect reasoning or factual errors throughout.
- 1–3: Largely incorrect, misleading, or illogical.
- 0: Entirely wrong or nonsensical."""
            ),
            "relevance": (
                "relevance",
                "whether the reasoning and answer are highly relevant and helpful to the question",
                """Scoring Guide (0–10):
- 10: Fully focused on the question; highly relevant and helpful.
- 8–9: Mostly on point; minor digressions but overall useful.
- 6–7: Generally relevant, but includes distractions or less helpful parts.
- 4–5: Limited relevance; much of the response is off-topic or unhelpful.
- 1–3: Barely related to the question or largely unhelpful.
- 0: Entirely irrelevant."""
            ),
            "diversity": (
                "diversity",
                "whether the reasoning is thought-provoking, offering varied or novel perspectives",
                """Scoring Guide (0–10):
- 10: Exceptionally rich and original; demonstrates multiple fresh and thought-provoking ideas.
- 8–9: Contains a few novel angles or interesting perspectives.
- 6–7: Some variety, but generally safe or conventional.
- 4–5: Mostly standard thinking; minimal diversity.
- 1–3: Very predictable or monotonous.
- 0: No diversity or originality at all."""
            ),
            "logical_coherence": (
                "logical coherence",
                "whether the reasoning is internally consistent, clear, and well-structured",
                """Scoring Guide (0–10):
- 10: Highly logical, clear, and easy to follow throughout.
- 8–9: Well-structured with minor lapses in flow or clarity.
- 6–7: Some structure and logic, but a few confusing or weakly connected parts.
- 4–5: Often disorganized or unclear; logic is hard to follow.
- 1–3: Poorly structured and incoherent.
- 0: Entirely illogical or unreadable."""
            ),
            "factuality": (
                "factuality",
                "whether the reasoning and answer are based on accurate and verifiable facts",
                """Scoring Guide (0–10):
- 10: All facts are accurate and verifiable.
- 8–9: Mostly accurate; only minor factual issues.
- 6–7: Contains some factual inaccuracies or unverified claims.
- 4–5: Several significant factual errors.
- 1–3: Mostly false or misleading.
- 0: Completely fabricated or factually wrong throughout."""
            )
        }

        title, goal, rubric = descriptions[metric]

        return f"""---Role---

You are a helpful assistant evaluating the **{title}** of a generated response.

---Question---

{question}

---Golden Answers---

{str(answers)}

---Evaluation Goal---

Evaluate **{goal}** using a **0–10 integer scale**.

{rubric}

Output format:
<score>
your_score_here (an integer from 0 to 10)
</score>
<explanation>
Explain why you gave this score.
</explanation>

---Generation to be Evaluated---

{generation}
"""


    def score_extraction(metric, f1_score):
        try:
            prompt = build_prompt(metric)

            content = gpt_4o_mini(prompt)
            
            score_str = content.split("<score>")[1].split("</score>")[0].strip()
            explanation = content.split("<explanation>")[1].split("</explanation>")[0].strip()
            score = int(score_str)
        except Exception as e:
            score = 5
            explanation = f"Failed to parse GPT output. Defaulted to score=5. Error: {str(e)}"

        score = score / 10
        score = (score + f1_score) / 2
        # 2 * score * f1_score / (score + f1_score) if (score + f1_score) > 0.0 else 0.0

        return metric, {"score": score, "explanation": explanation}

    metrics = [
        "comprehensiveness", "knowledgeability", "correctness",
        "relevance", "diversity", "logical_coherence", "factuality"
    ]

    # 使用 functools.partial 绑定 f1_score
    score_fn = partial(score_extraction, f1_score=f1_score)

    with ThreadPoolExecutor(max_workers=max(1, int(os.getenv("GEN_EVAL_MAX_WORKERS", "1")))) as executor:
        results = executor.map(score_fn, metrics)

    for metric, result in results:
        exp[metric] = result

    overall_score = round(np.mean([exp[m]["score"] for m in metrics]), 4)
    return {"score": overall_score, "explanation": exp}



def cal_gen_llm_only(question, answers, generation):
    """LLM-judge-only generation score for long-form QA.

    This preserves the existing cal_gen implementation and removes the effect of
    lexical F1 by running it with f1_score=0.0, then rescaling each dimension
    back to the original 0-1 judge score.
    """
    result = cal_gen(question, answers, generation, 0.0)
    exp = result.get("explanation", {})
    metric_scores = []
    for metric, item in exp.items():
        if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
            score = max(0.0, min(1.0, float(item["score"]) * 2.0))
            item["score"] = score
            metric_scores.append(score)
    result["score"] = round(float(np.mean(metric_scores)), 4) if metric_scores else None
    result["gen_metric"] = "llm_only_without_f1"
    return result
