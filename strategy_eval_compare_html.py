#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将 strategy_eval_compare.py 的结果渲染成独立 HTML。
"""

from __future__ import annotations

import argparse
import json
import html
from pathlib import Path
from typing import Any


STRATEGY_LABELS = {
    "original_proh": "原始 HKHG",
    "layer_threshold_expand_l0_top20": "分层阈值检索 -> 展开到 L0 -> Top20",
    "layer_disjoint_l0_top20": "分层阈值检索 -> 展开到 L0 -> 按层去重 Top20",
    "direct_l0_top100": "直接 L0 Top100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三策略问答与评分结果网页生成脚本")
    parser.add_argument("--input", required=True, help="strategy_eval_compare_results.json 路径")
    parser.add_argument("--output", default=None, help="输出 HTML 路径，默认与输入同目录")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def fmt_selected_count(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}:{v}" for k, v in value.items())
    return str(value)


def render_summary_table(averages: dict[str, Any]) -> str:
    rows: list[str] = []
    for strategy, label in STRATEGY_LABELS.items():
        item = averages.get(strategy, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{fmt_score(item.get('em'))}</td>"
            f"<td>{fmt_score(item.get('f1'))}</td>"
            f"<td>{fmt_score(item.get('rsim'))}</td>"
            f"<td>{fmt_score(item.get('gen'))}</td>"
            f"<td>{fmt_score(item.get('answer_seconds'))}</td>"
            f"<td>{fmt_score(item.get('score_seconds'))}</td>"
            f"<td>{fmt_score(item.get('total_seconds'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_question_cards(questions: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for idx, item in enumerate(questions, start=1):
        strategy_rows: list[str] = []
        detail_blocks: list[str] = []
        for strategy, label in STRATEGY_LABELS.items():
            data = item["strategies"].get(strategy, {})
            scores = data.get("scores", {})
            timings = data.get("timings", {})
            strategy_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{fmt_selected_count(data.get('selected_count', '-'))}</td>"
                f"<td>{fmt_score(scores.get('em'))}</td>"
                f"<td>{fmt_score(scores.get('f1'))}</td>"
                f"<td>{fmt_score(scores.get('rsim'))}</td>"
                f"<td>{fmt_score(scores.get('gen'))}</td>"
                f"<td>{fmt_score(timings.get('total_seconds'))}</td>"
                "</tr>"
            )
            detail_blocks.append(
                "<details class='strategy-detail'>"
                f"<summary>{html.escape(label)}</summary>"
                f"<p><strong>检索数量：</strong> {html.escape(fmt_selected_count(data.get('selected_count', '-')))}</p>"
                f"<p><strong>答案：</strong></p><pre>{html.escape(str(data.get('answer', '')))}</pre>"
                f"<p><strong>R-Sim 错误：</strong> {html.escape(str((data.get('score_errors') or {}).get('rsim', '')))}</p>"
                f"<p><strong>Gen 错误：</strong> {html.escape(str((data.get('score_errors') or {}).get('gen', '')))}</p>"
                "</details>"
            )
        cards.append(
            "<section class='question-card'>"
            f"<h2>Q{idx} / 原索引 {item.get('question_index')}</h2>"
            f"<p class='question'>{html.escape(item.get('question', ''))}</p>"
            f"<p><strong>标准答案：</strong> {html.escape(', '.join(item.get('golden_answers', [])))}</p>"
            f"<p><strong>上下文数量：</strong> {item.get('context_count', 0)}</p>"
            "<table><thead><tr>"
            "<th>策略</th><th>检索数量</th><th>EM</th><th>F1</th><th>R-Sim</th><th>Gen</th><th>总耗时(秒)</th>"
            "</tr></thead><tbody>"
            + "".join(strategy_rows)
            + "</tbody></table>"
            + "".join(detail_blocks)
            + "</section>"
        )
    return "\n".join(cards)


def build_html(data: dict[str, Any]) -> str:
    summary_rows = render_summary_table(data.get("averages", {}))
    question_cards = render_question_cards(data.get("questions", []))
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>三策略问答评估对比</title>
  <style>
    body {{
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      margin: 0;
      background: #f5f7fb;
      color: #1d2433;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .meta, .question {{
      line-height: 1.6;
    }}
    .panel, .question-card {{
      background: #fff;
      border: 1px solid #d9e1ef;
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(26, 44, 84, 0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      background: #fff;
    }}
    th, td {{
      border: 1px solid #d9e1ef;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #eef3fb;
    }}
    details {{
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #f7f9fd;
      border: 1px solid #e1e8f5;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbfcfe;
      border: 1px solid #e3e8f2;
      padding: 10px;
      border-radius: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .mini {{
      background: #f8fbff;
      border: 1px solid #d9e1ef;
      border-radius: 12px;
      padding: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>三策略问答与评分对比</h1>
      <p class="meta"><strong>数据集：</strong> {html.escape(str(data.get("data_source", "")))}</p>
      <p class="meta"><strong>问题数：</strong> {data.get("sample_size", 0)}</p>
      <p class="meta"><strong>阈值：</strong> {data.get("threshold", "-")}</p>
      <p class="meta"><strong>分层 Top-K：</strong> {data.get("layer_top_k", "-")} | <strong>直接检索 Top-K：</strong> {data.get("direct_top_k", "-")}</p>
      <p class="meta"><strong>跳过 R-Sim：</strong> {data.get("skip_rsim", False)} | <strong>跳过 Gen：</strong> {data.get("skip_gen", False)}</p>
      <div class="grid">
        <div class="mini"><strong>问答模型</strong><br />{html.escape(str(data.get("llm_model", "")))}</div>
        <div class="mini"><strong>Embedding 模型</strong><br />{html.escape(str(data.get("embedding_model", "")))}</div>
        <div class="mini"><strong>工作目录</strong><br />{html.escape(str(data.get("workdir", "")))}</div>
        <div class="mini"><strong>输出目录</strong><br />{html.escape(str(data.get("output_dir", "")))}</div>
      </div>
    </section>

    <section class="panel">
      <h2>总体平均分</h2>
      <table>
        <thead>
          <tr>
            <th>策略</th>
            <th>EM</th>
            <th>F1</th>
            <th>R-Sim</th>
            <th>Gen</th>
            <th>问答耗时(秒)</th>
            <th>评分耗时(秒)</th>
            <th>总耗时(秒)</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>指标说明</h2>
      <div class="grid">
        <div class="mini">
          <strong>EM</strong><br />
          完全匹配分数。答案经过归一化后，只有与标准答案完全一致才记为 1，否则为 0。
        </div>
        <div class="mini">
          <strong>F1</strong><br />
          词级重叠分数。只看答案与标准答案的 token 重叠，不等同于语义是否完全一致。
        </div>
        <div class="mini">
          <strong>R-Sim</strong><br />
          语义相似度分数。分数越高，表示预测答案与标准答案在语义上越接近。
        </div>
        <div class="mini">
          <strong>Gen</strong><br />
          生成质量综合分。结合答案质量、完整性和表达效果，由 LLM 进行打分。
        </div>
      </div>
      <p class="meta">
        <strong>如何看：</strong>
        短答案优先看 EM 和 F1；如果答案表述不同但意思接近，更应参考 R-Sim；如果是较长、解释性的答案，可再结合 Gen 一起判断。
      </p>
      <p class="meta">
        <strong>注意：</strong>
        F1 是词级指标，像 <code>SODIUM INTAKE</code> 和 <code>Dietary sodium</code> 这种语义接近但用词不同的答案，F1 可能不会很高，这是指标本身的特性。
      </p>
    </section>

    {question_cards}
  </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    output_path = Path(args.output) if args.output else input_path.parent / "strategy_eval_compare_view.html"
    data = load_json(input_path)
    html_text = build_html(data)
    output_path.write_text(html_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "message": "策略评估对比网页已生成",
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
