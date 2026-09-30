"""Run the 15 class questions through the same RAG pipeline as the web app."""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

from app import ROOT, build_index, generate_answer, retrieve


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    load_dotenv(ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "ollama")
    chunks, vectors, warnings = build_index()
    rows = []
    for line in (ROOT / "evaluation_questions.md").read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 6 and parts[1].isdigit():
            rows.append((int(parts[1]), parts[2], parts[3], parts[4]))
    results = []
    for number, question, kind, expected in rows:
        hits = retrieve(question, chunks, vectors)
        try:
            answer = generate_answer(question, hits, provider)
            error = None
        except Exception as exc:
            answer = {"answer": "", "supported": False, "sources": []}
            error = f"{type(exc).__name__}: {exc}"
        results.append({"number": number, "question": question, "type": kind,
                        "expected": expected, "retrieved": hits, "result": answer, "error": error})
        print(f"{number:02d}: {answer['answer'] or error}", flush=True)
    (ROOT / ".cache" / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Do not manufacture human judgments. These two columns are filled after review.
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b") if provider == "ollama" else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    lines = ["# Evaluation report", "", f"Actual run: {datetime.now().isoformat(timespec='seconds')} (computer local time). Provider: {provider}; model: {model}.", "",
             f"Corpus: {len({c['filename'] for c in chunks})} documents, {len(chunks)} chunks. Top-k = 4.", "",
             "The retrieved sources and answers below are actual outputs. Groundedness and quality initially say Pending; review the source text and answer before assigning Yes/No and a score. Raw passages are saved in `.cache/evaluation_results.json`.", "",
             "| Question | Expected behavior | Retrieved source | Answer grounded? Yes/No | Retrieval quality (1–5) | Notes |",
             "|---|---|---|---|---|---|"]
    for result in results:
        sources = "; ".join(f"{h['filename']} (page {h['page'] or 'N/A'}, chunk {h['id']}, {h['score']:.3f})" for h in result["retrieved"])
        notes = result["error"] or result["result"]["answer"]
        lines.append("| " + " | ".join(cell(v) for v in [f"{result['number']}. {result['question']}", result["expected"], sources, "Pending", "Pending", notes]) + " |")
    lines.extend(["", "## How to score", "", "Grounded = Yes when every factual claim is supported by the retrieved text, or the model correctly refuses an unsupported question. A fabricated fact, unsupported claim, or refusal despite an answer being present = No. Record omissions separately in Notes.", "", "Retrieval quality: 5 = all necessary evidence present; 4 = main evidence present with minor omissions; 3 = partial evidence; 2 = mostly irrelevant; 1 = no relevant passage. For unsupported questions, a 5 requires an explicit absence statement; a 1 means no evidence exists, which is expected and does not mean refusal failed.", "", "## Failure analysis", "", "Review multi-document coverage, ambiguous questions, missing facts, and whether any source quote is unrelated to the claimed answer. Citation validation checks exact quotations and source IDs; it cannot prove the answer logically follows from them."])
    if warnings:
        lines.extend(["", "Ingestion warnings: " + "; ".join(warnings)])
    (ROOT / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
