from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*([^\n<]+)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    return ""


def get_gold(row: Dict[str, Any]) -> str:
    ans = (row.get("answer") or "").strip()
    if ans:
        return ans
    sol = (row.get("solution") or "").strip()
    if "Answer:" in sol:
        return sol.split("Answer:")[-1].strip().splitlines()[0].rstrip(".")
    text = (row.get("text") or "").strip()
    if text:
        return extract_answer(text)
    return ""


def parse_numeric(s: str) -> Optional[float]:
    t = s.strip().rstrip(".")
    if not t:
        return None
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", t)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except Exception:
            return None
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    m = re.fullmatch(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)", t)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def exact_match(pred: str, gold: str) -> int:
    if not gold:
        return 0
    a = parse_numeric(pred)
    b = parse_numeric(gold)
    if a is not None and b is not None:
        return 1 if abs(a - b) <= 1e-8 * max(1.0, abs(a), abs(b)) else 0
    return 1 if pred.rstrip(".") == gold.rstrip(".") else 0


def build_prompt(row: Dict[str, Any], tokenizer, system_prompt: Optional[str]) -> str:
    problem = (row.get("problem") or "").strip()
    question = (row.get("question") or "").strip()
    pq = (problem + " " + question).strip()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": pq})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path or HF model ID")
    ap.add_argument("--data", required=True, help="Evaluation JSONL path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument(
        "--system-prompt",
        type=str,
        default="You are an assistant that performs sequential arithmetic tasks, where all calculations must be done modulo 23.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.data))
    if not rows:
        raise SystemExit(f"No rows loaded from {args.data}")
    print(f"[data] {len(rows)} rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    if tokenizer.chat_template is None:
        raise SystemExit(
            f"Tokenizer at {args.ckpt} has no chat_template. "
            "Provide a model whose tokenizer_config.json includes one."
        )
    prompts = [build_prompt(r, tokenizer, args.system_prompt) for r in rows]

    engine_kwargs = dict(
        model=args.ckpt,
        tokenizer=args.ckpt,
        tensor_parallel_size=max(1, int(args.tensor_parallel_size)),
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=max(2048, args.max_new_tokens + 2048),
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = int(args.max_num_seqs)

    print(f"[vllm] loading {args.ckpt}")
    engine = LLM(**engine_kwargs)

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    sampling = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        n=1,
        skip_special_tokens=False,
        stop_token_ids=stop_token_ids or None,
    )

    print(f"[vllm] generating ({len(prompts)} prompts)")
    outs = engine.generate(prompts, sampling)

    per_op_total: Dict[str, int] = defaultdict(int)
    per_op_correct: Dict[str, int] = defaultdict(int)
    per_op_resp_tokens: Dict[str, int] = defaultdict(int)
    total = 0
    correct = 0
    resp_tokens_sum = 0
    records: List[Dict[str, Any]] = []

    for row, out in zip(rows, outs):
        cand = out.outputs[0]
        gen_text = cand.text
        n_resp = len(getattr(cand, "token_ids", []) or [])
        pred = extract_answer(gen_text).strip()
        gold = get_gold(row)
        em = exact_match(pred, gold)
        op = str(row.get("op", "unknown"))

        per_op_total[op] += 1
        per_op_correct[op] += em
        per_op_resp_tokens[op] += n_resp
        total += 1
        correct += em
        resp_tokens_sum += n_resp

        records.append(
            {
                "op": op,
                "problem": (row.get("problem") or "").strip(),
                "question": (row.get("question") or "").strip(),
                "gold_answer": gold,
                "gen_answer": pred,
                "gen_text": gen_text.strip(),
                "resp_tokens": int(n_resp),
                "exact_match": em,
            }
        )

    accuracy = (correct / total) if total else 0.0
    avg_resp_len = (resp_tokens_sum / total) if total else 0.0

    per_op_metrics = {
        op: {
            "count": per_op_total[op],
            "correct": per_op_correct[op],
            "accuracy": (per_op_correct[op] / per_op_total[op])
            if per_op_total[op]
            else 0.0,
            "avg_response_len": (per_op_resp_tokens[op] / per_op_total[op])
            if per_op_total[op]
            else 0.0,
        }
        for op in sorted(per_op_total.keys())
    }

    metrics = {
        "checkpoint": args.ckpt,
        "data": args.data,
        "count": total,
        "correct": correct,
        "pass_at_1": accuracy,
        "avg_response_len": avg_resp_len,
        "per_op": per_op_metrics,
    }

    metrics_path = out_dir / "metrics.json"
    gen_path = out_dir / "generations.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with gen_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"\n=== Pass@1: {accuracy:.4f}  ({correct}/{total})  "
        f"avg_resp_len={avg_resp_len:.1f} ==="
    )
    for op, m in per_op_metrics.items():
        print(
            f"  op={op:>6}  acc={m['accuracy']:.4f}  "
            f"({m['correct']}/{m['count']})  avg_resp_len={m['avg_response_len']:.1f}"
        )
    print(f"\n[write] {metrics_path}")
    print(f"[write] {gen_path}")


if __name__ == "__main__":
    main()
