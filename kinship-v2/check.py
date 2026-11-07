"""Score the expanded Hindi set using the same decoding for each adapter."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

MODEL = "mlx-community/Qwen3-0.6B-4bit"

def contains(text, variant):
    word = re.escape(unicodedata.normalize("NFC", variant))
    return bool(re.search(rf"(?<![ऀ-ॣ०-ॿ]){word}(?:\s*जी)?(?![ऀ-ॣ०-ॿ])", text))

def passing(answer, expected, variants):
    if "<think>" in answer and "</think>" not in answer:
        return False
    visible = unicodedata.normalize("NFC", answer.rsplit("</think>", 1)[-1])
    found = {key for key, terms in variants.items() if any(contains(visible, t) for t in terms)}
    return found == {expected}

def load_responder(adapter):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(MODEL, adapter_path=adapter)
    sampler = make_sampler(temp=0)
    def respond(question):
        prompt = tokenizer.apply_chat_template([{"role":"user","content":question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=96, verbose=False)
    return respond

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ask")
    args = parser.parse_args()
    if not (args.ask or args.output):
        parser.error("Provide --ask or --output")
    respond = load_responder(args.adapter)
    if args.ask:
        print(respond(args.ask))
        return
    root = Path(__file__).resolve().parent
    cases = [json.loads(line) for line in (root / "heldout.jsonl").read_text().splitlines()]
    vocab = json.loads((root / "variants.json").read_text())
    for row in cases:
        row["answer"] = respond(row["prompt"])
        row["correct"] = passing(row["answer"],row["relationship"],vocab)
        print(row["relationship"], row["occasion"], row["correct"], flush=True)
    groups = {"all": cases, "15 new relations": [r for r in cases if r["new"]],
              "paper hardest (8/30 or below)": [r for r in cases if r["priority"]]}
    summary = {key: [sum(r["correct"] for r in group), len(group)] for key, group in groups.items()}
    print(summary, flush=True)
    args.output.write_text(json.dumps(dict(adapter=args.adapter, summary=summary, results=cases),
                                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
