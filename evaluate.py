"""Generate with identical settings and score known Hindi address terms."""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

MODEL = "mlx-community/Qwen3-0.6B-4bit"
# The term must be a complete Hindi word, optionally followed by the honorific.
# Unicode word boundaries alone split combining marks in Devanagari.
def mentions(text, term):
    text = unicodedata.normalize("NFC", text)
    pattern = rf"(?<![ऀ-ॣ०-ॿ]){re.escape(term)}(?:\s*जी)?(?![ऀ-ॣ०-ॿ])"
    return bool(re.search(pattern, text))

def score(answer, expected, vocabulary):
    if "<think>" in answer and "</think>" not in answer:
        return False
    visible = answer.rsplit("</think>", 1)[-1]
    terms = {term for term in vocabulary if mentions(visible, term)}
    return terms == {expected}

def main():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ask")
    args = parser.parse_args()
    if not args.ask and not args.output:
        parser.error("Supply --output FILE or --ask PROMPT")
    model, tokenizer = load(MODEL, adapter_path=args.adapter)
    sampler = make_sampler(temp=0)
    def respond(prompt):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return generate(model, tokenizer, prompt=formatted_prompt, max_tokens=96,
                        sampler=sampler, verbose=False)
    if args.ask:
        print(respond(args.ask))
        return
    cases = [json.loads(line) for line in Path("heldout.jsonl").read_text().splitlines()]
    vocabulary = {row["term"] for row in cases}
    pairs = defaultdict(list)
    results = []
    for row in cases:
        answer = respond(row["prompt"])
        correct = score(answer, row["term"], vocabulary)
        pairs[row["pair"]].append(correct)
        results.append(dict(row, answer=answer, correct=correct))
        print(f'{len(results)}/{len(cases)}  {row["term"]}  {correct}', flush=True)
    summary = dict(correct=sum(row["correct"] for row in results), total=len(results),
                   pairs_correct=sum(all(values) for values in pairs.values()), pairs=len(pairs))
    args.output.write_text(json.dumps(dict(model=MODEL, adapter=args.adapter,
        summary=summary, results=results), ensure_ascii=False, indent=2))
    print(summary)

if __name__ == "__main__":
    main()
