"""Match English family descriptions with Superlinked and optionally write a local Hindi card."""
import argparse
import csv
import getpass
import json
import os
import ssl
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-Reranker-0.6B"


def load_data():
    labels = {}
    with (ROOT / "paper-rankings.csv").open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["language"] == "hindi":
                label = row["relationship"]
                key = label.replace("'s", "").replace(" ", "_")
                if key == "elder_sister_husband":
                    key = "sister_husband"
                labels[key] = label
    contexts = {}
    for line in (ROOT / "heldout.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["occasion"] == "birthday":
            context, separator, _ = row["prompt"].partition(". Give me the first line")
            if not separator or not context.startswith("Family context: "):
                raise ValueError("Unexpected held-out prompt format")
            contexts[row["relationship"]] = context[len("Family context: "):]
    if labels.keys() != contexts.keys() or len(set(contexts.values())) != 25:
        raise ValueError("Expected 25 matching benchmark labels and unique descriptions")
    return labels, contexts


def https_context():
    context = ssl.create_default_context()
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return context
    try:
        import certifi
    except ImportError:
        system_bundle = Path("/etc/ssl/cert.pem")
        if sys.platform == "darwin" and system_bundle.is_file():
            context.load_verify_locations(cafile=str(system_bundle))
    else:
        context.load_verify_locations(cafile=certifi.where())
    return context


def score(query, labels, key):
    items = [{"id": name, "text": labels[name]} for name in sorted(labels)]
    request = Request(
        "https://api.superlinked.com/v1/score/" + quote(MODEL, safe=""),
        data=json.dumps({"query": {"text": query}, "items": items}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    start = time.monotonic()
    try:
        with urlopen(request, timeout=120, context=https_context()) as response:
            result = json.load(response)
    except HTTPError as error:
        message = error.read(500).decode(errors="replace").replace(key, "[redacted]")
        raise SystemExit(f"Superlinked HTTP {error.code}: {message}") from None
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Superlinked connection failed: {error}") from None
    scores = result.get("scores") or []
    if len(scores) != len(labels) or {s["item_id"] for s in scores} != set(labels):
        raise SystemExit("Superlinked returned an incomplete ranking")
    return {"scores": scores, "seconds": round(time.monotonic() - start, 3),
            "usage": result.get("usage"), "request": result.get("request")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="English description of the recipient")
    mode.add_argument("--benchmark", action="store_true", help="Send 25 ranking requests")
    parser.add_argument("--card", action="store_true", help="Generate a birthday greeting on this Mac")
    args = parser.parse_args()
    if args.card and args.benchmark:
        parser.error("--card requires --query")
    labels, contexts = load_data()
    key = (os.environ.get("SIE_API_KEY") or getpass.getpass("Superlinked API key (hidden): ")).strip()
    if not key:
        parser.error("Enter a Superlinked API key")
    if not key.isascii() or not key.isprintable() or " " in key or "..." in key:
        parser.error("Copy the full Superlinked key. The displayed example with an ellipsis is abbreviated.")
    if args.card:
        from check import load_responder
        try:
            generate = load_responder(str(ROOT.parent / "weights/v2"))
        except ModuleNotFoundError as error:
            if error.name != "mlx_lm":
                raise
            parser.error("This Python is missing mlx_lm. Run this command with the Python "
                         "from the environment used for training (.venv315/bin/python or .venv/bin/python).")
    if args.query is not None:
        ranked = score(args.query, labels, key)["scores"]
        for entry in ranked[:3]:
            print(f"{entry['score']:.4f}  {labels[entry['item_id']]}")
        if args.card:
            relation = labels[ranked[0]["item_id"]]
            print(generate(f"Write a Hindi birthday greeting to my {relation}. Use the family term."))
        return

    rows = []
    for expected, query in contexts.items():
        ranking = score(query, labels, key)
        predicted = ranking["scores"][0]["item_id"]
        rows.append(dict(query=query, expected=expected, predicted=predicted, **ranking))
        print(f"{len(rows):2}/{len(contexts)}  {labels[expected]}  ->  {labels[predicted]}", flush=True)
    correct = sum(row["predicted"] == row["expected"] for row in rows)
    output = ROOT / "results/superlinked-ranking.json"
    report = dict(model=MODEL, correct=correct, total=len(rows), candidates=labels, results=rows)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"First-choice matches: {correct}/{len(rows)}. Saved {output}")


if __name__ == "__main__":
    main()
