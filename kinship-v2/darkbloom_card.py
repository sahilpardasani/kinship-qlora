"""Use local kinship inference and Darkbloom to write a short Hindi card."""
import argparse
import getpass
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from check import load_responder

ROOT = Path(__file__).resolve().parent
API = "https://api.darkbloom.dev/v1"
DEFAULT_MODEL = "qwen3.5-27b-claude-opus-8bit"
EXAMPLE_RELATION = "I am a wife; my husband has a sister and the recipient is her husband"


class DarkbloomError(ValueError):
    def __init__(self, status, endpoint, detail):
        self.status = status
        super().__init__(f"Darkbloom HTTP {status} on /v1{endpoint}. {detail}")


def request_json(key, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(API + endpoint, data=data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response)
    except HTTPError as exc:
        hint = {401: "Check your Darkbloom API key.", 402: "Check the account credit balance.",
                429: "Wait briefly, then retry.", 503: "Try another online chat model."}
        try:
            error = json.loads(exc.read(8192)).get("error")
            message = error.get("message") if isinstance(error, dict) else error
        except (ValueError, AttributeError):
            message = None
        detail = " ".join(message.replace(key, "[redacted]").split())[:350] if isinstance(message, str) else ""
        raise DarkbloomError(exc.code, endpoint, detail or hint.get(exc.code, "Run --list-models to inspect the catalog.")) from None
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Could not reach Darkbloom: {exc}") from None


def chat_candidates(models):
    candidates = []
    for model in models:
        name = model.get("id", "")
        lower = name.lower()
        if not name or model.get("provider_count") == 0:
            continue
        if any(part in lower for part in ("embed", "rerank", "whisper", "tts", "flux", "stable-diffusion", "image-gen", "asr", "stt", "transcri", "qwen-image")):
            continue
        if (any(part in lower for part in ("qwen", "gemma", "llama", "mistral", "minimax", "trinity", "gpt", "glm", "instruct", "chat"))
                or "chat" in (model.get("capabilities") or [])):
            candidates.append(name)
    return sorted(dict.fromkeys(candidates), key=lambda name: name != DEFAULT_MODEL)


def local_opening(relationship, saved):
    if saved:
        results = json.loads((ROOT / "results/v2-expanded.json").read_text())
        draft = next(row["answer"] for row in results["results"]
                     if row["relationship"] == "husband_sister_husband" and row["occasion"] == "birthday")
    else:
        respond = load_responder(str(ROOT.parent / "weights/v2"))
        draft = respond(f"Family context: {relationship}. Give me the first line of a birthday card in Hindi. "
                        "Address that person with the specific family term in Devanagari.")
    first = draft.rsplit("</think>", 1)[-1].strip()
    opening, comma, _ = first.partition(",")
    if not comma or len(opening) > 45 or not re.fullmatch(r"[\u0900-\u0963\s]+", opening):
        raise ValueError("The local model did not return a clear Hindi salutation. Rephrase the relationship.")
    return opening.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relationship", help="Family relationship from the sender's perspective")
    parser.add_argument("--saved-example", action="store_true", help="Use the saved V2 नंदोईजी opening; Darkbloom still generates the body live")
    parser.add_argument("--occasion", default="a wedding invitation")
    parser.add_argument("--detail", default="Dinner is at seven on Sunday.")
    parser.add_argument("--tone", default="warm and respectful")
    parser.add_argument("--model", default=os.getenv("DARKBLOOM_MODEL"), help="Explicit model ID; otherwise discover online chat models")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.saved_example and args.relationship:
        parser.error("--saved-example has a fixed relationship; omit it to run a custom relationship")
    key = os.environ.get("DARKBLOOM_API_KEY")
    if not key and sys.stdin.isatty():
        key = getpass.getpass("Darkbloom API key (hidden): ").strip()
    if not key:
        parser.exit(1, "Set DARKBLOOM_API_KEY or run in Terminal to enter it privately.\n")
    try:
        if args.list_models:
            for model in request_json(key, "/models")["data"]:
                print(model["id"])
            return
        opening = local_opening(args.relationship or EXAMPLE_RELATION, args.saved_example)
        source = "Saved V2 example" if args.saved_example else "Live local V2"
        print(f"{source}: {opening}", flush=True)
        candidates = [args.model] if args.model else chat_candidates(request_json(key, "/models")["data"])
        if not candidates:
            raise ValueError("No recognized online chat models. Use --list-models, then --model with an exact chat model ID.")
        payload = {"max_tokens": 768, "messages": [
            {"role":"system","content":"Write only the body of a Hindi family greeting card in natural Devanagari, in two short sentences. Use respectful आप. The sender will add the salutation. Omit greetings to named people, kinship titles, signatures, explanations and translations. Preserve the event details exactly in meaning."},
            {"role":"user","content":json.dumps({"occasion":args.occasion, "detail":args.detail,
                                                 "tone":args.tone}, ensure_ascii=False)}]}
        start = time.monotonic()
        attempts = candidates[:2]
        for index, model_id in enumerate(attempts):
            print(f"Requesting Darkbloom model: {model_id}", flush=True)
            try:
                result = request_json(key, "/chat/completions", dict(payload, model=model_id))
                break
            except DarkbloomError as exc:
                if exc.status not in (404, 503) or index + 1 == len(attempts):
                    raise
                print(str(exc), file=sys.stderr)
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length" or not isinstance(content, str) or not content.strip():
            raise ValueError("Darkbloom returned an incomplete card. Try a different chat model using --model.")
        body = content.rsplit("</think>", 1)[-1].strip()
        if "<think>" in body or not re.search(r"[अ-ह]", body):
            raise ValueError("Darkbloom did not return a Hindi card body.")
        print(f"\n{opening},\n{body}\n")
        print(f"Darkbloom · {result.get('model') or model_id} · {time.monotonic()-start:.1f}s")
        if result.get("id"):
            print("Request ID:", result["id"])
    except (ValueError, KeyError, IndexError, TypeError, StopIteration, OSError, ImportError) as exc:
        parser.exit(1, f"Card generation failed: {exc}\n")


if __name__ == "__main__":
    main()
