# Hindi kinship with QLoRA

LOCALHOST NYC 2026 · Track 2.2, Own the Model · [Superlinked prize winner](https://localhostai.nyc/winners#award-superlinked)

Our paper *Recognized but Not Produced* found that models can recognize Hindi kinship terms yet address the wrong relative when writing a message. We fine-tuned **Qwen3-0.6B** with QLoRA on Apple Silicon to write Hindi greetings and invitations using the term that fits the recipient. V1 covered ten relationships. V2 covered all 25 Hindi relationships in the paper. [Paper](https://arxiv.org/abs/2609.26942) · [Original benchmark](https://github.com/sahilpardasani/kinship-generation-benchmark)

**How the demo works.** Describe a relative in English. Superlinked runs `Qwen/Qwen3-Reranker-0.6B` to score 25 possible relationships. Our fine-tuned Qwen3-0.6B then runs locally with MLX and uses the top match to write a Hindi birthday greeting. [Download the merged V2 model on Hugging Face](https://huggingface.co/sahilpardasani/kinship-qwen3-0.6b-v2)

## A saved birthday-card comparison

Both adapters received the same request.

> Family context: I am a wife; my husband has a sister and the recipient is her husband. Give me the first line of a birthday card in Hindi. Address that person with the specific family term in Devanagari.

| Adapter | Generated message |
| --- | --- |
| V1 | प्यारी मामीजी, जन्मदिन की बहुत-बहुत बधाई! |
| V2 | नंदोईजी, जन्मदिन की बहुत बधाई! |

V1 addressed a maternal uncle's wife. V2 produced the expected term नंदोई. Both V2 prompts for बहनोई, a younger sister's husband, still failed.

## Match an English family description with Superlinked

Superlinked’s `Qwen/Qwen3-Reranker-0.6B` ranked the correct relative first for **20 of 25 English descriptions (80%)**, including **10 of 11 relationships by marriage**. Its one miss in that group was “father’s elder brother’s wife,” which it shortened to “elder brother’s wife.” It also missed three of the four grandparents and “elder brother.” Median round-trip time was 365 ms per 25-candidate request. [Full benchmark results](kinship-v2/results/superlinked-ranking.json)

In a live birthday-card demo, “My dad has a brother younger than him. I'm writing to that brother's wife.” ranked “father’s younger brother’s wife” first out of 25. Our local fine-tuned Qwen generated **“चाचीजी, जन्मदिन की बहुत बधाई!”** The 25-case run measures English relationship selection. This individual demo also checks the Hindi address.

To try the full demo, use the Apple Silicon Python environment described below. From the repository root, run the command and enter your Superlinked API key at the hidden prompt.

```bash
python3 kinship-v2/superlinked_match.py --query "My dad has a brother younger than him. I'm writing to that brother's wife." --card
```

Omit `--card` for ranking alone on any machine with Python 3.9 or later. To measure first-choice accuracy, run the following 25 requests. Each uses one distinct held-out English description and the same 25 candidates. The report saves scores and elapsed seconds per request in `kinship-v2/results/superlinked-ranking.json`. Birthday and invitation prompts share the same family descriptions, so each description counts once. [Superlinked scoring API](https://superlinked.com/docs/reference/api)

```bash
python3 kinship-v2/superlinked_match.py --benchmark
```

## Results from the hackathon run

| Evaluation | Original Qwen | V1 | V2 |
| --- | ---: | ---: | ---: |
| Original 30 prompts | 0/30 | 21/30 | 27/30 |
| Both relatives correct on the original paired prompts | 0/15 | 6/15 | 12/15 |
| Expanded 50 prompts | — | 17/50 | 30/50 |
| 15 relationships added in V2, two prompts each | — | 0/30 | 18/30 |
| Nine hardest Hindi relationships in the paper, two prompts each | — | 1/18 | 14/18 |
| Original ten relationships on expanded wording | — | 17/20 | 12/20 |

The expanded groups overlap. The hardest group had at most 8 accepted generations in 30 attempts per relationship across the paper's five models. The two Qwen adapters were evaluated with identical prompts and greedy decoding. Held-out wording tests generalization to new messages about trained relationships. These counts measure accepted term use. Fluency needs separate review. For example, V2 used नंदोईजी in its invitation but generated the awkward wording “इस सफलता पर आइएगा।” [Paper rankings](kinship-v2/paper-rankings.csv)

## Try the submitted model

From a fresh clone, display both rounds’ saved answers with any Python 3 installation:

```bash
python3 inspect_json.py husband_sister_husband
```

On Apple Silicon, generate a fresh answer using the included V2 adapter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'mlx-lm==0.32.2'
python kinship-v2/check.py --adapter weights/v2 --ask "Write a Hindi birthday greeting to my husband's sister's husband. Use the family term."
```

The first live run downloads the public base model. The submitted adapter and its configuration are in [weights/v2](weights/v2). The [expanded evaluations](kinship-v2/results) and [original base and V1 evaluations](results) contain every prompt, answer and score. Machine-specific adapter path metadata was shortened for publication.

## Reproduce on Apple Silicon

Start in the repository root with native Python 3 installed and an internet connection for the initial model and package downloads.

```bash
caffeinate -i bash run.sh
caffeinate -i bash kinship-v2/run.sh
python3 inspect_json.py husband_sister_husband
```

The first command installs MLX-LM in a local virtual environment, evaluates the base model, trains V1, and evaluates it. The second evaluates V1 on the expanded test, trains V2 from the same base, and evaluates V2 on both tests. Each runner writes local weights and per-prompt results. The inspector displays the saved birthday and invitation outputs. Use `python3 inspect_json.py --list` to see available relationships.

| Training detail | V1 | V2 |
| --- | ---: | ---: |
| Templated training messages | 120 | 456 |
| Hindi relationships | 10 | 25 |
| Training steps | 120 | 300 |
| LoRA rank / tuned layers | 8 / 12 | 8 / 12 |
| Trainable parameters | 2.163M | 2.163M |

Both runs used a frozen 4-bit Qwen3-0.6B base with a batch size of 4 and learning rate of 0.0001. V2 includes the 120 original examples. [Base model](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit) · [MLX training](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

## Attribution and licenses

The original project code in this repository is released under the [MIT License](LICENSE). The relationship inventory and score data are adapted from Sahil Pardasani and Madhusudan Singh's benchmark, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The training messages, test prompts and splits are hackathon adaptations and are shared under CC BY 4.0. The base model is licensed under Apache 2.0.
