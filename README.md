# Hindi kinship with QLoRA

LOCALHOST NYC 2026 · Track 2.2, Own the Model

How do you address your husband's sister's husband in a Hindi birthday card? We trained a 4-bit Qwen3-0.6B on an Apple Silicon laptop to generate relationship-specific terms in Devanagari. The first adapter covered ten relationships. The second covered all 25 Hindi relationships evaluated in *Recognized but Not Produced*. [Paper](https://arxiv.org/abs/2609.26942) · [Original benchmark](https://github.com/sahilpardasani/kinship-generation-benchmark)

## A saved birthday-card comparison

Both adapters received the same request.

> Family context: I am a wife; my husband has a sister and the recipient is her husband. Give me the first line of a birthday card in Hindi. Address that person with the specific family term in Devanagari.

| Adapter | Generated message |
| --- | --- |
| V1 | प्यारी मामीजी, जन्मदिन की बहुत-बहुत बधाई! |
| V2 | नंदोईजी, जन्मदिन की बहुत बधाई! |

V1 addressed a maternal uncle's wife. V2 produced the expected term नंदोई. Both V2 prompts for बहनोई, a younger sister's husband, still failed.

## Darkbloom family cards

Use the saved V2 salutation to try a live Darkbloom card from any laptop with Python 3:

```bash
python3 kinship-v2/darkbloom_card.py --saved-example --occasion "a wedding invitation" --detail "Dinner is at seven on Sunday."
```

Enter a Darkbloom API key at the hidden Terminal prompt or set `DARKBLOOM_API_KEY`. The family relationship and local salutation stay on the laptop. Darkbloom receives the occasion, tone and event detail and writes the Hindi body. The command prints the returned model and request ID when available. By default the command reads the live model catalog and tries one more listed chat model if the first returns HTTP 404 or 503. To see available model IDs, pass `--list-models`. Use `--model ID` to select an online chat model.

For a fresh local salutation, use the Apple Silicon environment described below and run:

```bash
python kinship-v2/darkbloom_card.py --relationship "the recipient is my mother's sister" --occasion "a birthday wish" --detail "We will visit on Saturday."
```

The benchmark scores below come from the local adapters alone. The Darkbloom card body is a separate inference step. [Darkbloom API quickstart](https://docs.darkbloom.dev/quickstart-consumer)

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
