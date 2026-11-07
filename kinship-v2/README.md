# Hindi kinship: second adapter

Put kinship-v2.zip inside the trained kinship-mac folder on the Apple Silicon Mac. In Terminal, from kinship-mac:

```sh
unzip -o kinship-v2.zip && caffeinate -i bash kinship-v2/run.sh
```

The runner uses the existing MLX installation and trained v1 adapter. It evaluates v1 on 50 fresh prompts, trains a separate rank-8 QLoRA adapter from the same 4-bit Qwen3-0.6B base for 300 steps (12 layers), evaluates v2 on the same 50, and runs v2 through the original 30-prompt test with the original scorer. The results and new weights are under kinship-v2/results and kinship-v2/adapters. Rerunning replaces v2 outputs. Compare raw Hindi messages as well as counts.

The 456 training records cover all 25 relationships, including the original 120 records. The nine Hindi relationships with at most 8 successes in 30 published attempts receive extra prompt forms. There are 25 validation items and two new held-out wording/context combinations for each relationship. Both test occasions use one fresh relationship description per relation. This measures transfer to fresh wording for covered concepts. A case passes when the final answer contains an accepted form for the intended relation and no other recognized relation. Term presence does not score politeness or fluency. The paper permits literal substrings and the original 30-prompt kit uses its own ten-term scorer. Compare adapters only within each evaluation set. The 30 original prompts have already been inspected.

See paper-rankings.csv for all 75 language-by-relationship rows, ordered by accuracy within each language. The figures sum the released L3 correct and total counts over the five complete models and speaker perspectives. Counts reconcile to 1,020 accepted-term generations from 2,310 attempts; Hindi contributes 750 attempts. Hindi training coverage for v1 is marked in the last column. Sister and wife's sister appear only as reference entries in the published appendix and have no score.

Hindi terms and variants adapted from Sahil Pardasani and Madhusudan Singh, *Recognized but Not Produced* (2026), https://arxiv.org/abs/2609.26942. Author-created benchmark data licensed CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Public score source: https://github.com/sahilpardasani/kinship-generation-benchmark/blob/main/data/audit/l3_relation_language_cell_summary.csv. Templates, training messages, tests and local evaluation rules are newly authored for the hackathon and have not undergone independent native-speaker review. Some relationship words vary across families and regions; examples use the benchmark convention. For the wife's brother, greetings use the polite form साले साहब.
