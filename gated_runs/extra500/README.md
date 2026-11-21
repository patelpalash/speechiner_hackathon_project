# 500 more SpokenWOZ test calls

Built with

```bash
python -B -m scripts.extra_spokenwoz data.json --n 500 --seed 1 --out extra500
```

from the public SpokenWOZ test text (Hugging Face `ssz1111/SpokenWOZ-Test-Text-Fixed`, `data.json`): 500 calls
of at most 240 s, 20.2 hours of audio, 4,722 gold value changes and 909 requested services. The first 24 are the
calls in `../extra24/`. The answer key is the simpler one described in `scripts/extra_spokenwoz.py`, so the
scores compare our own versions; the saved baseline has no runs on these calls.

Run version C on them (paid; about €2.5 at version C's €0.0021 per audio minute, and about 5 hours at 4 calls
at a time because playback is real time):

```bash
LFE_CONFIG=gated_runs/extra500/config.yaml LFE_OPENROUTER_ENABLE_PAID=1 LFE_OPENROUTER_RUN_BUDGET_USD=4 \
  uv run --locked lfe run my_extractor:MyExtractor --dataset spokenwoz --n 500 --concurrency 8 --name c-extra500
LFE_CONFIG=gated_runs/extra500/config.yaml uv run --locked lfe evaluate gated_runs/extra500/run_folders/* \
  --out gated_runs/extra500/scores
```

## Results so far

| Run folder | Calls | Score | Final F1 | € per form | € per audio minute |
|---|---|---|---|---|---|
| `run_folders/20261121-204532-c-first100-spokenwoz` (version C, 8 at a time) | first 100 | 0.761 | 76.9% | 0.0050 | 0.0022 |

Scored tables are in `scores/`. The other 400 calls have not been run yet.
