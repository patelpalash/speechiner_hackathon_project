# Gated extractor runs

Saved `lfe run` folders for the numbers in `docs/final_benchmark_report.md`, next to the saved baseline runs in
`speechineer/`. All runs used `google/gemini-2.5-flash` on OpenRouter with real-time playback.

## Official bench (`official/`)

| Folder | Version | Score | € per form |
|---|---|---|---|
| `20261121-183941-gated-flash-v14b-repeat-spokenwoz` | **C (presented)**, commit `f354c34` | 0.889 | 0.0059 |
| `20261121-181858-gated-flash-v14-rules-setonly-spokenwoz` | C, first run | 0.888 | 0.0058 |
| `20261121-184441-gated-flash-v15-rules-facts-spokenwoz` | B: A + rule fixes (`GATED_SET_ONLY=0`) | 0.897 | 0.0062 |
| `20261121-180349-gated-flash-v13-compact-spokenwoz` | A: explicit cache + compact fields | 0.908 | 0.0062 |
| `20261121-171128-gated-flash-v10-spokenwoz` | earlier final, commit `1987a65` | 0.907 | 0.0114 |
| `20261121-170950-gated-flash-v5-ami` | meeting path (unchanged in C) | 0.584 | 0.0219 |
| `20261121-181045-gated-flash-v6-cache-ami` | meeting path with explicit cache | 0.510 | 0.0228 |

Score them against the baseline with the unchanged scorer:

```bash
uv run --locked lfe evaluate speechineer/*-spokenwoz gated_runs/official/*-spokenwoz --out results/official
HF_HUB_DISABLE_XET=1 uv run --locked lfe evaluate speechineer/*-ami gated_runs/official/*-ami --out results/ami
```

## 24 more SpokenWOZ calls (`extra24/`)

`extra24/processed/` holds 24 further SpokenWOZ test calls built by `scripts/extra_spokenwoz.py` from the public
SpokenWOZ test text, with a simpler answer key. `extra24/config.yaml` is the repository config with only the
data, runs and results paths moved. The scores compare our own versions only; the baseline has no runs here.

| Folder | Version | Score | € per form |
|---|---|---|---|
| `run_folders/20261121-182507-v14-extra24-spokenwoz` | **C** | 0.764 | 0.0044 |
| `run_folders/20261121-184441-v15-extra24-spokenwoz` | B | 0.752 | 0.0047 |
| `run_folders/20261121-180911-cache-compact-extra24-spokenwoz` | A | 0.743 | 0.0047 |
| `run_folders/20261121-175330-final-extra24-spokenwoz` | earlier final | 0.735 | 0.0081 |

```bash
LFE_CONFIG=gated_runs/extra24/config.yaml uv run --locked lfe evaluate gated_runs/extra24/run_folders/* --out gated_runs/extra24/scores
```

## Scored tables (`scores/`)

`summary.csv`, `per_conversation.csv`, `values.csv` and `calls.csv` from `lfe evaluate` for the official
SpokenWOZ runs, the 24 extra calls and AMI (the HTML reports are left out to keep the folder small).
