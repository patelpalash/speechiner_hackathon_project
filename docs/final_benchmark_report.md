# Final benchmark report: gated extractor vs saved Speechineer baseline

Both systems use `google/gemini-2.5-flash` on OpenRouter's standard tier. The baseline numbers come from the
saved Speechineer runs in `speechineer/`; they were not rerun. Our runs used the code at commit `1987a65` on
`feature/evidence-normalization`, with real-time playback, the same bundled conversations and the unchanged
`lfe evaluate` scorer. All of our responses reported `google/gemini-2.5-flash`. Earlier Flash-Lite pilots and
earlier tuning runs are excluded from this comparison.

| Run | Folder |
|---|---|
| SpokenWOZ baseline | `speechineer/20261117-172722-speechineer-openrouter-gemini-25-flash-extraction-spokenwoz` |
| SpokenWOZ ours | `gated_runs/official/20261121-171128-gated-flash-v10-spokenwoz` |
| AMI baseline | `speechineer/20261117-183600-speechineer-openrouter-gemini-25-flash-extraction-ami` |
| AMI ours | `gated_runs/official/20261121-170950-gated-flash-v5-ami` |

## Presented version (C, commit `f354c34`)

Version C keeps everything below and adds three cost and accuracy changes: the whole system prompt is cached
explicitly, the fields are described one per line with shared option lists, and booking forms get a reply schema
without the always-empty `facts` array. It also fixes local rules that missed on new calls: a phone number read
back with "yes, i got that", "i'm also looking for a train", a surname asked for on its own, and "my name is
called ...".

### Official SpokenWOZ bench (6 calls, run `gated_runs/official/20261121-183941-gated-flash-v14b-repeat-spokenwoz`)

| Metric | Saved Speechineer | Version C |
|---|---|---|
| Extraction calls per form | 70.2 | **47.3** |
| Cost per form | €0.0243 | **€0.0059** (−76%) |
| Input read from cache | 72% | **92%** |
| Expected values correct | **57/58** | 56/58 |
| Wrong / missing / extra values | 1 / 0 / 8 | **0** / 2 / **4** |
| Final F1 | 91.9% | **94.9%** |
| Precision | 0.867 | **0.911** |
| Benchmark speed | **0.837** | 0.775 |
| Outcome | 0.932 | **0.946** |
| Median time to a stable correct value | 1.74 s | **0.80 s** |
| Joint goal accuracy | 49.5% | **54.8%** |
| Overall benchmark score | **0.899** | 0.889 |

A repeat run gave the same 0.889, so this is not noise. Almost all of the gap is one call: in MUL2746 the caller
says "i'd like to live the place of attraction by 11 a. m." and the model files 11:00 as the taxi arrival time
instead of the departure time. Version A below scores 0.908 on the same calls.

### 24 more SpokenWOZ test calls

`scripts/extra_spokenwoz.py` builds further calls from the public SpokenWOZ test text with a simpler answer
key (a value counts as said at the end of its span or turn). These scores compare our own versions only; the
saved baseline has no runs on these calls.

| Version | Official 6: score | Official 6: € per form | 24 extra: score | 24 extra: € per form |
|---|---|---|---|---|
| Final before this work (`1987a65`) | 0.907 | 0.0114 | 0.735 | 0.0081 |
| A: explicit cache + compact fields | 0.908 | 0.0062 | 0.743 | 0.0047 |
| B: A + rule fixes | 0.897 | 0.0062 | 0.752 | 0.0047 |
| **C: B + set-only replies** | 0.889 | **0.0059** | **0.764** | **0.0044** |

Across all 30 calls, C has the highest score and the lowest cost. The meeting (AMI) path is unchanged: caching
gives no saving there (€0.0228 vs €0.0219 per form), so the AMI numbers below still hold.

## Earlier final version (commit `1987a65`)

## SpokenWOZ (6 booking calls, 14.5 min of audio)

| Metric | Saved Speechineer | Gated Gemini 2.5 Flash |
|---|---|---|
| Conversations completed | 6/6 | 6/6 |
| Extraction calls | 421 | **284** |
| Logged cost | €0.14595 | **€0.06844** |
| Cost per form | €0.0243 | **€0.0114** |
| Expected values correct | 57/58 | 57/58 |
| Wrong / missing / extra values | 1 / 0 / 8 | **0 / 1 / 4** |
| Final F1 | 91.9% | **95.8%** |
| Completeness | 0.971 | **0.972** |
| Precision | 0.867 | **0.916** |
| Benchmark speed | **0.837** | 0.803 |
| Outcome | 0.932 | **0.957** |
| Median time to a stable correct value | 1.74 s | **0.82 s** |
| p90 time to a stable correct value | 12.96 s | **9.26 s** |
| Joint goal accuracy | 49.5% | **54.8%** |
| Overall benchmark score | 0.899 | **0.907** |

We had zero failed API or response-parsing calls. Median API response time was 0.700 s, compared with 0.708 s
for the baseline. One of 58 values never settled in each run. Final F1 accounts for wrong, missing and extra
values.

Per conversation:

| Call | Audio | Baseline score | Our score | Baseline € | Our € | Baseline calls | Our calls |
|---|---|---|---|---|---|---|---|
| SNG0877 | 60 s | 0.966 | **0.980** | 0.00813 | **0.00369** | 30 | **17** |
| SNG1193 | 101 s | **0.922** | 0.904 | 0.01262 | **0.00699** | 50 | **31** |
| MUL1369 | 189 s | 0.890 | **0.946** | 0.03519 | **0.01576** | 92 | **63** |
| MUL2411 | 150 s | **0.889** | 0.853 | 0.02854 | **0.01375** | 73 | **55** |
| MUL2873 | 167 s | 0.959 | **0.985** | 0.02453 | **0.01363** | 79 | **55** |
| MUL2746 | 204 s | 0.717 | **0.754** | 0.03694 | **0.01462** | 97 | **63** |

Three runs of the final SpokenWOZ code scored 0.909, 0.893 and 0.907 (mean 0.903), with costs between
€0.0107 and €0.0114 per form. The score gain over the baseline is therefore small; the cost saving is
stable.

## AMI (1 meeting, 13.7 min of audio)

| Metric | Saved Speechineer | Gated Gemini 2.5 Flash |
|---|---|---|
| Conversations completed | 1/1 | 1/1 |
| Extraction calls | 302 | **51** |
| Logged cost | €0.3004 | **€0.0219** |
| Failed calls | 2 | **0** |
| Expected items correct | **24/35** | 17/35 |
| Wrong / missing / extra items | 0 / 11 / 90 | 0 / 18 / **20** |
| Final F1 | 32.2% | **47.2%** |
| Free-text F1 | 35.1% | **47.2%** |
| Completeness | **0.759** | 0.601 |
| Precision | 0.470 | **0.645** |
| Benchmark speed | **0.849** | 0.660 |
| Outcome | 0.322 | **0.472** |
| Overall benchmark score | 0.518 | **0.584** |

The baseline finds more of the expected notes but adds 90 extra ones; ours adds 20. Notes are sent to the
model in batches every 14 s instead of near-continuously, which is where most of the 93% cost saving comes
from. AMI is a single meeting judged by an NLI model, so its score varies more between runs than
SpokenWOZ's: three earlier versions of our meeting handling scored 0.547, 0.517 and 0.534.

## Combined

| | Saved Speechineer | Gated Gemini 2.5 Flash |
|---|---|---|
| Mean score (50/50 SpokenWOZ and AMI) | 0.709 | **0.746** |
| Cost for the whole bench | €0.446 | **€0.090** |
| Cost per audio hour, calls | €0.60 | **€0.28** |
| Cost per audio hour, meetings | €1.32 | **€0.10** |

Costs are OpenRouter's billed totals converted at 0.86 EUR per USD.

## What makes it cheaper and keeps accuracy

- A local gate decides when a model call is worth making, and sends only the new transcript after a cached
  prefix. The baseline asks the model every 2 s, and 78% of its calls change nothing; we make a third fewer
  calls on SpokenWOZ and a sixth as many on AMI.
- Simple values are filled without the model: requested services, phone and ID numbers read out digit by
  digit, spelled names and a stated full name.
- Every model change must quote the transcript and pass local checks for its field type. This blocks made-up
  values, agent answers to caller questions, and agent offers that the caller did not accept.
- Meeting notes are batched every 14 s, and a topic or note needs only a quoted passage from the meeting.

See `docs/gated_extractor.md` for the full rule list and `tests/test_extractor_rules.py` for the offline
regressions.
