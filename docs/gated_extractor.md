# Gated extractor

`my_extractor:MyExtractor` listens while one extraction request per conversation is in flight. Its local
gate prioritizes likely booking values and corrections. For meeting notes, it batches evidence over a longer
window. A deadline releases pending evidence even if the speaker pauses. Identical partials and finished turns
are coalesced after a successful update. If the unfinished turn produced no update, the finished turn can be
reviewed again. At stream end the extractor drains the pending evidence.

Requests contain the form schema, current populated fields, recent context and unreviewed transcript. OpenRouter
receives a structured JSON schema for the delta. Each change includes a short transcript quote. The extractor
checks the quote, field type and value before updating the form. SpokenWOZ's explicit service requests can fill
the service field locally. A station spelling variant is accepted only when it maps to one listed option.
AMI note changes add or remove individual fact lines and preserve the other lines.

### Evidence rules added after the SNG1193 pilot

Every update still needs a quote found in the transcript. The check now also accepts equivalent wording:

- British and American spellings compare equal (`center` matches the option `centre`, `theater` matches `theatre`).
- "myself and five others" grounds 6 people, and rejects 5.
- A name said in parts ("first name is rebecca, my last name is louie") grounds "rebecca louie".
- A weekday is computed from the conversation's own reference: in "today is thursday, two days from now",
  saturday is grounded and thursday is rejected. With no offset, "today is thursday" can still be the day.
- The service field accepts "book a table", "eat" and similar wording for `restaurant`. The caller asking to
  book or reserve a table also selects `restaurant` locally.
- When the agent asks the caller to spell a name, a local decoder reads code words ("l like lima",
  "h as in hotel", "s for sierra") and plain letters, and publishes the corrected name when the agent moves on.
  Letters that fit inside the name already heard confirm it and change nothing. A spelled name is then locked
  against model replies that repeat the misheard form.

- A time stated without am/pm is shown as said ("quarter past 11" is 11:15 even if the model writes 23:15).
  "at night" and "tonight" mark the evening.
- Booking requests are spaced at least 1 s apart (was 4 s), and the gate treats spelling variants as options.
- Each dropped proposal is logged in `calls.jsonl` under `delta_diagnostics.dropped` with its reason.

### Rules added while tuning on Gemini 2.5 Flash

- Dictated numbers (`strategies/dictation.py`): the agent's ask ("may I have your id number") or the caller's
  "my telephone number is" picks the field. Each caller group extends the number as soon as it is heard, and
  "double 1" is 11. Agent echoes and backchannels keep the dictation open. While a number is being dictated,
  the model cannot write either number field. Afterwards it cannot copy those digits into the other field.
- "My name is mary edwards" fills an empty name at once; the model may still correct it.
- An agent answer to the caller's question ("what is the address?") is information, not a caller value.
- An area given as the opposite of another ("the opposite side of the north") grounds the other direction.
- An option the recognizer misspelled ("depart from kingsland") grounds `kings lynn` when that option is at
  least 0.75 similar and clearly closer than any other option. Options shorter than six letters never match.
- "looking for train" without a subject, and "a place to stay", select the service locally.
- A quote may differ from the transcript in one non-number word when it has five or more words.
- Meetings: a topic needs only a quoted passage, not its name. A meeting quote is also grounded by any run of
  six words in a row that the transcript says, so a quote that spans speakers still counts. Several notes for
  one field in one reply are merged.
- One timed-out request counts its reserve as spent and no longer stops the run. Three unresolved failures
  in a row still stop it.

`tests/test_extractor_rules.py` holds the regressions, with synthetic phrasings beyond the pilot calls.
`scripts/scripted_sng1193.py` replays SNG1193 with scripted replies that make the pilot's mistakes, so the
normal scorer can check the local rules end to end without a provider (`Scripted`), and runs the local rules
alone on any conversation to look for false fills (`LocalOnly`). Neither measures model accuracy.

## Offline check

From the repository root:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv run --locked --no-sync python -B -m scripts.check_gated_extractor
```

The smoke check injects an in-memory backend and feeds it the bundled transcript replays. It also checks
corrections, partial timing, final buffer handling, and note merges. Request counts measure scheduling.
Use the live pilot to measure accuracy and time to correct values.

## OpenRouter

Store `OPENROUTER_API_KEY` as an environment secret and allow outbound HTTPS to `openrouter.ai`.
OpenRouter's `GET /api/v1/key` endpoint can check access without making a completion.
The default model ID is `google/gemini-2.5-flash`, the hackathon baseline model (rates: input 0.30, output 2.50, cache read 0.03, cache write 0.0833 USD per million tokens). It uses the OpenRouter key.

Before approving a live run, set the following non-secret variables. Confirm the route's current rates with
OpenRouter for the selected model.

| Variable | Units or behavior |
|---|---|
| `LFE_OPENROUTER_INPUT_USD_PER_M` | USD per million uncached input tokens |
| `LFE_OPENROUTER_OUTPUT_USD_PER_M` | USD per million output tokens |
| `LFE_OPENROUTER_EUR_PER_USD` | EUR per USD for benchmark cost reports |
| `LFE_OPENROUTER_CACHE_READ_USD_PER_M` | Optional fallback rate if a response includes cache reads without a billed total |
| `LFE_OPENROUTER_CACHE_WRITE_USD_PER_M` | Optional fallback rate if a response includes cache writes without a billed total |
| `LFE_OPENROUTER_RUN_BUDGET_USD` | Local spending limit shared across conversations in one process. Default `2`. |
| `LFE_OPENROUTER_ENABLE_PAID` | Set to `1` only for an authorized live run |

The extractor logs completed attempts and uses OpenRouter's billed total for cost when supplied. The configured
rates are used for reservations and as a fallback. A request whose charge is unknown (a timeout or a missing
billed total) counts its full reserve as spent. Three in a row stop further requests. An interrupted request
can still bill at OpenRouter, but `max_tokens` and `max_price` bound it by the reserve. The local cap resets per process, so set
a provider-side key limit at the team's $20 ceiling for cumulative enforcement.

After live testing is authorized, start with `SNG0877`, the shortest bundled conversation at about 60 seconds.
The rates below are OpenRouter's standard Gemini 2.5 Flash route, the model every reported run used. Set
`LFE_OPENROUTER_EUR_PER_USD` to the exchange rate used for the comparison (0.86 in our reports) before running.

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export LFE_OPENROUTER_INPUT_USD_PER_M=0.30
export LFE_OPENROUTER_OUTPUT_USD_PER_M=2.50
export LFE_OPENROUTER_CACHE_READ_USD_PER_M=0.03
export LFE_OPENROUTER_CACHE_WRITE_USD_PER_M=0.0833333333333333
export LFE_OPENROUTER_EUR_PER_USD=0.86
LFE_OPENROUTER_ENABLE_PAID=1 LFE_OPENROUTER_RUN_BUDGET_USD=0.05 \
  uv run --locked lfe run my_extractor:MyExtractor --dataset spokenwoz --ids SNG0877 --name gated-pilot
```

Set `PILOT_RUN` to the run folder printed by the CLI, then score the saved artifacts.

```bash
uv run --locked lfe evaluate speechineer/*-spokenwoz "$PILOT_RUN" --out results/gated-pilot-comparison
```

Compare the two `SNG0877` rows in `per_conversation.csv` for cost, call count, speed, final F1, and
final decision latency. Check `calls.csv` for failed calls and unknown charges. The report's summary rows
cover the whole saved baseline and the single pilot, so use the per-conversation comparison for the pilot.
The `lfe check` CLI runs the extractor and can make paid calls when the backend is enabled.
