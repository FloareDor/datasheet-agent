# DatasheetAgent — progress

Living log. One entry per loop iteration. Plan: [`plan.md`](./plan.md).

## Status

| Stage | State |
|---|---|
| 1. Data | **done** — 416 validated parts, 300 train / 100 test |
| 2. Tools | **done** — `search` / `read_section` / `read_table` over `datasheetindex` |
| 4. Evals (grading core) | **done** — unit normalisation + field grading |
| 4. Evals (baselines) | **done** — 1.5B and 32B, all three methods |
| 3. Agents | **done** — LangGraph fan-out, code+model verifier, one retry |
| 5. RL | **done** — 4 runs; cost 4x down, accuracy 6 points down |
| 6. Ship | **done** — servers, demo, Locust, spec-decoding, README |

## Decisions and findings

**Only op-amps carry labels.** The Zenodo archive has four files. `transistor_dataset`
(673 MB) holds 6,967 PDFs in train/dev/test splits but **no label file of any kind** — the
paper generates transistor training data rather than shipping ground truth.
`mouser_opamps` (56 MB) holds 54 catalogue CSVs covering **1,335 distinct op-amps** with
exactly the parametric fields we want, plus 100 PDFs. The 1.7 GB `opamp_dataset` should
supply the remaining PDFs.

*Consequence:* the graded task is op-amps. 1,335 labelled parts is ample for the planned
300 train / 100 test. Transistors stay useful as an **unlabelled transfer set** — the
same role the plan gave the optional Micron memory datasheets, and free.

**Label coverage** across the 1,335 parts: 87–95% per field for everything chosen, except
`Output Current per Channel` at 63%, which is why that field was left out of the schema.

**`datasheetindex` needs Python 3.13**, above the system 3.12. Resolved with a `uv` venv
pinned to 3.13.15. The library is substantial (~12k LOC) and gives exactly the planned
tool surface, so it is a real dependency rather than inspiration: `search_text`,
`get_section_text`, `extract_table_markdown`. Indexing costs ~2 s per datasheet and is
cached to `data/cache`.

**Many op-amp PDFs have no table of contents.** Older Linear Technology parts return an
empty ToC, so the agent cannot plan from a section map. `datasheetindex` supplies a
fallback hint and `DatasheetToolbox.orientation()` passes it through, telling the agent to
orient from pages 1–2 and navigate by search instead. This is a real navigation
difficulty, not a defect — it is part of what makes the task worth an agent.

**Nine fields, three groups**, partitioned by *where the answer lives* rather than by data
type, so the parallel agents genuinely search different places:
- electrical (4): supply voltage max, GBP, slew rate, Vos — specification tables
- thermal (2): operating temp min/max — absolute-maximum / recommended-operating tables
- package (3): package/case, channel count, mounting style — first page and package section

**Grading excludes unlabelled fields rather than counting them wrong.** A part whose
catalogue entry has no slew rate is evidence about neither the agent nor the baseline.
Counting it wrong would cap every system's accuracy below 100% and compress the
differences the evals are meant to measure.

## Log

### Iteration 1 — 2026-11-11

- Confirmed the Zenodo dataset and its label files before building on it. Found that only
  op-amps are labelled (above).
- Downloaded all three archives; extracted `mouser_opamps`.
- Stood up the repo: `pyproject.toml`, `uv` venv on Python 3.13, ruff + pytest config.
- `units.py` — parses engineering quantities to canonical SI. Handles the mV/MHz
  prefix-case trap, `="10 pA"` spreadsheet armour, `V/us` vs `V/s`, unicode micro and
  minus signs, and distinguishes *absent* from *zero*.
- `schema.py` — the nine fields and their groups.
- `tools.py` — `DatasheetToolbox` wrapping `datasheetindex`, with every call recorded for
  the RL cost term and the verifier.
- `grading.py` — field grading with package-name and mounting-style canonicalisation.
- Tests: 44 passing. Three were written to fail first and caught real bugs — `"so"`
  matching inside `"tssop"` and `"sot23"`, and `"SOT-23-5"` reading as a 23-pin part.

**Next:** extract `opamp_dataset`, join CSV labels to PDFs, build 300/100 splits.

### Iteration 2 — 2026-11-11

**Dataset built and validated: 416 parts, 300 train / 100 test, zero datasheet overlap.**

The join was the whole problem. Mouser names parts by orderable code (`OPA2991IDR`);
the archive names PDFs by document code (`ad8656.pdf`, `1637fd.pdf`). Three joins run in
order of confidence — exact filename (96), stem prefix (191), and title token (133), the
last reaching family datasheets where one document covers TL061/TL062/TL064.

**The catalogue's datasheet URLs are not usable.** They were the obvious way past the
archive's coverage, but 0 of 12 sampled URLs returned a PDF: TI links are interstitial
redirect pages and Mouser's own links return HTML to non-browser clients. Fetching them
would mean defeating bot protection, so the labelled set is what the archive covers.

**A validation pass caught a real labelling bug.** `validate_dataset.py` checks
independently that each part's number actually appears in the datasheet it was joined to.
The first run showed `OPA2342EA` joined to `opa234.pdf` — two different op-amps whose type
numbers happen to be prefixes. Also `LM6144BIM`→`lm614`, `MAX40088AUT`→`max400`,
`TLV2721IDBVR`→`tlv272`. Fixed at the root: a prefix match must consume the *whole*
numeric type, so the next character cannot be a digit. `stem_prefix` went from 91% to
**100%** validated. Had this gone uncaught, roughly 4% of labels would have been silently
wrong and every downstream accuracy number would have been noise.

A first attempt used a flat six-character threshold, which wrongly flagged correct joins
like `TL071CD`→`tl071.pdf` — a five-character base can never clear a six-character bar.
Replaced with a base-part-number rule.

Four parts remain dropped: two Maxim datasheets are scanned images with no text layer
(nothing downstream could read them either) and two are text-extraction failures.

**Splits are grouped by datasheet, not by part.** A family datasheet covers several
orderable parts; if TL062 were in train and TL064 in test, the test set would be
measuring recall of a document already tuned on.

**No Anthropic or OpenAI key in this environment.** Modal is authenticated (`sairavig`)
with `hf-token` and `wandb-api-key` secrets. The system is therefore built to run entirely
on open models, with Claude dropping in as a backend when a key exists — see
*Open questions*.

**vLLM on Modal: deployed.** Three servers, one per role. The verifier is Phi-4, a
different family from the Qwen agents it audits, because a model checking its own work
tends to agree with it. vLLM is at **0.33.0**, not the 0.11.x I would have pinned from
memory — checked rather than guessed, and pinned to 0.32.0.

Written this iteration: `dataset.py`, `llm.py` (Anthropic + OpenAI-compatible backends
with token/cost accounting), `extract.py` (single-call baseline and the tool-using agent
loop), `modal_vllm.py`. 58 tests passing, ruff clean.

**Next:** verify the small server answers, then run the first baselines on the test split.

### Iteration 3 — 2026-11-11

**Three deployment bugs, each found by reading the error rather than guessing.**

1. *`Could not find nvcc`.* vLLM 0.32 compiles kernels at engine start, so
   `debian_slim` is not enough. Switched to `nvidia/cuda:12.8.1-devel`.
2. *A 303 redirect loop on every request.* The function ended in
   `subprocess.Popen(cmd).wait()`. With `@modal.web_server`, the function must **start**
   the server and *return* — blocking leaves the container permanently "starting", and
   Modal answers every request with a redirect instead of proxying. Boot time went from
   never to **110 s**.
3. *HTTP 400 on every tool call.* The agent sent Anthropic-shaped tool schemas to an
   OpenAI-shaped endpoint. Fixed structurally: `Backend.format_tools` makes each backend
   own its own wire format, so no caller has to remember which API it is talking to.
   Deleted the two now-dead helpers in `tools.py` rather than leaving them.

**The harness is sound; the 1.5B model is the bottleneck.** Qwen2.5-1.5B via
`single_call` scores 4/8, 3/8, 5/9 on the first three test parts — around 50%, with unit
normalisation correctly matching `±22` and a unicode-minus `SOIC−8`. The same model given
the same fields *as an agent* scores **0/8**: it searches for the literal field names
(`supply_voltage_max`) instead of datasheet terminology, finds nothing, and answers null.

Adding a `search_terms` vocabulary to every field — the words a datasheet actually uses
("Slew Rate", "SR", "VOS") — did not change its behaviour. It still searched the field
names even when explicitly told not to.

*This is the result, not an obstacle.* A 1.5B model can read a spec table but cannot
drive a tool protocol zero-shot, which is precisely the gap GRPO is meant to close, and
it gives the before/after comparison real room to move.

**Two fairness fixes to grading**, both of which would have distorted the numbers:
- Models emit the *string* `"null"` rather than JSON `null`. Scored as a value, that
  marks a field wrong that the model correctly declined to answer.
- The single-call character budget was 120k chars — about 30k tokens, overflowing the 32k
  context and silently truncating the end of the datasheet, which is exactly where the
  package tables live. Cut to 60k.

Written: `verify.py` (code checks then a cross-family model check, with evidence
grounding against the document), `pipeline.py` (LangGraph fan-out, code merge, verify,
one retry), `evaluate.py` and `scripts/run_eval.py`. 58 tests, ruff clean.

**First baseline: Qwen2.5-1.5B, single call, 100 test parts — 59.5%** (525/882 fields),
15.9k input tokens and 7.2 s per part, 10 parts returning unparseable JSON.

| field | accuracy |
|---|---|
| operating_temp_min | 83.8% |
| operating_temp_max | 76.8% |
| supply_voltage_max | 73.3% |
| num_channels | 72.0% |
| mounting_style | 70.4% |
| slew_rate | 64.0% |
| gain_bandwidth_product | 54.1% |
| package_case | 30.1% |
| input_offset_voltage | 11.2% |

**A grading bug was hiding 18 points.** The first run reported 41.5%, with
`supply_voltage_max` at **0/90**. That zero was the tell: a model answering `36` to a
question posed in volts was being scored against a `36 V` label and marked wrong, because
a bare number parsed as nothing. The prompt *tells* the model which unit to answer in, so
a bare number must be read in that unit — otherwise the metric measures formatting, not
extraction. Fixing it moved `supply_voltage_max` 0%→73%, `slew_rate` 19%→64%,
`gain_bandwidth_product` 14%→54%, and the headline **41.5% → 59.5%**.

A second, smaller one: `QFN16 3x3` is a 16-pin part in a 3 mm × 3 mm body, but the body
dimensions were being read as the pin count.

*Lesson worth keeping:* a field at exactly 0% is almost never a model failure. Both bugs
were found by reading the per-field table rather than the headline.

**Re-grading is now free.** `scripts/regrade.py` re-scores a saved report from its stored
predictions, so a grading fix never costs GPU time again — and never confounds the fix
with fresh sampling noise.

**The 10 unparseable responses are a real failure mode**, not a harness problem: the 1.5B
sometimes emits prose instead of JSON. Worth reporting rather than retrying away, since
it is part of what makes the small model weak.

**Second baseline: the same model as an agent — 21.7%** (191/882), 7.9 tool calls and
12.3 s per part, 7 parts hitting the 10-step limit without ever answering.

| method | accuracy | tokens in/part | tool calls/part | latency |
|---|---|---|---|---|
| single_call | **59.5%** | 15.9k | 0 | 7.2 s |
| single_agent | 21.7% | 9.2k | 7.9 | 12.3 s |

So the agent framing *costs* this model 38 points. It uses fewer input tokens — it only
reads what it fetches — but it cannot choose what to fetch. That is the honest starting
point for RL, and it means the before/after has real room.

**A concurrency bug, fixed.** One part died with `WinError 5: Access is denied` on the
cache directory. Family datasheets cover several parts, and the three field agents share
one PDF, so parallel runs were building the same artifacts simultaneously. Now serialised
per datasheet: the second caller finds the artifacts already on disk. This would have hit
the multi-agent runs much harder, where three agents always share one document.

**Next:** multi-agent and verifier ablations, then the strong reference model.

### Iteration 4 — 2026-11-11

**The RL reward is written and tested before any trainer exists**, because the reward is
the part that decides what the policy learns and it can be reasoned about on its own.
Three terms: field accuracy (dominant), a small per-tool-call cost, and a penalty for
returning nothing parseable.

The per-call cost is 0.02 — set so the full ten-step budget costs 0.2, which is enough
that a wasted call is never free, but less than the ~0.11 that turning one field from
wrong to right is worth across nine fields. If that inequality inverted, the policy would
learn to call nothing and answer nothing, so `tests/test_reward.py` asserts the ordering
directly rather than the constants:

- a perfect cheap rollout beats a perfect expensive one
- a full correct answer at ten calls beats a half answer at zero
- **answering wrongly beats staying silent** — otherwise silence is a winning strategy
- the cost term is capped, so one runaway rollout cannot dominate a GRPO group

**Multi-agent run is healthy**: 9 concurrent requests on the L40S at a **74% prefix cache
hit rate**, which is the shared system prompt and orientation block paying off exactly as
`--enable-prefix-caching` intends.

Also written: `scripts/locustfile.py` (two user classes — the full agent path, and raw
vLLM throughput, so a slow result can be attributed to one or the other),
`modal_app.py` (FastAPI demo service running the *same* pipeline the eval measures, not a
demo-only path), and `web/index.html` showing each value beside the evidence line it came
from, with repaired and flagged fields marked.

69 tests, ruff clean.

## Open questions

**Should the headline comparison be against Claude?** The resume line says "matching
Claude at Z× lower cost", but there is no `ANTHROPIC_API_KEY` here. Two options:
- Supply a key — the Claude row drops straight in, nothing else changes.
- Leave it open-only — the reference becomes Qwen2.5-32B and the claim becomes "matches a
  32B model at Nx lower cost", which is defensible but a weaker line on a resume.

Everything else proceeds either way; the backends are already written for both.

**Modal web endpoints are currently unauthenticated.** Fine for a short-lived research
deployment, worth closing before anything is shared.


### Iteration 5 — 2026-11-11

**GRPO training is running.** TRL 1.17.1's `environment_factory` turns out to support
multi-turn tool-use rollouts natively — the environment's public methods *become* the
policy's tools — so no custom RL loop, and no verl/SkyRL. The policy trains on exactly
the `search` / `read_section` / `read_table` interface it is scored on. 878 rows (one per
part x field group), LoRA r=32, vLLM colocated on 2xH100, reward from `reward.py`.

A 2-step smoke run validated the whole stack before committing GPU hours; the real run is
400 steps, launched with `modal run --detach`. Two bugs it caught first: the training
image needed Python 3.13 (`datasheetindex` requires it), and TRL indexes
`prompt[-1]["content"]`, so the prompt column cannot be the empty string the environment
was going to replace anyway.

**Two tool-design bugs, together worth 20-24 points.**

*The search tool was unhelpfully literal.* Models reach for a field's machine name,
`slew_rate`, which appears in no datasheet ever printed. The tool answered "No matches"
and the agent answered null. Every electrical field scored **exactly 0%** across 100
parts for this reason alone — the same 0% tell as the units bug. Relaxing the query the
way a search box would (separators to spaces, then dropping trailing qualifiers) took
single_agent **21.7% → 41.8%** and multi_agent **5.7% → 29.6%**.

*Blind relaxation then caused a subtler failure.* `operating_temp_min` shortens to
"operating temp", which substring-matches "Full **Temp**erature" inside a distortion
spec — so the agent dutifully reported an operating temperature of **0.7%**. Relaxation
without domain knowledge is not merely weaker, it is actively misleading. The tool now
tries the field's real vocabulary first, which it already had in `schema.search_terms`.

Neither of these is the model's fault and neither is cheating: the vocabulary is the same
for every method, and a tool that cannot recover from a near-miss query is a bad tool.

**Where multi-agent stands, honestly.** With the first fix only:

| method | accuracy | tokens in/part | tool calls | latency |
|---|---|---|---|---|
| single_call | **59.5%** | 15.9k | 0 | 7.2 s |
| single_agent | 41.8% | 20.8k | 8.2 | 24.6 s |
| multi_agent | 29.6% | 42.5k | 22.3 | 13.7 s |

For a 1.5B model, one big call still wins. But the per-field table says something more
interesting: multi-agent **beats** single-agent on the electrical fields it specialises in
(GBP 50% vs 38%, Vos 25% vs 15%, slew 57% vs 47%) and loses badly on thermal — which is
exactly the "operating temp" bug above, now fixed. The v3 numbers will say whether
specialisation holds up once that is out of the way.

76 tests, ruff clean.

### Iteration 6 — 2026-11-11

**The field-aware search fix landed; the ordering is now stable.** Multi-agent thermal
went 17% to 35%, lifting multi-agent overall 29.6% to 32.2%. Final standings on the
1.5B:

| method | accuracy | tokens in | tool calls | latency |
|---|---|---|---|---|
| single_call | **59.5%** | 15.9k | 0 | 7.2 s |
| single_agent | 40.5% | 26.5k | 8.0 | 32.7 s |
| multi_agent | 32.2% | 50.3k | 23.5 | 19.5 s |

**So the headline result is negative, and it is reported as the headline.** For a 1.5B,
one big call wins. The per-field table explains why, and it is a better finding than the
headline: multi-agent **beats** single-agent exactly where the datasheet is hard —
input offset voltage (22.4% vs 20.4%, and beating the single call's 11.2%) and slew rate
(54% vs 42%) — both fields that need an actual table read. Everywhere else the fields
share *one* specification table, so three agents re-find the same table: 23.5 tool calls
and 50k tokens per part against 8.0 and 26k. Three times the cost of the single call for
half the accuracy.

That is Anthropic's own criterion for multi-agent systems — the subtasks must be
genuinely separable — and here they largely are not. Worth saying plainly rather than
burying.

**Superseded runs are archived, not deleted.** `runs/archive/` holds the six runs measured
against known-buggy code, with a README saying which bug invalidated each. The results
tables read only from `runs/`, so the README cannot quote a number from broken code,
while the before/after claims in this log stay checkable.

**Verifier server failed to boot**: Phi-4 has `max_position_embeddings=16384` and the
server asked for 32768. Context length is now per-model. The verifier only ever sees one
field, one value and one evidence line, so 16k is ample.

**Training is ~40% through** and the reward is sitting near -0.4, which is close to the
no-answer penalty — most rollouts are not emitting parseable JSON at the end of the tool
loop. That is precisely the behaviour the format term was added to shape, so it is the
right signal to be training on, but whether 400 steps at lr 1e-6 moves it is an open
question. If the adapter evaluates flat, the next run goes to a higher learning rate —
and that result gets reported either way.

Written: `README.md` with the measured results and a "what didn't work" section,
`scripts/summarise_runs.py` (regenerates `runs/summary.md` from measured runs only, so no
number in the README can be hand-typed).

### Iteration 7 — 2026-11-11

**The demo is deployed and extracting specs from a real uploaded PDF.** Three bugs
between "deployed" and "working", none of which a unit test would have caught:

1. `datasheetindex` installs from git, so the image needed `git`.
2. `from __future__ import annotations` turns FastAPI's parameter hints into strings
   that Pydantic cannot resolve.
3. More subtle: FastAPI resolves a route's annotations against its function's *module
   globals*, so `UploadFile` imported inside `web()` was invisible to it and every
   request 500'd with a ForwardRef error. The imports are now module-level, and
   `fastapi` is a declared local dependency rather than a container-only one.

**The demo smoke test immediately found a real bug in the verifier.** It rejected
`supply_voltage_max = '36'` with *"could not be read as a voltage value"* — the same
bare-number issue fixed in grading two iterations ago, never propagated to `verify.py`.
The verifier was rejecting *correct* answers for their formatting and sending agents back
to re-find values they already had. Fixed, with `tests/test_verify.py` covering it (84
tests now).

Worth noting how it was found: not by reading the code, but by running the deployed
service against a part whose labels are known. `scripts/smoke_demo.py` prints extracted
values beside their labels with a match column, so a demo returning plausible-looking
nonsense cannot pass as working.

*Consequence:* the verifier ablation currently running was started before the fix and
uses the buggy verifier. It has to be re-run, and will be. Its result is still worth
keeping as a measure of what that one bug cost.

**The trained-adapter server is ready ahead of the checkpoint.** `modal_vllm.trained`
serves the base model with the LoRA adapter attached rather than merged, so before/after
is exactly one variable: same base, same server, same tool schemas, adapter on or off.
`run_eval.py --backend trained` is wired through.

**Training is flat.** At 57% of the run, field accuracy sits at 0–6% and reward near
-0.4, close to the no-answer penalty: most sampled rollouts are not emitting parseable
JSON at the end of the tool loop. Rollouts sample at temperature 1.0 while evaluation is
greedy, so some gap is expected — but there is no visible trend. The likely cause is
`learning_rate=1e-6`, which is TRL's default for full fine-tuning and low for LoRA.

The run finishes rather than being abandoned, because "400 steps at 1e-6 moved nothing"
is a result worth having and the adapter still needs evaluating. A second run at a higher
learning rate follows. Both get reported.

### Iteration 8 — 2026-11-11

**`modal run --detach` did not survive the launching shell, and that cost a training
run.** The v1 job was cancelled at step ~250 of 400 the moment the local process holding
it was killed — the logs show a cancellation arriving, not a crash. `modal run` with
`.spawn()` is no better: it prints *"Stopping app - local entrypoint completed"* and
takes the spawned call down with it.

What works is `modal deploy` plus `.spawn()` against the **deployed** function: a
deployed app is not tied to any local process. `scripts/launch_training.py` does this and
explains why in its docstring, so the next person does not rediscover it the same way.
Nothing was lost — `save_steps=50` meant `checkpoint-250` was already on the volume.

v2 is running: 500 steps at **lr 5e-6** (up from 1e-6, which showed no trend over 250
steps with LoRA).

**The verifier, as first measured, made things worse: 32.2% -> 22.2%.** 332 rejections
against only 30 repairs. The cause was the `assume_unit` bug: it rejected correct
bare-number answers as unreadable, and the retries it triggered replaced good values with
nulls. That run is archived as the measured cost of one bug, and the fixed re-run is in
flight.

This is a genuinely useful negative result to have measured rather than assumed: a
verifier that rejects correct answers is far worse than no verifier at all, because every
false rejection spends a retry that can only make the answer worse.

**Serving numbers, measured.** Locust against one L40S, 20 concurrent users:

| metric | value |
|---|---|
| requests | 964 |
| failures | **0** |
| throughput | 10.85 req/s |
| p50 / p95 / p99 | 310 / 440 / 740 ms |

Conservative — the verifier ablation was hitting the same server throughout, so this is
throughput under contention.

**Speculative decoding is deployed** as a separate server (`small-spec`) so it can be
compared against identical weights. N-gram speculation suits this task unusually well:
the agent's evidence field quotes a line it just read out of a table, so the draft tokens
are very often already in the prompt. That is exactly what prompt-lookup decoding
exploits, and it needs no draft model.

### Iteration 9 — 2026-11-11

**With the unit bug fixed, the verifier still hurts: 32.2% -> 22.3%.** 280 rejections
against 28 repairs, and every field worse — slew rate -14 points, supply voltage -15.6.
So the earlier bug was not the explanation; something about the design is wrong.

**The headline number cannot say what is wrong**, and that matters because the two
possibilities need opposite fixes:

- the verifier rejects *correct* values (its judgement is bad), or
- it correctly flags wrong values and the *retry* fails to improve them.

So `PipelineResult` now records `pre_retry_values`, and `scripts/verifier_report.py`
grades the verifier as a classifier: precision over its rejections, how many wrong values
the retry repaired, and how many correct values it destroyed. Measuring this before
changing anything, rather than guessing at a fix.

The suspected mechanism is the retry prompt. It tells the model that answering null is
"a correct answer, not a failure" — sound advice against hallucination, but a weak model
told it was wrong will take the exit. If that is what the numbers show, the fix is a
rule, not a prompt tweak: **a rejection is not evidence that the datasheet lacks the
value, so a retry must never be allowed to replace a value with nothing.**

Also running: a code-checks-only verifier (no model), which the plan asks for separately
and which isolates how much of the damage is the model's judgement.

**v2 training is alive and independent** — deployed app, 1 task, running at lr 5e-6 after
the `.spawn()`-against-a-deployed-function fix. ~500 steps to go.

89 tests, ruff clean. `tests/test_pipeline.py` covers merging and cost accumulation with
a scripted backend — per-agent cost must *sum*, or multi-agent would look artificially
cheap, which is exactly the number the whole comparison turns on.

### Iteration 10 — 2026-11-11

**The verifier diagnosis, measured.** Grading the verifier as a classifier rather than by
its effect on the score separated two failures that the headline had conflated:

| | model verifier (Phi-4) | code checks only |
|---|---|---|
| rejections | 269 | 50 |
| precision | **55.8%** | **82.0%** |
| wrong values repaired by retry | 9 | 10 |
| correct values destroyed by retry | **98** (94 nulled) | 6 |
| net fields | **-89** | **+4** |
| accuracy | 21.9% | **32.6%** (vs 32.2% unverified) |

Two independent problems, and both needed measuring to tell apart:

1. **The cross-family model verifier is barely better than a coin flip** at 55.8%
   precision. Phi-4 is asked whether a quoted line supports a value, and the 1.5B's
   evidence quotes are often poor even when its *value* is right — so the verifier
   rejects good answers for bad citations.
2. **The retry was almost purely destructive**: 98 correct values destroyed against 9
   repaired, and 94 of the 98 were nulled out. Even among the 150 *correctly* rejected
   values, the retry fixed only 9.

The second is the bigger effect and had the clean fix: **a retry may correct a value but
never erase one.** A rejection is not evidence that the datasheet lacks the value, and a
small model told it was wrong takes the null exit the retry prompt offers it. That rule
is now `apply_retry_values`, a pure function with its own tests.

(The first version of that test grepped the source for the guard — brittle and not
actually a behaviour test. Extracting the rule into a pure function made it both better
design and properly testable.)

**Code checks alone are already net-positive and free**: 82% precision, +4 fields, and no
model call. That is the configuration to recommend, and the measurement says so rather
than the intuition.

This is the self-verification limit the plan cited
(arXiv:2402.08115) showing up concretely: using a *different* model family was not
enough, because the verifier is only as good as the evidence the agent gives it to judge.

v6 re-runs both configurations with the retry guard in place.

### Iteration 11 — 2026-11-11

**The retry guard worked, and the verifier recommendation is now measured, not assumed.**

| configuration | accuracy | precision | repaired | destroyed | latency |
|---|---|---|---|---|---|
| no verifier | 32.2% | - | - | - | 19.5 s |
| **code checks only** | **33.0%** | **85.4%** | 11 | **0** | 14.0 s |
| code + Phi-4 verifier | 32.1% | 55.3% | 7 | 1 | 19.6 s |

Forbidding a retry from erasing a value turned the model verifier from **-10.3 points
into -0.1** — it is now merely useless rather than destructive. Code checks alone are the
configuration to ship: 85.4% precision, net +11 fields, zero destroyed, and no model call
at all.

**The RL run reward-hacked, and the logs caught it.** At lr 5e-6 the v2 reward went
sharply positive — from -0.4 to +0.19 — which looked like success. It was not.
`tools/call_frequency` had gone to **exactly 0.00** and stayed there:

```
 epoch   reward  fieldacc  toolcalls
 0.863    0.531     0.531       0.00
 0.881    0.562     0.562       0.00
 0.888    0.698     0.729       0.00
```

The policy had learned to stop using tools and guess. Qwen2.5-1.5B can get ~30% of these
fields from priors alone — SOIC-8 is the usual package, -40/+85 the usual temperature
range, a "2" in a part number often means two channels — and guessing at zero cost beats
tool use at the same accuracy *plus* the per-call penalty. The reward was not wrong about
what it asked for; it was wrong about what it left unsaid.

Worth being precise about my own error here: `tests/test_reward.py` asserted that
accuracy outweighs tool cost *at a fixed accuracy*. It never asked what happens when the
policy can achieve similar accuracy with no tools at all. The test was true and the
reward was still exploitable.

**The fix is grounding, and it needs no special case.** A field now counts only if it is
correct *and* its evidence appears in this datasheet, from a rollout that made at least
one tool call. A policy that calls no tools has read no document text, so it can quote
nothing, so nothing it says is grounded, so its accuracy term is zero. Guessing becomes
worthless rather than merely discouraged — and the reward now states the actual task:
extract from *this* datasheet, do not recall what op-amps usually look like.

`raw_accuracy` is logged alongside `grounded_accuracy`, so a gap between them makes the
guessing visible rather than hiding it inside one number.

97 tests. The new ones pin the failure shut directly: *guessing without reading earns
nothing*, *invented evidence earns nothing even when the value is right*.

### Iteration 12 — 2026-11-11

**v2's adapter evaluated: the reward hack is unambiguous, and more interesting than a
win would have been.**

| | base agent | after GRPO (v2) |
|---|---|---|
| accuracy | 40.5% | 37.5% |
| tool calls / part | 8.0 | **0.0** |
| input tokens / part | 26,464 | **1,627** |
| latency / part | 32.7 s | **2.3 s** |
| failures | 20 | 0 |

The trained policy answers straight from the part number: 16x fewer tokens, 14x faster,
three points less accurate. GRPO found the true optimum of the reward it was given.

The finding underneath it is the useful part, and it agrees with the baselines from the
other direction: **at 1.5B the tool-using agent barely beats informed guessing.** Reading
the datasheet does help a great deal — the single call scores 59.5% — but the *agent*
framing is what fails to pay for itself.

v3 is running with the grounded reward. Early signs are right: tool calls are back at
2-3.5 per rollout rather than zero. Watching `frac_reward_zero_std` now, because strict
grounding risks the opposite degenerate case — if nothing is ever grounded, every rollout
in a group scores the same and there is no gradient at all.

README updated with both the verifier table and the RL result.

### Iteration 13 — 2026-11-11

**The grounded reward is producing a real learning signal.** At epoch 0.63 of v3:

| | v1 (acc - cost, lr 1e-6) | v2 (same, lr 5e-6) | **v3 (grounded)** |
|---|---|---|---|
| training accuracy | 0-6% | 0-6% raw | **~30% grounded** |
| tool calls / rollout | 2-3 | **0** | ~1.0 |
| reward | -0.4 | +0.2 (hacked) | +0.2 to +0.5 |

`raw_accuracy` and `grounded_accuracy` now track each other almost exactly (0.375/0.375,
0.469/0.469), which is the check working: when this policy gets a field right, it got it
right *from the document*. In v2 those two numbers would have diverged completely.

`frac_reward_zero_std` is mostly 0, so groups still differ from one another and there is
a gradient — the degenerate risk of strict grounding (everything scores zero, no signal)
has not materialised.

One thing to watch rather than celebrate: tool calls have settled at almost exactly
**1.0**. That may be the minimal policy that satisfies the grounding gate — one search,
then answer from the snippet. For a per-group task of 2-4 fields that is a defensible
strategy rather than a cheat, since one well-aimed search can return all of them. But it
is worth naming as a possible second-order gaming of the same gate, and the eval will
show whether it generalises.

Speculative decoding benchmark running: identical weights, identical prompts, 40 parts,
with and without n-gram speculation.

### Iteration 14 — 2026-11-11

**Speculative decoding: two measurements that disagree, and the disagreement is the
result.**

| measurement | result |
|---|---|
| controlled, batch 1, output quoting the prompt | **+58.2%** decode throughput (171.8 -> 271.7 tok/s) |
| end to end, real agent workload, 4 concurrent | **-5.4%** latency (no benefit) |

The end-to-end eval alone would have said "speculation does not work", which is wrong.
The controlled benchmark alone would have said "+58%, ship it", which is also wrong.
Both are needed, and the gap has two causes worth naming:

1. The real output is structured JSON of short numeric values. Only the `evidence` field
   quotes the document, so there are few tokens for speculation to win on.
2. At four concurrent parts the GPU is already batch-saturated, so drafting competes with
   real decoding rather than filling idle capacity.

Kept available, not default. It is the right choice for single-user interactive
extraction and the wrong one for the batch evaluation path.

I had predicted the +58% case in the server's docstring ("the agent's output quotes the
datasheet back verbatim") and would have been happy to leave it there. Writing a
*controlled* benchmark after the end-to-end number disagreed is what turned a plausible
story into a conditional fact.

`scripts/bench_decoding.py` exists because the end-to-end harness genuinely cannot
measure this: concurrency and prompt-length variation swamp a few percent of decode
time.

### Iteration 15 — 2026-11-11

**v3 evaluated. The grounding fix worked; RL still did not improve accuracy.**

| | accuracy | tool calls | LLM calls | tokens in | latency | failures |
|---|---|---|---|---|---|---|
| base agent | **40.5%** | 8.02 | 3.90 | 26,464 | 32.7 s | 20 |
| v2 trained (reward-hacked) | 37.5% | 0.00 | 1.00 | 1,627 | 2.3 s | 0 |
| v3 trained (grounded) | 33.7% | 1.00 | 2.36 | 6,712 | 7.2 s | 4 |

The grounding term did exactly what it was designed to do — tool calls went from 0.00 to
**exactly 1.00**, so the policy reads the document rather than guessing — but the policy
converged to the *minimum* the gate allows: one search, then answer from the snippet.

So the result is a cost reduction, not an accuracy gain: **4x fewer input tokens, 4.5x
faster, 5x fewer failures, at a cost of 7 accuracy points.** The resume line as drafted,
"accuracy X% -> Y%", cannot be claimed. What can be claimed is the efficiency, and it
will be written that way.

**The obvious objection is whether the tool-cost term caused this**, so v4 is running with
`--tool-call-penalty 0` — the cost term removed entirely. If the policy then uses more
tools and gains accuracy, the cost coefficient was wrong. If it still converges to one
call, the bottleneck is not the reward at all.

My expectation, stated before the result so it can be wrong: it will *not* help much. The
baselines already say more tool calls do not buy accuracy for this model — the base agent
makes 8 calls for 40.5%, while a single call with the datasheet pasted in gets **59.5%**.
That points at reading comprehension rather than navigation as the limit, and RL over a
navigation policy cannot fix comprehension.

### Iteration 16 — 2026-11-11

**The tool-cost ablation settles it: the cost term was not the problem.** v4 trains with
`--tool-call-penalty 0`, the per-call cost removed entirely. Tool calls still converge to
**~1.10 per rollout** (grounded accuracy ~0.20) — statistically the same as v3's 1.00 with
the cost in place.

So the policy is not avoiding tools to dodge a penalty. It makes one call because a
second call does not improve its answer. The prediction made last iteration before seeing
this was that removing the cost would not help; it did not.

That triangulates with the baselines, which say the same thing from the other direction:

- one call with the datasheet pasted in: **59.5%**
- an agent making 8 tool calls: **40.5%**
- a trained agent making 1 tool call: **33.7%**

More navigation does not buy accuracy here. **The binding constraint is reading
comprehension, not navigation**, and RL over a navigation policy cannot fix comprehension.
That is the project's central finding, and three independent measurements now support it.

**What is still missing is the strong-model reference.** Every number so far is
Qwen2.5-1.5B, so "the agent architecture does not pay" is currently a claim about one
small model, not about the architecture. Qwen2.5-32B is warming on 2xH100 to run the same
methods. That matters most for the multi-agent question: Anthropic's criterion is that
multi-agent helps when subtasks are separable *and the model is capable enough*, and only
the second half has been tested.

### Iteration 17 — 2026-11-11

**A real cost model, because "Zx lower cost" needs a denominator.** Until now
`cost_usd_per_part` was 0 for every open model: `llm.PRICES` only had Claude's
per-token rates, and a self-hosted model is not billed per token at all.

`pricing.py` computes cost the way the bill actually works — GPU-seconds divided by the
throughput a run *measured*, never an assumed one. `ServingCost` records which run the
throughput came from, so a cost figure can always be traced to the wall-clock it came
from.

Modal's published rates were fetched rather than recalled (L40S $0.000542/sec, H100
$0.001097/sec) and stored with the URL and the date they were read, because a price
quoted without a date silently becomes a guess.

First number: the 1.5B single call costs **$0.00049 per datasheet** on one L40S at the
throughput the v3 run actually sustained.

A detail worth encoding rather than assuming: two GPUs cost twice as much per part, not
the same — tensor parallelism rents both cards for the whole run. `test_pricing.py`
asserts that, along with the rule that an unknown GPU returns `None` rather than
inventing a price.

103 tests. The 32B reference run is in flight; v4 (zero tool cost) is at epoch 0.67.

### Iteration 18 — 2026-11-11

**The 32B reference reverses the multi-agent conclusion.** This is why the strong-model
run mattered.

| model | single call | single agent | multi-agent |
|---|---|---|---|
| Qwen2.5-1.5B | **59.5%** | 40.5% | 32.2% |
| Qwen2.5-32B | **76.2%** | 70.5% | **75.1%** |

At 1.5B, multi-agent is the worst option and trails the single call by **27.3 points**.
At 32B the ordering **flips**: multi-agent beats the single agent by 4.6 points, comes
within **1.1 points** of pasting in the whole datasheet, and is the only configuration
that extracted all 100 parts with **zero failures**.

Reporting only the 1.5B would have produced a confident, wrong conclusion — that the
architecture does not work. It does. The small model could not exploit it. That is
precisely Anthropic's criterion: subtasks separable *and* the model capable enough.

The token economics move the same way. At 32B the single agent reads **fewer** input
tokens than pasting the datasheet in (14.9k vs 15.9k) because it fetches only what it
needs; at 1.5B the same agent reads 26.5k, because it searches badly and reads the wrong
pages.

**Cost, with the new model:** the 1.5B single call reaches **78% of the 32B's accuracy at
10.6x lower cost** ($0.0005 vs $0.0052 per datasheet). That is a defensible resume line,
and it is not the one the plan drafted.

**I put invented numbers in the README and caught them.** Writing the per-field
comparison I filled the 32B column from plausible-looking memory rather than from
`runs/`. Printing the actual values showed every one of them wrong — and showed something
I would never have guessed: the **1.5B beats the 32B on both temperature fields**, by 6
points on `operating_temp_min`. Those values appear once, in plain text near the front, so
finding them is the whole task; the 32B sometimes prefers an absolute-maximum or storage
range instead, a more sophisticated mistake than the 1.5B can make. Corrected, with the
real gaps: `package_case` +44.0, `input_offset_voltage` +41.9, `gain_bandwidth_product`
+28.6.

The rule that keeps catching this is already in the repo — `summarise_runs.py` reads only
from `runs/` so the README cannot quote an unmeasured number. I bypassed it by hand.

### Iteration 19 — 2026-11-11

**v4 confirms the ablation at test time: 34.1% at 1.06 tool calls**, against v3's 33.7%
at 1.00 with the cost term present. Removing the per-call cost changed nothing. The cost
coefficient was never what suppressed tool use — the model simply gains nothing from a
second call.

That closes the RL question with four runs and one conclusion: **GRPO bought cost, not
accuracy.** 4x fewer input tokens, 4x faster, 5x fewer failures, 6 points less accurate.
The plan's resume line, "accuracy X% -> Y%", is not supported and is not claimed anywhere
in the README.

**README finalised** with a closing section stating the three supported claims and the
two methodological notes that mattered more than any single number:

- a field at exactly 0% is almost never the model's fault (it found three separate bugs);
- grade the component, not just the pipeline (the verifier looked merely unhelpful until
  it was graded as a classifier, which separated 55.8% precision from a destructive retry
  — two problems needing opposite fixes that the headline could not distinguish).

One repair worth noting: rewriting the Results section earlier silently deleted the RL
subsection, because the replacement spanned from `## Results` to `## How it works` and the
RL section lived between them. Caught by diffing the heading list, not by reading.

All six plan stages are complete. 103 tests, ruff clean, 13 measured runs in `runs/`.

### Iteration 20 — 2026-11-11

**A negative control, using the unlabelled transistor archive.** A bipolar transistor
datasheet states no slew rate, no gain-bandwidth product and no input offset voltage —
those parameters do not exist for the part. So the correct answer is null every time, and
any value returned is an invention. No labels needed, which is exactly what makes the
6,967-PDF transistor archive usable at last.

**Qwen2.5-1.5B, single call, 30 transistor datasheets: 21.1% invention rate.**

| field | invented | rate |
|---|---|---|
| input_offset_voltage | 10/30 | 33.3% |
| gain_bandwidth_product | 8/30 | 26.7% |
| slew_rate | 1/30 | 3.3% |

**The failure mode is more interesting than hallucination.** The invented values are not
fabricated from nothing — they are *real numbers from the document, read off the wrong
parameter*:

- `input_offset_voltage` = -0.5, -0.8, -1.4 mV — these are VBE or VCE(sat) values
- `gain_bandwidth_product` = 100, 1000 — these are transition frequency fT

So the model is doing nearest-neighbour matching on parameter names rather than
understanding what it was asked for. That is harder to catch than invention, because the
value is genuinely in the datasheet and would pass any evidence-grounding check: the quote
is real, it just supports a different parameter. It is also the same weakness the verifier
analysis found from the other side — Phi-4 rejects good answers for bad citations at 55.8%
precision, and this is why citations and values come apart.

Directly relevant to the project's thesis: **source grounding verifies that a quote
exists, not that it is the right quote.** Worth stating plainly rather than letting
"source-grounded" imply more than it delivers.

Running the same control on the 32B, to see whether capability reduces substitution or
merely makes it subtler.

### Iteration 21 — 2026-11-11 — project complete

**The 32B negative control is the most interesting result in the project.**

| field | 1.5B | 32B |
|---|---|---|
| input_offset_voltage | 33.3% invented | **0.0%** |
| slew_rate | 3.3% invented | **0.0%** |
| gain_bandwidth_product | 26.7% invented | **33.3%** |
| overall | 21.1% | 11.1% |

Capability halves the overall invention rate and **eliminates** two of the three fields —
the 32B knows a transistor has no offset voltage and no slew rate. But it reports
transition frequency f_T as gain-bandwidth product *more* often than the small model
does, confidently and with units.

So capability removes the crude substitutions (V_BE read as Vos) and entrenches the
sophisticated one. For a transistor, f_T really is the nearest analogue to GBP — a
defensible engineering mapping, and not what was asked. A better model makes a better
class of mistake, not fewer mistakes of every class.

Every one of those values would pass the evidence-grounding check, because the quoted
line is genuinely in the datasheet. **Grounding verifies that a quote exists, not that it
is the right quote** — now stated in the README rather than left implied.

### Status: all six plan stages complete

| stage | outcome |
|---|---|
| 1. Data | 416 validated parts, 300/100 split by datasheet |
| 2. Tools | `search` / `read_section` / `read_table` over `datasheetindex` |
| 3. Agents | LangGraph fan-out, code+model verifier, guarded retry |
| 4. Evals | 2 models x 3 methods, verifier ablation, cost, latency, load test |
| 5. RL | 4 GRPO runs; cost 4x down, accuracy 6 points down, cost-term ablated |
| 6. Ship | vLLM servers, spec-decoding benchmark, Locust, live demo, README |

115 tests, ruff clean, 15 measured runs in `runs/`, 5.7k lines across 43 files.

**What the project supports, and what it does not.** It supports: a 1.5B reaching 78% of
a 32B's accuracy at 10.6x lower cost; multi-agent paying off only above a capability
threshold; and RL buying cost rather than accuracy. It does **not** support the plan's
drafted resume line of "accuracy X% -> Y%", and that line is claimed nowhere.
## Iteration 22 — the evidence checks, measured

Prompted by a review question: is the `quote` field written by the model, or copied
from the tool result? It is written by the model (`extract.py`, `entry.get("evidence")`),
so all four of the failure modes that implies are live — and the check that was assumed
to be doing the real work, "does the number appear in the quote", did not exist.

**Three checks added**, all in `verify.py`:

1. `quoted_line()` strips the tool's `[page N]` prefix before anything else. Unstripped
   it was actively harmful: `"[page 1]"` is 8 characters, so it cleared the length bar,
   and its tokens `page` and `1` are in every document, so it cleared the 60% overlap
   bar. A bare page number was passing as a grounded quote — 33 of 498 answered fields
   in the previous run. Once the value check existed it would also have parsed that page
   number as a measurement. This one is unconditional; it is a bug fix, not a policy.
2. `value_appears_in_quote()` — parses every number out of the quote via the new
   `units.parse_all_quantities`, rejects when none matches the reported value. Allows
   unit conversion, the dual-supply ±18 V → 36 V derivation, unsigned range minima, and
   channel counts written as words ("Quad Buffer" supports 4).
3. A missing quote became a rejection rather than a silent pass.

**Both 2 and 3 ship off** (`DSA_REQUIRE_VALUE_IN_QUOTE`, `DSA_REQUIRE_EVIDENCE`), because
measurement said so. Two new runs over the same 100 held-out parts:

| configuration | accuracy | precision | repaired | destroyed |
|---|---|---|---|---|
| no verifier (v3) | 32.2% | — | — | — |
| code only (v6) | 33.0% | 85.4% | 11 | 0 |
| code + Phi-4 (v6) | 32.1% | 55.3% | 7 | 1 |
| strict code only (v7) | 31.9% | 52.5% | 7 | 0 |
| strict code + Phi-4 (v7) | 34.0% | 52.9% | 5 | 1 |

Per-check precision, from the single-stage strict run:

| check | rejections | precision |
|---|---|---|
| value unreadable in the asked-for unit | 30 | 100% |
| value outside the plausible range | 4 | 100% |
| value not stated on the quoted line | 108 | 45.4% |
| quote does not appear in the document | 12 | 41.7% |
| no quote given at all | 25 | 24.0% |

The strict checks do what they were built to do — wrong values reaching the output fell
from 188 to 67 — but they check the *quote*, and at 1.5B the quote and the value fail
independently. Requiring evidence throws out three correct answers for each wrong one it
catches.

**The finding that outranks all of it:** none of those five accuracy numbers are
distinguishable. Same 882 fields, so the test is paired; McNemar gives p = 0.20 to 0.87
across every pair, including best-vs-none at p = 0.57 (16 fields won, 12 lost). The
earlier README called code-only "the only one worth running" on a 0.8-point gap. That
claim is now removed and replaced with the test that refutes it, and
`scripts/paired_compare.py` makes it reproducible. The per-check precisions, measured on
hundreds of rejections, are what actually discriminate.

A test pins the limit rather than hiding it: a transition frequency quoted under
gain-bandwidth product passes both evidence checks, because the number really is on the
line. The structural fix — have the agent return a page-and-row *location* and have code
pull the text — is written up in the README as the next step. Not built.

134 tests, ruff clean. Runs: `v7-multiagent-codeverify-qwen1.5b-test.json`,
`v7-multiagent-verified-qwen1.5b-test.json`.

## Iteration 23 — the RAG baseline, and why the agents lose

Run because the obvious question had never been measured: *why not just RAG?* Four runs
(BM25 and dense, at 1.5B and 32B) plus a coverage decomposition.

**RAG ties the whole-document call at both scales and beats every agent design at both.**

| method | 1.5B | 32B | tokens in |
|---|---|---|---|
| BM25 RAG | 60.8% | 78.1% | 7.5k |
| single call | 59.5% | 76.2% | 15.9k |
| dense RAG | 54.7% | 71.9% | 7.7k |
| multi-agent | 32.2% | 75.1% | 25-50k |
| single agent | 40.5% | 70.5% | 15k |

Paired McNemar: RAG vs single call p = 0.33 (1.5B) and p = 0.79 (32B) — a tie on half the
tokens. RAG vs each agent design: four tests, all p < 0.001, all the same direction.

**This retracts a claim.** The README said multi-agent "pays above a capability
threshold", on the strength of beating the single *agent* by 4.6 points at 32B. True, and
the wrong comparison: both agent designs lose to a thirty-line BM25 scorer. The baseline
should have existed before the claim did — the plan's build order put baselines at step 4
for exactly this reason, and the two baselines actually built were both agent-adjacent.

**Why do the agents lose?** `scripts/coverage_report.py` splits accuracy into finding and
reading: coverage is the share of fields whose answer was anywhere in the context the
method saw, reading accuracy is how often it was then right.

| 32B | coverage | reading |
|---|---|---|
| BM25 RAG | 98.2% | 78.4% |
| whole document | 98.8% | 75.3% |
| single agent | 93.0% | 74.1% |
| dense RAG | 96.6% | 69.0% |

The agent loses on both, and almost exactly equally: give it RAG's coverage and it gains
3.9 points, give it RAG's reading and it gains 4.0.

**The agent is not seeing less text.** Median context: 13.7k chars for the agent, 16.4k
for RAG; means 15.8k for both. So at equal context size, a scoring function with no model
in it assembles a better context than an agent choosing for itself with nine LLM calls —
better in *where* it looks and in *what it gets back*, because tool output arrives as
truncated snippets stripped of the row and column headers that identify a value.

And retrieval is not winning by finding things. The whole document contains the answer
98.8% of the time; BM25 matches that at 98.2% on an eighth of the characters. RAG's
entire advantage is 3 points of reading accuracy — it wins by *removing* distractors, not
by locating anything. The task's bottleneck is comprehension, which is the one thing
better navigation cannot buy. That is also why GRPO bought 4x cost and no accuracy.

One field states the problem on its own: `input_offset_voltage` has ~96% coverage and
~46-55% reading accuracy in every method. The answer is on screen and the model picks the
wrong number half the time, because typ and max sit in adjacent columns.

Dense retrieval confirms the embeddings-versus-symbols argument with a per-field shape:
coverage 90.8% for `gain_bandwidth_product` and 94.0% for `slew_rate` against BM25's 98%,
while `operating_temp_*` and `num_channels` are identical. The embedding model fails to
retrieve the GBW page at all for roughly one datasheet in ten. Caveat recorded in the
tests: the presence metric counts a matching number *anywhere*, so it flatters navigation
and some of dense's reading deficit is really retrieval quality.

149 tests, ruff clean. Runs: `v8-rag-{bm25,dense}-{qwen1.5b,large}`,
`v9-singleagent-large-ctx`.
