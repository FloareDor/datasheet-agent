<div align="center">

<img src="./docs/logo.svg" alt="" height="80" />

# DatasheetAgent

*Tool-using agents that read chip datasheets — with every value checked against its source*

[![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-parallel_agents-1c3c3c?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![TRL](https://img.shields.io/badge/TRL-GRPO-ffd21e?style=flat-square)](https://huggingface.co/docs/trl/grpo_trainer)
[![vLLM](https://img.shields.io/badge/vLLM-serving-30a2ff?style=flat-square)](https://docs.vllm.ai)
[![Modal](https://img.shields.io/badge/Modal-GPU-7fee64?style=flat-square)](https://modal.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Results](#results) • [How it works](#how-it-works) • [Get started](#get-started) • [Negative control](#a-negative-control-what-happens-when-the-answer-does-not-exist) • [What didn't work](#what-didnt-work)

</div>

A datasheet is 15–60 pages of tables. The values an engineer wants — slew rate, gain-bandwidth
product, operating temperature range, package — are scattered across specification tables,
absolute-maximum tables and ordering information, often with a different column per part in a
family. That is too much to paste into a small model's context, so the model is given tools and
has to find the values itself.

This repo extracts nine parametric fields from op-amp datasheets with parallel tool-using agents
and a source-grounded verifier, then trains a 1.5B model to do it with GRPO — and measures all of
it against a held-out split and against a plain retrieval baseline.

**The baseline wins.** BM25 retrieval matches whole-document prompting on half the tokens and beats
every agent configuration at both model sizes. The interesting part of this repo is the measurement
that shows *why*, and the claim it forced me to retract.

> [!NOTE]
> Every number below is measured on a held-out 100-part test split and reproducible from
> [`runs/`](runs/). Nothing is estimated. `tests/test_readme_claims.py` fails this README if a
> quoted accuracy stops matching its run.

## Features

- **Three tools over a real datasheet index** — `search`, `read_section`, `read_table`, built on
  [Infineon's `datasheetindex`](https://github.com/Infineon/datasheetindex), with field-aware query
  fallback so a search for `slew_rate` still finds "Slew Rate".
- **Parallel field agents** — electrical, thermal and package agents fan out in
  [LangGraph](https://langchain-ai.github.io/langgraph/) and merge in code, because the groups are
  disjoint and there is nothing to arbitrate.
- **A verifier that checks the source, not the model** — unit and range checks first, then a model
  from a *different* family, with rejected fields returned to their own agent once. The quote it
  checks is written by the agent, not copied from the tool, which is both why the check is worth
  running and where it runs out.
- **Agentic RL** — GRPO on Qwen2.5-1.5B through TRL's multi-turn `environment_factory`, with a
  grounding gate that makes guessing worthless.
- **A retrieval baseline that is allowed to win** — BM25 and dense RAG over the same chunks, same
  prompt, same grader, plus a coverage decomposition that separates failing to *find* a value from
  failing to *read* it.
- **Honest cost accounting** — GPU-seconds at Modal's published rates divided by the throughput
  each run actually sustained, never an assumed token price.
- **Everything served on [Modal](https://modal.com)** — vLLM inference, GRPO training, a live demo
  page and a Locust load test.

## Results

100 held-out parts, 882 graded fields.

| model | method | accuracy | $/datasheet | tokens in | tool calls | failures |
|---|---|---|---|---|---|---|
| Qwen2.5-32B | BM25 RAG | **78.1%** | $0.0035 | **7.5k** | 0 | 6 |
| Qwen2.5-32B | single call | 76.2% | $0.0052 | 15.9k | 0 | 9 |
| Qwen2.5-32B | multi-agent | 75.1% | $0.0139 | 25.4k | 12.4 | **0** |
| Qwen2.5-32B | dense RAG | 71.9% | $0.0052\* | 7.7k | 0 | 3 |
| Qwen2.5-32B | single agent | 70.5% | $0.0089 | 14.9k | 12.6 | 4 |
| Qwen2.5-1.5B | BM25 RAG | **60.8%** | **$0.0004** | **7.5k** | 0 | 3 |
| Qwen2.5-1.5B | single call | 59.5% | $0.0005 | 15.9k | 0 | 10 |
| Qwen2.5-1.5B | dense RAG | 54.7% | $0.0009\* | 7.7k | 0 | 1 |
| Qwen2.5-1.5B | single agent | 40.5% | $0.0026 | 26.5k | 8.0 | 20 |
| Qwen2.5-1.5B | RL-trained agent | 33.7% | $0.0011 | 6.7k | 1.0 | 4 |
| Qwen2.5-1.5B | multi-agent | 32.2% | $0.0018 | 50.3k | 23.5 | 32 |

\* The dense figure charges GPU time for CPU embedding work, because cost is wall-clock based.
Treat it as an upper bound.

**The 1.5B reaches 78% of the 32B's accuracy at a tenth of the cost** — 59.5% vs 76.2% for
$0.0005 against $0.0052, 10.6× cheaper.

**And the tool-using agents lose to a thirty-line BM25 retriever at both model sizes.** That is the
headline result of this repo, it is not the one it was built to produce, and the section below is
the measurement rather than the rationalisation.

### Why not just RAG?

The first question anyone asks about a tool-using agent, and for a long time this repo answered it
with reasoning instead of a baseline. Here is the baseline: fixed-size chunking, one query per
field built from the same `search_terms` the agent's tools get, top-3 chunks per field merged and
capped at 20, then one call with the same prompt, the same output contract and the same grader.
Only the retrieved text differs.

Comparisons are paired — every method grades the same fields, so the test is McNemar's rather than
an eyeball of two percentages:

| comparison | result | p |
|---|---|---|
| BM25 RAG vs single call, 1.5B | tie (38 won, 48 lost) | 0.33 |
| BM25 RAG vs single call, 32B | tie (29 won, 26 lost) | 0.79 |
| BM25 RAG vs single agent, 1.5B | **RAG wins** 131–60 | <0.001 |
| BM25 RAG vs multi-agent, 1.5B | **RAG wins** 228–44 | <0.001 |
| BM25 RAG vs single agent, 32B | **RAG wins** 120–34 | <0.001 |
| BM25 RAG vs multi-agent, 32B | **RAG wins** 113–44 | <0.001 |

**Retrieval ties whole-document prompting at both scales on half the tokens, and beats every agent
configuration at both scales.** Four tests against the agents, all p < 0.001, all the same
direction. The agents never win.

> [!CAUTION]
> **Retracted: "multi-agent pays above a capability threshold."** An earlier version of this README
> made that claim because at 32B the multi-agent beat the single *agent* by 4.6 points. The
> comparison is real and it was the wrong one — both agent designs lose decisively to a scoring
> function with no model in it. Ranking configurations that all lose to the baseline is not a
> finding. The baseline should have existed before the claim did; the build order in
> [`context/plan.md`](context/plan.md) put baselines at step 4 for exactly this reason, and the two
> baselines actually built were both agent-adjacent.

#### Where the agents lose it: finding, or reading?

Accuracy alone cannot say whether a wrong answer means the text never reached the model or reached
it and was misread. [`scripts/coverage_report.py`](scripts/coverage_report.py) splits it:

> **accuracy = coverage × reading accuracy**

where coverage is the share of fields whose answer was anywhere in the context that method put in
front of the model, over the seven numeric fields at 32B:

| method | coverage | reading accuracy |
|---|---|---|
| BM25 RAG | **98.2%** | **78.4%** |
| whole document | 98.8% | 75.3% |
| single agent | **93.0%** | 74.1% |
| dense RAG | 96.6% | 69.0% |

The agent is the only method that sees a subset it chose itself, so it is the only one that *can*
fail to navigate — and it does, missing the answer three times as often as retrieval. But it also
reads worse on what it does find. Splitting the 8.1-point gap counterfactually: give the agent
RAG's coverage and it gains **3.9 points**; give it RAG's reading accuracy and it gains **4.0**. An
even split.

**It is not seeing less text.** Median context is 13.7k characters for the agent against 16.4k for
RAG, means 15.8k for both. At equal context size, a thirty-line scoring function assembles a better
context than an agent choosing for itself across nine LLM calls — better in where it looks, and
better in what comes back, because tool output arrives as truncated snippets stripped of the row
and column headers that identify a value. The agent's own evidence quotes show it: `"...range
includes ground - Differential input voltage..."` offered as the source for a supply-voltage
answer.

**And retrieval is not winning by finding things either.** The whole document contains the answer
98.8% of the time; BM25 matches that at 98.2% on an eighth of the characters. Retrieval's entire
advantage is three points of reading accuracy. **It wins by removing distractors, not by locating
anything** — which means better navigation was never the lever, and an architecture whose value
proposition is better navigation is optimising a part that was not broken.

#### Embeddings versus symbols

Dense retrieval (MiniLM) loses to BM25 by 6.1 points at 1.5B and 6.2 at 32B, p < 0.001. The
per-field shape is the reason, and it is the one a datasheet predicts:

| field | BM25 | dense | diff |
|---|---|---|---|
| gain_bandwidth_product | 55.1% | 29.6% | **−25.5** |
| slew_rate | 60.0% | 43.0% | **−17.0** |
| num_channels | 66.0% | 56.0% | −10.0 |
| input_offset_voltage | 19.4% | 12.2% | −7.2 |
| supply_voltage_max | 77.8% | 73.3% | −4.5 |
| operating_temp_max | 79.8% | 75.8% | −4.0 |
| package_case | 31.0% | 32.0% | +1.0 |
| operating_temp_min | 80.8% | 84.9% | +4.0 |
| mounting_style | 78.6% | 86.7% | **+8.2** |

Embeddings collapse on exactly the fields a datasheet writes as a terse symbol beside a number —
`SR`, `GBW`, `V_OS` — and **win** on the fields written in ordinary words, `mounting_style`
("surface mount") and `operating_temp_min` ("Operating Temperature Range"). Retrieval coverage says
the same thing before any model reads anything: dense reaches the gain-bandwidth page for 90.8% of
datasheets against BM25's 98.0%. A query embedding of "slew rate" lands near prose about slewing,
not near a table cell reading `SR`.

> [!NOTE]
> The coverage metric counts a number equal to the label appearing *anywhere* in the context, with
> no requirement that it sit by the right parameter. It is an upper bound on navigation, pinned as
> such by a test. The bias is deliberate and one-directional: it can only make navigation look
> better than it is, and the conclusion drawn from it is that navigation is not the problem.

### Where the small model actually loses

| field | 1.5B single call | 32B single call | gap |
|---|---|---|---|
| package_case | 31.0% | 75.0% | **+44.0** |
| input_offset_voltage | 11.2% | 53.1% | **+41.9** |
| gain_bandwidth_product | 54.1% | 82.7% | **+28.6** |
| mounting_style | 70.4% | 88.8% | +18.4 |
| num_channels | 72.0% | 87.0% | +15.0 |
| slew_rate | 64.0% | 71.0% | +7.0 |
| supply_voltage_max | 73.3% | 75.6% | +2.3 |
| operating_temp_max | 76.8% | 74.8% | −2.0 |
| operating_temp_min | 83.8% | 77.8% | **−6.0** |

The gap concentrates in the fields that require reading a table *correctly* rather than merely
finding it: `package_case` must match the exact orderable suffix in an ordering table, and
`input_offset_voltage` is a maximum at 25 °C that varies by grade and by column.

The temperature fields invert — the **1.5B is better** at both. Those values usually appear once,
in plain text near the front ("−40 °C to +105 °C"), so finding them is the whole task and there is
nothing for extra capability to buy. The 32B occasionally prefers an absolute-maximum or storage
range instead, which is a more sophisticated mistake than the 1.5B is capable of making.

**Navigation is not the bottleneck — comprehension is.**

### RL: four runs, one honest conclusion

GRPO on Qwen2.5-1.5B via TRL's multi-turn `environment_factory`, 878 rollouts (part × field group),
LoRA r=32, vLLM colocated on 2×H100.

| run | reward | accuracy | tool calls | tokens in | latency | failures |
|---|---|---|---|---|---|---|
| base agent | — | **40.5%** | 8.02 | 26,464 | 32.7 s | 20 |
| v2 | accuracy − tool cost | 37.5% | **0.00** | 1,627 | **2.3 s** | 0 |
| v3 | **grounded** − tool cost | 33.7% | 1.00 | 6,712 | 7.2 s | 4 |
| v4 | **grounded**, no tool cost | 34.1% | 1.06 | 6,258 | 8.4 s | 3 |

> [!WARNING]
> **v2 reward-hacked.** Rewarded for accuracy minus a per-call cost, the policy converged to
> *exactly zero* tool calls and answered from the part number alone. Qwen2.5-1.5B can guess ~30% of
> these fields from priors — SOIC-8 is the usual package, −40/+85 the usual temperature range — so
> guessing at zero cost beats tool use at similar accuracy *plus* the cost. GRPO found the true
> optimum of the reward it was given.

The admission: `tests/test_reward.py` asserted accuracy outweighs tool cost **at a fixed accuracy**.
It never asked what happens when the policy reaches similar accuracy with no tools at all. The test
was true and the reward was still exploitable.

**Grounding fixed it, with no special case.** A field now counts only if it is correct *and* its
evidence appears in this datasheet, from a rollout that made at least one tool call. A policy that
calls no tools has read nothing, so it can quote nothing, so nothing is grounded. Tool calls went
0.00 → 1.00.

**The policy then stopped at exactly the minimum the gate allows** — one search, then answer from
the snippet. The obvious suspicion is that the per-call cost suppressed further tool use, so **v4
removed the cost term entirely**. Tool calls stayed at 1.06 and accuracy at 34.1%: unchanged. The
cost coefficient was not the cause.

**So RL bought cost, not accuracy:** 4× fewer input tokens, 4× faster, 5× fewer failures than the
base agent, at 6 points lower accuracy. Three independent measurements agree on why — a single call
scores 59.5%, an 8-call agent 40.5%, a 1-call trained agent 34.1%. More navigation does not buy
accuracy at this size, and RL over a navigation policy cannot fix comprehension.

## How it works

```
datasheet.pdf
     │
     ▼
 datasheetindex ──> page-matched text, section map, markdown tables
     │
     ▼
  three tools: search · read_section · read_table
     │
     ├──> electrical agent  (supply V, GBP, slew rate, Vos)
     ├──> thermal agent     (operating temp min/max)       [parallel, LangGraph]
     └──> package agent     (package, channels, mounting)
     │
     ▼
  merge (in code — the groups are disjoint, so there is nothing to decide)
     │
     ▼
  verifier: code checks, then a model from a different family
     │          · is the value parseable in the right unit?
     │          · is it inside a plausible range?
     │          · does the quoted evidence actually appear in this document?
     ▼
  rejected fields go back to their own agent with the reason, once
```

**The cheap checks run first.** Five configurations, same 100 held-out parts:

| configuration | accuracy | precision | repaired | destroyed | latency |
|---|---|---|---|---|---|
| no verifier | 32.2% | — | — | — | 19.5 s |
| code checks only | 33.0% | 85.4% | 11 | **0** | 14.0 s |
| code + Phi-4 verifier | 32.1% | 55.3% | 7 | 1 | 19.6 s |
| strict evidence checks only | 31.9% | 52.5% | 7 | **0** | 29.9 s |
| strict evidence + Phi-4 | 34.0% | 52.9% | 5 | 1 | 27.7 s |

> [!WARNING]
> **None of these accuracy differences are real.** Every configuration grades the same
> 882 fields, so the paired test is McNemar's, and no pair reaches significance:
> best-vs-none is p = 0.57 (16 fields won, 12 lost), and the widest gap in the table is
> p = 0.20. Reading the top 34.0% as "the verifier adds 1.8 points" would be reading
> noise. An unpaired interval is no kinder — ±1.6 points at 1σ, which swallows the whole
> column. **The accuracy column cannot answer this question at n = 100 parts.** It is
> printed here because leaving it out would be worse, not because it decides anything.

What *does* discriminate is the verifier graded as a classifier, where the per-check
precisions differ by 4× and are measured on hundreds of rejections rather than inferred
from a difference of a dozen fields:

| code check | rejections | precision |
|---|---|---|
| value unreadable in the asked-for unit | 30 | **100%** |
| value outside the plausible range | 4 | **100%** |
| value not stated on the quoted line | 108 | 45.4% |
| quote does not appear in the document | 12 | 41.7% |
| no quote given at all | 25 | **24.0%** |

The two strict checks are the bottom rows, and they are **off by default**
(`DSA_REQUIRE_VALUE_IN_QUOTE`, `DSA_REQUIRE_EVIDENCE`). They work as designed — they cut
wrong values reaching the output from 188 to 67 — but they are checks on the *quote*, and
at 1.5B the quote and the value fail independently. Rejecting every unevidenced value
throws out three correct answers for every wrong one it catches, because this model
frequently reads the right number off the right line and then cites it badly. **Demanding
that a small model justify itself mostly punishes it for being bad at justifying itself.**

The checks that hold at 100% are the ones that look at the value alone and need no quote.

The verifier is a different model family on purpose — a model asked to check its own work
tends to agree with it ([arXiv:2402.08115](https://arxiv.org/pdf/2402.08115v1.pdf)) — and
it still was not enough, at 55.3% precision.

> [!IMPORTANT]
> **The quote is text the model typed, not text a tool returned.** The agent reads a table
> through `read_table`, then *writes out* the line it says it used, so it can mistype it,
> splice two rows, or invent it wholesale. That is what makes the check worth having — and
> what caps it. Grounding is token overlap against the document, so it asks only whether
> the line exists. Checking the number against its own quote asks only whether the value is
> on that line. **Neither asks whether it is the right line**, which is exactly the error
> the negative control found: a transition frequency quoted under gain-bandwidth product
> passes both, because the quote is real and the number really is on it.
>
> The fix is structural, not another check: have the agent return a *location* in the tool
> output — page and row — and have code pull the text from that spot, so the quote cannot
> be authored by the model at all. That leaves exactly one open question (did it point at
> the right row) instead of four. Not built.

> [!NOTE]
> An earlier version of this section called the code checks "the only ones worth running",
> on a 0.8-point difference. That was the accuracy column being read as though it could
> resolve a point. It cannot, and the paired test above is what says so.

> [!IMPORTANT]
> **A rejection must never be allowed to erase a value.** Before that rule the verifier cost
> **10 points** (32.2% → 22.3%). Graded as a classifier, its retries destroyed 98 correct values —
> 94 by nulling them — while repairing 9. The retry prompt tells the model that answering null is
> "a correct answer, not a failure", which is sound against hallucination and an open exit for a
> small model told it was wrong. With the guard in place the same verifier costs 0.1 points.

## The data

[Zenodo 4558344](https://zenodo.org/records/4558344) — datasheets from the ACM TECS paper on
hardware component knowledge bases.

A caveat the archive does not advertise: **only op-amps have labels.** The 673 MB transistor archive
is 6,967 PDFs with no ground truth at all. The labels live in the small `mouser_opamps` file — 54
catalogue CSVs covering 1,335 parts.

Joining the two sides is the actual work. Mouser names a part by its orderable code (`OPA2991IDR`);
the archive names PDFs by document code (`ad8656.pdf`, `1637fd.pdf`). Three joins run in order of
confidence — exact filename, stem prefix, and title token, which reaches family datasheets where one
document covers TL061/TL062/TL064.

**416 parts survive validation: 300 train / 100 test, split by datasheet rather than by part** so a
family datasheet cannot appear on both sides.

`scripts/validate_dataset.py` independently checks that each part's number actually appears in the
datasheet it was joined to. It caught `OPA2342EA` joined to `opa234.pdf` — two different op-amps
whose type numbers are prefixes of one another — along with `LM6144BIM`→`lm614` and
`MAX40088AUT`→`max400`. Roughly 4% of labels were silently wrong before that check existed.

## A negative control: what happens when the answer does not exist

A bipolar transistor datasheet states no slew rate, no gain-bandwidth product and no input offset
voltage — those parameters do not exist for the part. So on transistor datasheets the correct answer
is **null every time**, and any value returned is an invention. This needs no labels, which is what
finally makes the 6,967-PDF transistor archive useful.

30 transistor datasheets, single call:

| field | Qwen2.5-1.5B | Qwen2.5-32B |
|---|---|---|
| input_offset_voltage | 33.3% invented | **0.0%** |
| slew_rate | 3.3% invented | **0.0%** |
| gain_bandwidth_product | 26.7% invented | **33.3%** |
| **overall** | **21.1%** | **11.1%** |

**The errors are not hallucinations.** They are real numbers from the document, read off the wrong
parameter. The 1.5B reports V<sub>BE</sub> or V<sub>CE(sat)</sub> (−0.5, −0.8, −1.4 mV) as input
offset voltage, and transition frequency f<sub>T</sub> as gain-bandwidth product.

**Capability removes the crude substitutions and entrenches the sophisticated one.** The 32B never
invents an offset voltage or a slew rate — it knows transistors have neither — but it reports
f<sub>T</sub> as GBP *more* often than the small model does, confidently and with units ("200 MHz",
"50 MHz"). For a transistor, f<sub>T</sub> genuinely is the nearest analogue to gain-bandwidth
product; it is a defensible engineering mapping, and it is not what was asked for.

> [!CAUTION]
> This is the sharpest limit on the whole "source-grounded" idea: **grounding verifies that a quote
> exists, not that it is the right quote.** Every one of these invented values would pass the
> evidence check, because the line really is in the datasheet — it just supports a different
> parameter.

## Get started

### Prerequisites

- Python **3.13** (`datasheetindex` requires it) and [uv](https://docs.astral.sh/uv/)
- A [Modal](https://modal.com) account for GPU inference and training
- Modal secrets named `hf-token` and `wandb-api-key`

### Install

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev,serve]"
```

### Build the dataset

```bash
python scripts/build_dataset.py --train 300 --test 100
python scripts/validate_dataset.py --drop
```

### Serve the models and evaluate

```bash
modal deploy -m datasheet_agent.modal_vllm     # small / large / verifier / trained servers
python scripts/run_eval.py --method multi_agent --split test --backend small
python scripts/summarise_runs.py               # regenerates runs/summary.md
```

`--method` is one of `single_call`, `single_agent`, `multi_agent`; `--backend` is `small`,
`small-spec`, `trained` or `large`. Add `--no-verifier` or `--verifier-backend none` to ablate the
verifier. The two strict evidence checks are off by default and switch on through the
environment, so the ablation reproduces without editing source:

```bash
DSA_REQUIRE_VALUE_IN_QUOTE=1 DSA_REQUIRE_EVIDENCE=1 \n  python scripts/run_eval.py --method multi_agent --split test --backend small \n  --verifier-backend verifier --tag v7-multiagent-verified-qwen1.5b

python scripts/verifier_report.py runs/v7-multiagent-verified-qwen1.5b-test.json
python scripts/paired_compare.py v3-multiagent-qwen1.5b v7-multiagent-verified-qwen1.5b
```

### Train with GRPO

```bash
python scripts/upload_training_data.py
python scripts/launch_training.py --steps 400   # modal deploy + spawn, survives the local shell
```

> [!TIP]
> `scripts/regrade.py` re-scores a saved run from its stored predictions, so a grading fix never
> costs GPU time again — and never confounds the fix with fresh sampling noise.

### Demo and load test

```bash
modal deploy -m datasheet_agent.modal_app      # live demo page, web/index.html
locust -f scripts/locustfile.py --host <url>   # 964 requests, 0 failures, 10.9 req/s on one L40S
```

## Project structure

| path | what |
|---|---|
| `src/datasheet_agent/units.py` | engineering quantities → canonical SI |
| `src/datasheet_agent/schema.py` | the nine fields, their groups and their datasheet vocabulary |
| `src/datasheet_agent/tools.py` | `search` / `read_section` / `read_table` over `datasheetindex` |
| `src/datasheet_agent/grading.py` | field grading, package-name canonicalisation |
| `src/datasheet_agent/extract.py` | single-call baseline and the tool-using agent loop |
| `src/datasheet_agent/pipeline.py` | LangGraph fan-out, merge, verify, retry |
| `src/datasheet_agent/verify.py` | code checks then the cross-family model check |
| `src/datasheet_agent/reward.py` | the GRPO reward, with the grounding gate |
| `src/datasheet_agent/rl_env.py` | the RL environment: tools as methods |
| `src/datasheet_agent/pricing.py` | GPU-seconds → cost per datasheet, with its price source |
| `src/datasheet_agent/modal_*.py` | vLLM servers, GRPO training, demo service |
| `runs/` | every measured run; `runs/archive/` holds superseded ones and says why |

## What didn't work

**Fetching datasheets from the catalogue's own URLs.** The obvious way past the archive's coverage,
and a dead end: 0 of 12 sampled URLs returned a PDF. TI links are interstitial redirect pages;
Mouser's own links serve HTML to non-browser clients. Getting them would mean defeating bot
protection, so the labelled set is what the archive covers.

**A literal search tool.** The first agent runs scored **exactly 0%** on every electrical field. The
cause was not the model: it searched for `slew_rate`, a string that appears in no datasheet ever
printed, and the tool answered "No matches". Relaxing the query the way a search box would took
single-agent 21.7% → 41.8% and multi-agent 5.7% → 29.6%.

**Blind query relaxation, which then broke it differently.** `operating_temp_min` shortens to
"operating temp", which substring-matches "Full **Temp**erature" inside a distortion spec — so the
agent reported an operating temperature of **0.7%**. Relaxation without domain knowledge is not just
weaker, it is actively misleading. The tool now tries the field's real vocabulary first.

**Speculative decoding, on the workload that actually matters.** N-gram speculation is 58% faster at
decoding when the output quotes the prompt and the server is idle — 171.8 → 271.7 tok/s, one request
at a time. On the real agent workload it is **5% slower**: the output is structured JSON of short
numeric values, so few tokens are copied from the prompt, and at four concurrent parts the GPU is
already batch-saturated, so drafting competes with real decoding instead of filling idle capacity.
Kept available, not enabled by default — numbers in [`runs/specdecode.md`](runs/specdecode.md).

**Scoring formatting instead of extraction.** An early run reported 41.5% with `supply_voltage_max`
at exactly 0/90. A model answering `36` to a question posed in volts was marked wrong against a
`36 V` label, because a bare number parsed as nothing. The prompt *states* the answer unit, so a bare
number has to be read in it. Fixing it moved the headline to 59.5%.

**Equal step budgets that nobody can meet.** Capping multi-agent at 4 steps to match single-agent
left 56 of 100 parts hitting the limit *without ever being asked for an answer*. The agent now gets
one final call with no tools when its budget runs out.

## What this project shows

Four claims, each backed by a run in [`runs/`](runs/):

1. **Plain BM25 retrieval beats every tool-using agent here, at both model scales.** Six paired
   tests: RAG ties whole-document prompting (p = 0.33 and 0.79) on half the tokens, and beats both
   agent designs at 1.5B and at 32B (all p < 0.001). The agents never win.
2. **The bottleneck is comprehension, not navigation — and that is measured, not argued.** The
   whole document contains the answer 98.8% of the time and BM25 reaches 98.2% on an eighth of the
   characters, so retrieval's whole advantage is three points of *reading* accuracy. It wins by
   removing distractors. `input_offset_voltage` is the thesis in one field: ~96% coverage, ~46–55%
   reading accuracy in every method, because typ and max sit in adjacent columns.
3. **A 1.5B model reaches 78% of a 32B's accuracy at a tenth of the cost** — 59.5% vs 76.2% for
   $0.0005 vs $0.0052 per datasheet.
4. **Agentic RL cut cost, not error.** GRPO took the 1.5B agent to 4× fewer tokens and 4× lower
   latency and taught it to ground every answer in the document — but accuracy fell 6 points. Three
   independent measurements agree on why, and the coverage split above says what they were all
   pointing at: RL over a navigation policy cannot fix comprehension.

And four notes on method that mattered more than any single result:

- **A field at exactly 0% is almost never the model's fault.** That pattern found three separate bugs
  here: a unit parser scoring formatting instead of extraction, a literal search tool, and a verifier
  rejecting correct answers.
- **Test what happens when the answer does not exist.** Every accuracy number above is measured where
  an answer exists. The negative control is the only measurement that shows the system inventing, and
  it exposed a failure — right number, wrong parameter — that no accuracy metric and no grounding
  check can catch.
- **Grade the component, not just the pipeline.** The verifier looked merely unhelpful (−10 points)
  until it was graded as a classifier, which showed 55.3% precision *and* retries destroying 98
  correct values against 9 repaired. Those need opposite fixes, and the headline number could not
  distinguish them.
- **Build the baseline that could embarrass the architecture, and build it first.** Every agent
  number in this repo was measured for weeks against other agents. The first time a thirty-line
  BM25 scorer was run on the same split, it beat all of them, and a claim had to be withdrawn.
  A baseline you skip is not a baseline you passed.
- **Check whether the number you are comparing can hold the comparison.** Five verifier
  configurations span 2.1 accuracy points on 100 parts, and a paired McNemar test says not one pair
  is distinguishable (p = 0.20 to 0.87). An earlier draft of this README picked a winner from that
  column anyway. The per-check precisions, measured on hundreds of rejections instead of a
  difference of a dozen fields, differ by 4× and are the numbers that actually decide something —
  [`scripts/paired_compare.py`](scripts/paired_compare.py) is what settled it.

## Resources

- [Infineon `datasheetindex`](https://github.com/Infineon/datasheetindex) — the PDF index these tools wrap
- [LangGraph](https://langchain-ai.github.io/langgraph/) · [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer) · [vLLM](https://docs.vllm.ai) · [Modal](https://modal.com/docs)
- [Building multi-agent systems](https://claude.com/resources/articles/building-multi-agent-systems-when-and-how-to-use-them) — when fan-out pays
- [arXiv:2402.08115](https://arxiv.org/pdf/2402.08115v1.pdf) — the limits of self-verification
- Dataset: [Zenodo 4558344](https://zenodo.org/records/4558344), CC BY-NC-SA 4.0, not redistributed here
