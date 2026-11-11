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
it against a held-out split.

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
  from a *different* family, with rejected fields returned to their own agent once.
- **Agentic RL** — GRPO on Qwen2.5-1.5B through TRL's multi-turn `environment_factory`, with a
  grounding gate that makes guessing worthless.
- **Honest cost accounting** — GPU-seconds at Modal's published rates divided by the throughput
  each run actually sustained, never an assumed token price.
- **Everything served on [Modal](https://modal.com)** — vLLM inference, GRPO training, a live demo
  page and a Locust load test.

## Results

100 held-out parts, 882 graded fields.

| model | method | accuracy | $/datasheet | tokens in | tool calls | failures |
|---|---|---|---|---|---|---|
| Qwen2.5-32B | single call | **76.2%** | $0.0052 | 15.9k | 0 | 9 |
| Qwen2.5-32B | **multi-agent** | **75.1%** | $0.0139 | 25.4k | 12.4 | **0** |
| Qwen2.5-32B | single agent | 70.5% | $0.0089 | 14.9k | 12.6 | 4 |
| Qwen2.5-1.5B | single call | 59.5% | **$0.0005** | 15.9k | 0 | 10 |
| Qwen2.5-1.5B | single agent | 40.5% | $0.0026 | 26.5k | 8.0 | 20 |
| Qwen2.5-1.5B | RL-trained agent | 33.7% | $0.0011 | 6.7k | 1.0 | 4 |
| Qwen2.5-1.5B | multi-agent | 32.2% | $0.0018 | 50.3k | 23.5 | 32 |

**The 1.5B reaches 78% of the 32B's accuracy at a tenth of the cost** — 59.5% vs 76.2% for
$0.0005 against $0.0052, 10.6× cheaper.

### Multi-agent only pays once the model is good enough

This finding needed both models to see, and it reverses on model size:

| | single call − multi-agent | multi-agent vs single agent |
|---|---|---|
| **Qwen2.5-1.5B** | **27.3 points** worse | multi-agent **8.3 points worse** |
| **Qwen2.5-32B** | **1.1 points** worse | multi-agent **4.6 points better** |

At 1.5B, splitting the work across three field agents is the worst option available. At 32B the
ordering **flips**: multi-agent beats the single agent, comes within a point of pasting in the whole
datasheet, and is the only configuration that extracted all 100 parts with **zero failures**.

That is exactly [Anthropic's criterion](https://claude.com/resources/articles/building-multi-agent-systems-when-and-how-to-use-them)
— subtasks must be separable *and the model capable enough to exploit the separation*. Measuring
only the 1.5B would have produced the confident wrong conclusion that the architecture does not
work. It does; the small model could not use it.

The token economics move the same way. At 32B the single agent reads **fewer** input tokens than
pasting the datasheet in (14.9k vs 15.9k) because it fetches only what it needs. At 1.5B the same
agent reads 26.5k — it searches badly and reads the wrong pages.

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

**The cheap checks run first**, and measurement says they are the only ones worth running:

| configuration | accuracy | precision | repaired | destroyed | latency |
|---|---|---|---|---|---|
| no verifier | 32.2% | — | — | — | 19.5 s |
| **code checks only** | **33.0%** | **85.4%** | 11 | **0** | 14.0 s |
| code + Phi-4 verifier | 32.1% | 55.3% | 7 | 1 | 19.6 s |

The verifier is a different model family on purpose — a model asked to check its own work tends to
agree with it ([arXiv:2402.08115](https://arxiv.org/pdf/2402.08115v1.pdf)) — and it still was not
enough. Phi-4 judges whether a quoted line supports a value, and the 1.5B's *evidence quotes* are
often poor even when its *value* is right, so it rejects good answers for bad citations at 55.3%
precision: barely better than a coin flip.

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
verifier.

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

Three claims, each backed by a run in [`runs/`](runs/):

1. **A 1.5B model reaches 78% of a 32B's accuracy at a tenth of the cost** — 59.5% vs 76.2% for
   $0.0005 vs $0.0052 per datasheet.
2. **Multi-agent extraction pays only above a capability threshold.** At 32B it beats the single
   agent by 4.6 points and finishes all 100 parts with zero failures; at 1.5B it is the worst option
   by 27 points. Measuring one model alone would have given the opposite answer.
3. **Agentic RL cut cost, not error.** GRPO took the 1.5B agent to 4× fewer tokens and 4× lower
   latency and taught it to ground every answer in the document — but accuracy fell 6 points, because
   the limit is reading comprehension rather than navigation.

And three notes on method that mattered more than any single result:

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

## Resources

- [Infineon `datasheetindex`](https://github.com/Infineon/datasheetindex) — the PDF index these tools wrap
- [LangGraph](https://langchain-ai.github.io/langgraph/) · [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer) · [vLLM](https://docs.vllm.ai) · [Modal](https://modal.com/docs)
- [Building multi-agent systems](https://claude.com/resources/articles/building-multi-agent-systems-when-and-how-to-use-them) — when fan-out pays
- [arXiv:2402.08115](https://arxiv.org/pdf/2402.08115v1.pdf) — the limits of self-verification
- Dataset: [Zenodo 4558344](https://zenodo.org/records/4558344), CC BY-NC-SA 4.0, not redistributed here
