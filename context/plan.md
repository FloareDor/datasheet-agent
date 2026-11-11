# DatasheetAgent — Plan

**Pitch:** Datasheets are too long for a small model to read in one go. Parallel agents navigate
them with tools, a verifier checks every value against its source, and the small model is trained
with RL so it matches Claude at a fraction of the cost.

**Constraints (from user):**
- Run all Modal jobs with `modal run --detach` (long jobs must survive disconnects).
- Clean code: typed, small modules, no dead scaffolding, tests where they earn their place.
- Work autonomously in a loop until the project is finished.

---

## 1. Data
- Open datasheets (transistors + op-amps) with labels. **Confirm the label files exist in the
  archive first** before building anything on top.
- ~6 fields per part. ~300 datasheets train / ~100 test.
- *Optional:* 30 Micron memory datasheets, self-labeled, as a transfer test.

## 2. Tools
- `search`, `read_section`, `read_table`. Reuse Infineon's `datasheetindex` library.

## 3. Agents (LangGraph workflow)
- Field agents in parallel, one per group: electrical, thermal, package.
- Merge results in code.
- Verifier: code checks first, then a model from a different family. Wrong fields go back to their
  agent with a reason, one retry.

## 4. Evals
- Field accuracy vs. labels, cost per datasheet, latency.
- Compare: one big call (Claude) · single agent vs. multi-agent at equal token budget ·
  with vs. without verifier.

## 5. RL *(cut first if short on time)*
- GRPO on Qwen 1.5B on Modal. Reward = fields correct − small cost per tool call.
- Compare before vs. after RL, and against Claude.

## 6. Ship
- vLLM on Modal with speculative decoding.
- Load test with Locust.
- Live demo page: upload a datasheet → get the spec sheet back.
- README with results table and a "what didn't work" section.

**Build order:** 1 → 2 → 4 (baselines) → 3 → 5 → 6. About a week.

---

## Resume point (real numbers only)
> **DatasheetAgent** | *Python, LangGraph, GRPO, vLLM, Modal*
> – Built parallel tool-using agents with a source-grounded verifier to extract chip specs from
> datasheets; trained a 1.5B model with **agentic RL** (GRPO): accuracy **X% → Y%**, matching
> Claude at **Z× lower cost**.

## Links
**Data and tools**
- Datasheet dataset (Zenodo): https://zenodo.org/records/4558344
- Infineon datasheet tools: https://github.com/Infineon/datasheetindex
- Infineon agentic datasheet extraction paper: https://arxiv.org/pdf/2608.28439

**Building**
- LangGraph: https://github.com/langchain-ai/langgraph
- Modal docs: https://modal.com/docs
- TRL GRPO trainer: https://huggingface.co/docs/trl/grpo_trainer
- Multi-turn tool-use RL (Qwen 1.5B + GRPO): https://januverma.substack.com/p/multi-turn-tool-use-with-rl
- VerlTool (multi-turn agentic RL): https://arxiv.org/pdf/2509.01055v2
- vLLM docs: https://docs.vllm.ai
- Locust: https://locust.io

**Why this design**
- Anthropic on multi-agent systems: https://claude.com/resources/articles/building-multi-agent-systems-when-and-how-to-use-them
- Self-verification limits: https://arxiv.org/pdf/2402.08115v1.pdf

**Why it fits Micron**
- Agentic Solutions Engineer: https://www.presalescollective.com/job/engineer-agentic-solutions-65184
- AI Enablement: https://builtin.com/job/engineer-it-ai-enablement/8806251
