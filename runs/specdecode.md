# Speculative decoding (n-gram / prompt-lookup)

Same weights, same server config, only `--speculative-config` differs. Measured two
ways, which disagree — and the disagreement is the finding.

## Controlled: one request at a time, output quoting the prompt

`scripts/bench_decoding.py --repeats 6` — a real datasheet page in the prompt, asking
the model to quote table rows back verbatim.

| server | decode throughput |
|---|---|
| `small` (no speculation) | 171.8 tok/s |
| `small-spec` (n-gram) | **271.7 tok/s** |
| | **+58.2%** |

This is the case prompt-lookup decoding is built for: the draft tokens are already in
the prompt, so acceptance is high.

## End to end: the actual agent workload, 4 concurrent parts

`run_eval.py --method single_call --limit 40`, both servers.

| server | accuracy | latency / part |
|---|---|---|
| `small` | 54.9% | 19.53 s |
| `small-spec` | 56.6% | 20.59 s |
| | (within noise) | **+5.4% slower** |

## What that means

Speculation wins by 58% on verbatim copying at batch size 1, and does nothing — slightly
negative — on the real workload. Two reasons, both expected in hindsight:

1. **The real output is not mostly copied.** It is structured JSON whose values are
   short numbers. Only the `evidence` field quotes the document, so the fraction of
   tokens speculation can win on is small.
2. **The server is already batch-saturated.** At four concurrent parts the GPU has
   enough work to keep its units busy, and speculation's extra draft compute competes
   with real decoding instead of filling idle capacity.

So it is kept available but not enabled by default. It would be the right choice for a
single-user interactive deployment doing quote-heavy extraction, and the wrong one for
the batch evaluation path.
