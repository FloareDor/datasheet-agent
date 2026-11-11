# Superseded runs

These were measured against code with bugs that were later fixed. They are kept so the
progress log's before/after numbers can be checked, and are excluded from the results
tables by living in this directory.

| run | superseded because |
|---|---|
| `baseline-singleagent` | `search` was literal: a query of `slew_rate` returned "No matches", so every electrical field scored 0%. |
| `agent-forceanswer` | Same literal-search bug; only added the forced final answer. |
| `multiagent-eqbudget` | Same bug, plus a 4-step budget that left 56/100 parts never asked for an answer. |
| `multiagent-full` | Same literal-search bug at a full 10-step budget. |
| `v2-singleagent`, `v2-multiagent` | Query relaxation was blind: `operating_temp_min` shortened to "operating temp", which substring-matched "Full **Temp**erature" in a distortion spec and returned a temperature in percent. |

The `v3-*` runs in the parent directory are the current ones.

| `v3-multiagent-verified-BUGGYVERIFIER` | The verifier's numeric check did not pass `assume_unit`, so it rejected correct bare-number answers ("36" for a field asked in volts) as unreadable. 332 rejections, 30 repairs, accuracy 32.2% -> 22.2%. Kept as the measured cost of that one bug. |
| `v4-multiagent-verified` | Correct verifier, but run before `pre_retry_values` was recorded, so the verifier could not be graded as a classifier. Result (32.2% -> 22.3%) reproduced in `v5`. |
| `v5-*-NORETRYGUARD` | Measured before retries were forbidden from erasing values. These are the runs that *established* the problem: the model verifier had 55.8% precision and its retries destroyed 98 correct values (94 nulled) while repairing 9. Code-only checks had 82% precision and were net +4. |
