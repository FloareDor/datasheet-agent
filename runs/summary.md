# Results

Every row is a measured run over the 100-part test split. 13 runs.

## Headline

| run | model | accuracy | tokens in | tokens out | LLM calls | tool calls | latency (s) | failures |
|---|---|---|---|---|---|---|---|---|
| `large-multiagent-test` | Qwen2.5-32B-Instruct | 75.1% | 25,432 | 1,884 | 11.39 | 12.4 | 23.9 | 0 |
| `large-singleagent-test` | Qwen2.5-32B-Instruct | 70.5% | 14,916 | 1,010 | 2.83 | 12.6 | 23.0 | 4 |
| `large-singlecall-test` | Qwen2.5-32B-Instruct | 76.2% | 15,934 | 418 | 1.00 | 0.0 | 14.2 | 9 |
| `spec-off-singlecall-test` | Qwen2.5-1.5B-Instruct | 54.9% | 15,231 | 498 | 1.00 | 0.0 | 19.5 | 7 |
| `spec-on-singlecall-test` | Qwen2.5-1.5B-Instruct | 56.6% | 15,231 | 410 | 1.00 | 0.0 | 20.6 | 5 |
| `v2trained-singleagent-test` | datasheet-grpo | 37.5% | 1,627 | 263 | 1.00 | 0.0 | 2.3 | 0 |
| `v3-multiagent-qwen1.5b-test` | Qwen2.5-1.5B-Instruct | 32.2% | 50,302 | 2,854 | 13.70 | 23.5 | 19.5 | 32 |
| `v3-singleagent-qwen1.5b-test` | Qwen2.5-1.5B-Instruct | 40.5% | 26,464 | 3,597 | 3.90 | 8.0 | 32.7 | 20 |
| `v3-singlecall-qwen1.5b-test` | Qwen2.5-1.5B-Instruct | 59.5% | 15,934 | 380 | 1.00 | 0.0 | 7.2 | 10 |
| `v3trained-singleagent-test` | datasheet-grpo | 33.7% | 6,712 | 845 | 2.36 | 1.0 | 7.2 | 4 |
| `v4trained-nocost-singleagent-test` | datasheet-grpo | 34.1% | 6,258 | 1,030 | 2.27 | 1.1 | 8.4 | 3 |
| `v6-multiagent-codeverify-qwen1.5b-test` | Qwen2.5-1.5B-Instruct | 33.0% | 48,526 | 2,393 | 14.05 | 23.8 | 14.0 | 32 |
| `v6-multiagent-verified-qwen1.5b-test` | Qwen2.5-1.5B-Instruct | 32.1% | 53,415 | 2,945 | 20.14 | 23.2 | 19.6 | 35 |

## Accuracy by field

| field | `large-multiagent-test` | `large-singleagent-test` | `large-singlecall-test` | `spec-off-singlecall-test` | `spec-on-singlecall-test` | `v2trained-singleagent-test` | `v3-multiagent-qwen1.5b-test` | `v3-singleagent-qwen1.5b-test` | `v3-singlecall-qwen1.5b-test` | `v3trained-singleagent-test` | `v4trained-nocost-singleagent-test` | `v6-multiagent-codeverify-qwen1.5b-test` | `v6-multiagent-verified-qwen1.5b-test` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gain_bandwidth_product | 84.7% | 78.6% | 82.7% | 53.8% | 53.8% | 12.2% | 37.8% | 39.8% | 54.1% | 5.1% | 6.1% | 42.9% | 39.8% |
| input_offset_voltage | 45.9% | 41.8% | 53.1% | 15.8% | 15.8% | 3.1% | 22.4% | 20.4% | 11.2% | 6.1% | 6.1% | 20.4% | 20.4% |
| mounting_style | 93.9% | 82.7% | 88.8% | 66.7% | 69.2% | 88.8% | 34.7% | 57.1% | 70.4% | 77.5% | 80.6% | 36.7% | 31.6% |
| num_channels | 78.0% | 82.0% | 87.0% | 67.5% | 70.0% | 41.0% | 17.0% | 35.0% | 72.0% | 33.0% | 36.0% | 28.0% | 19.0% |
| operating_temp_max | 76.8% | 74.8% | 74.8% | 74.4% | 79.5% | 46.5% | 35.4% | 55.6% | 76.8% | 47.5% | 45.5% | 33.3% | 35.4% |
| operating_temp_min | 79.8% | 81.8% | 77.8% | 74.4% | 76.9% | 87.9% | 34.3% | 58.6% | 83.8% | 52.5% | 47.5% | 31.3% | 34.3% |
| package_case | 71.0% | 59.0% | 75.0% | 32.5% | 32.5% | 30.0% | 15.0% | 14.0% | 31.0% | 18.0% | 22.0% | 12.0% | 14.0% |
| slew_rate | 76.0% | 71.0% | 71.0% | 55.0% | 52.5% | 9.0% | 54.0% | 42.0% | 64.0% | 2.0% | 2.0% | 53.0% | 55.0% |
| supply_voltage_max | 68.9% | 62.2% | 75.6% | 52.9% | 58.8% | 17.8% | 40.0% | 42.2% | 73.3% | 64.4% | 64.4% | 40.0% | 40.0% |

## Verifier

| run | rejections | fields repaired | accuracy |
|---|---|---|---|
| `v6-multiagent-codeverify-qwen1.5b-test` | 41 | 26 | 33.0% |
| `v6-multiagent-verified-qwen1.5b-test` | 261 | 29 | 32.1% |
