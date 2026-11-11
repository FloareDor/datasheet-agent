# Load test: vLLM small server (Qwen2.5-1.5B on one L40S)

`locust -f scripts/locustfile.py --host <small server> --headless -u 20 -r 5 -t 90s ChatUser`

| metric | value |
|---|---|
| requests | 964 |
| failures | **0** |
| throughput | 10.85 req/s |
| mean | 331 ms |
| p50 | 310 ms |
| p90 | 400 ms |
| p95 | 440 ms |
| p99 | 740 ms |
| max | 974 ms |

20 concurrent users, 32-token completions, one L40S with prefix caching on.

Conservative: the verifier ablation was running against the same server throughout, so
this is throughput under contention rather than on an idle box.
