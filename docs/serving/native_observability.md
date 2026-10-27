# Native request observability

Enable structured request lifecycle logs on every Prefill and Decode service:

```bash
export OBSERVABILITY_ENABLED=1
export OBSERVABILITY_LOG_DIR=/path/to/shared/inference-logs
```

Restart the existing service with its original model, parallelism, and network
arguments. Keep native engine statistics enabled. Logs are written under
`<root>/vllm/<hostname>/<role>/` without per-token records or extra inference probes.

Optional settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OBSERVABILITY_ROLE` | `vllm` | Role label and directory, e.g. `p0` or `d0` |
| `OBSERVABILITY_LOG_RETENTION_DAYS` | `0` | Manual deletion; positive days enable automatic cleanup |
| `OBSERVABILITY_LOG_MAX_MB` | `256` | Per-segment rotation threshold in MiB |
| `OBSERVABILITY_ALERT_CONFIG` | Unset | Private JSON alert configuration |

Files rotate on UTC date changes and size. Filenames contain the UTC date; a
busy day can have multiple segments. By default no files are deleted: operators
can manually remove files older than seven days according to available space.
Optionally set `OBSERVABILITY_LOG_RETENTION_DAYS=7` for automatic cleanup by
modification time. Retention does not cap the total size of the shared directory. Logging uses bounded background queues and buffered
writes; queue overflow drops records and increments health counters.

The Router supplies the canonical `request_id` through
`X-Observability-Request-Id`. It is propagated through API, engine, scheduler,
worker, and NIXL pull metadata while preserving internal connector IDs. Every
structured record has UTC millisecond `timestamp` and integer
`timestamp_unix_ns`, derived from the same event time. Framework timestamps retain
the precision of their original creation time. Durations use a monotonic clock.

Lifecycle events cover request receipt, queueing, dispatch, prefill, KV
publication/pull/readiness, prompt recomputation, first decode start/output,
final decode output, engine statistics, and HTTP response completion.
`decode_first_output_ready` and `decode_last_output_ready` mark observed model
output availability, not individual CUDA kernel completion. Requests that finish
in prompt recomputation need not enter regular decode; cancellation or failure
need not produce a final output event. NIXL push and Mooncake internal operations
are not individually instrumented.

For optional Feishu custom-bot alerts, provide a private file:

```json
{
  "url": "https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE_ME",
  "secret": "optional-bot-signing-secret",
  "slow_request_seconds": 30
}
```

Set `OBSERVABILITY_ALERT_CONFIG` to that file and restart. Omit `secret` when the
bot does not use signature verification. Messages include the fixed keyword
`推理告警`. The sender checks Feishu business codes and retries failed delivery up
to three times. vLLM notifies GPU faults, KV failures, corrupted output, and log
failures; the companion Router handles request failures and end-to-end latency.
Notifications are rate limited per process; there is no disk replay queue.
Node/process liveness remains the responsibility of infrastructure monitoring.

Set `OBSERVABILITY_ENABLED=0` and restart to disable the added observations.
