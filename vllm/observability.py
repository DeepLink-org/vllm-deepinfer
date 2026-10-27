# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native logging and real-request telemetry. No synthetic serving requests."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import hmac
import itertools
import json
import logging
import os
import queue
import socket
import threading
import time
import urllib.request
import uuid
from collections import deque
from contextlib import suppress
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import pybase64 as base64
import regex as re

_runtime = None
_runtime_lock = threading.Lock()
_metrics = None
_enabled = None
_request_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "native_request_context", default=None
)
_contexts: dict[str, deque[dict[str, Any]]] = {}
_STOP = object()


class _Flush:
    def __init__(self):
        self.done = threading.Event()
        self.error = None


_INFERENCE = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/generate",
    "/inference/v1/generate",
)


_ESSENTIAL_ALERTS = frozenset(
    (
        "cuda_oom",
        "gpu_fault",
        "kv_transport_failed",
        "kv_pull_failed",
        "corrupted_generation",
        "log_queue_full",
        "log_io_failed",
        "log_writer_stalled",
    )
)


def feishu_message(payload, secret=None):
    labels = payload["commonLabels"]
    fields = payload["commonAnnotations"]
    text = (
        f"[推理告警] {labels['severity']} {labels['alertname']}\n"
        f"状态: {payload['status']}\n服务: {labels['service']}\n"
        f"组件: {labels['component']} / {labels['role']}\n节点: {labels['node']}\n"
        f"时间: {payload['alerts'][0]['startsAt']}"
    )
    for key in (
        "request_id",
        "route",
        "elapsed_seconds",
        "threshold_seconds",
        "error",
        "message",
    ):
        if key in fields:
            text += f"\n{key}: {str(fields[key])[:1000]}"
    body = {"msg_type": "text", "content": {"text": text}}
    if secret:
        timestamp = str(int(time.time()))
        signing_key = f"{timestamp}\n{secret}".encode()
        body.update(
            timestamp=timestamp,
            sign=base64.b64encode(
                hmac.new(signing_key, b"", hashlib.sha256).digest()
            ).decode(),
        )
    return body


def enabled():
    global _enabled
    if _enabled is None:
        explicit = os.getenv("OBSERVABILITY_ENABLED")
        if explicit is not None and explicit not in ("0", "1"):
            raise ValueError("OBSERVABILITY_ENABLED must be 0 or 1")
        _enabled = (
            explicit == "1"
            if explicit is not None
            else bool(
                os.getenv("OBSERVABILITY_LOG_DIR")
                or os.getenv("VLLM_OBSERVABILITY_LOG_DIR")
                or os.getenv("OBSERVABILITY_ALERT_CONFIG")
                or os.getenv("OBSERVABILITY_WEBHOOK_URL_FILE")
            )
        )
    return _enabled


@cache
def setting(name, default):
    value = float(os.getenv("OBSERVABILITY_" + name, default))
    valid = value >= 0 if name == "LOG_RETENTION_DAYS" else value > 0
    if not valid or value == float("inf"):
        minimum = "nonnegative" if name == "LOG_RETENTION_DAYS" else "positive"
        raise ValueError(f"OBSERVABILITY_{name} must be finite and {minimum}")
    return value


def _after_fork():
    global _runtime, _runtime_lock, _contexts
    _runtime, _runtime_lock = None, threading.Lock()
    _contexts = {}
    setting.cache_clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def runtime():
    global _runtime
    if _runtime is None or _runtime.pid != os.getpid():
        with _runtime_lock:
            if _runtime is None or _runtime.pid != os.getpid():
                _runtime = Runtime()
    return _runtime


class Runtime:
    def __init__(self):
        self.pid = os.getpid()
        self.node = socket.gethostname()
        self.sequence = itertools.count(1)
        self.console = os.getenv("OBSERVABILITY_CONSOLE", "0") == "1"
        self.role = os.getenv("OBSERVABILITY_ROLE", "vllm")
        self.service = os.getenv("OBSERVABILITY_SERVICE", "online-inference")
        if not all(re.fullmatch(r"[\w.-]+", s) for s in (self.role, self.service)):
            raise ValueError("observability service/role must be simple identifiers")
        root = (
            str(Path(os.environ["OBSERVABILITY_LOG_DIR"]) / "vllm")
            if os.getenv("OBSERVABILITY_LOG_DIR")
            else os.getenv("VLLM_OBSERVABILITY_LOG_DIR")
        )
        self.directory = Path(root) / socket.gethostname() / self.role if root else None
        self.retention = setting("LOG_RETENTION_DAYS", 0) * 86400
        self.max_bytes = setting("LOG_MAX_MB", 256) * 1024 * 1024
        self.interval = setting("LOG_INTERVAL_S", 60)
        self.flush_interval = setting("LOG_FLUSH_MS", 50) / 1000
        self.records: queue.Queue[Any] = queue.Queue(maxsize=8192)
        self.alerts: queue.Queue[Any] = queue.Queue(maxsize=256)
        self.closed = False
        self.dropped = self.io_errors = self.webhook_errors = self.delivered = 0
        self.last_alert = {}
        self.alert_started = {}
        self.alert_lock = threading.Lock()
        self.conditions = {}
        self.writer_heartbeat = time.monotonic()
        self.url = None
        self.feishu = False
        self.signing_secret = None
        config_path = os.getenv("OBSERVABILITY_ALERT_CONFIG")
        if config_path:
            alert = json.loads(Path(config_path).read_text())
            self.url = alert["url"]
            self.feishu = alert.get("format", "feishu") == "feishu"
            self.signing_secret = alert.get("secret") or None
        elif os.getenv("OBSERVABILITY_WEBHOOK_URL_FILE"):
            self.url = (
                Path(os.environ["OBSERVABILITY_WEBHOOK_URL_FILE"]).read_text().strip()
            )
            self.feishu = self.url.startswith("https://open.feishu.cn/open-apis/bot/")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("alert URL must be HTTP(S)")
        self.secrets = [
            os.environ[k]
            for k in ("VLLM_API_KEY", "OPENAI_API_KEY", "HF_TOKEN")
            if os.environ.get(k)
        ]
        self.writer = threading.Thread(
            target=self._write_loop, name="vllm-log-writer", daemon=True
        )
        self.sender = threading.Thread(
            target=self._alert_loop, name="vllm-alert-sender", daemon=True
        )
        self.writer.start()
        self.sender.start()
        atexit.register(self.close)

    def put(self, record):
        if self.closed:
            return
        try:
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            self.notify("log_queue_full", "critical", {"dropped": self.dropped})

    def notify(self, kind, severity, fields, status="firing"):
        if not self.url or self.closed or kind not in _ESSENTIAL_ALERTS:
            return
        key = (kind, status)
        with self.alert_lock:
            now = time.monotonic()
            if now - self.last_alert.get(key, -1e20) < 300:
                return
            self.last_alert[key] = now
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            if status == "firing":
                starts = self.alert_started.setdefault(kind, timestamp)
            else:
                starts = self.alert_started.pop(kind, timestamp)
        labels = {
            "alertname": kind,
            "severity": severity,
            "service": self.service,
            "component": "vllm",
            "role": self.role,
            "node": socket.gethostname(),
        }
        fingerprint = hashlib.sha256(
            json.dumps(labels, sort_keys=True).encode()
        ).hexdigest()[:16]
        annotations = {k: str(v)[:2000] for k, v in fields.items()}
        annotations["summary"] = f"vLLM {kind} on {self.role}"
        payload = {
            "version": "4",
            "receiver": "native-observability",
            "status": status,
            "groupKey": fingerprint,
            "commonLabels": labels,
            "commonAnnotations": annotations,
            "alerts": [
                {
                    "status": status,
                    "labels": labels,
                    "annotations": annotations,
                    "startsAt": starts,
                    "endsAt": timestamp
                    if status == "resolved"
                    else "0001-01-01T00:00:00Z",
                    "fingerprint": fingerprint,
                }
            ],
        }
        try:
            self.alerts.put_nowait(payload)
        except queue.Full:
            self.webhook_errors += 1

    def _alert_loop(self):
        while True:
            lag = time.monotonic() - self.writer_heartbeat
            if lag > 60:
                self.notify("log_writer_stalled", "critical", {"stalled_seconds": lag})
            if _metrics is not None:
                for key, value in (
                    ("log_dropped", self.dropped),
                    ("log_io_errors", self.io_errors),
                    ("webhook_errors", self.webhook_errors),
                    ("log_writer_lag", lag),
                ):
                    _metrics[key].set(value)
            try:
                payload = self.alerts.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if payload is _STOP:
                    return
                for attempt in range(3):
                    try:
                        body = (
                            feishu_message(payload, self.signing_secret)
                            if self.feishu
                            else payload
                        )
                        assert self.url is not None
                        request = urllib.request.Request(
                            self.url,
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=3) as response:
                            raw = response.read(65536)
                        if self.feishu:
                            result = json.loads(raw)
                            if result.get("code", result.get("StatusCode")) != 0:
                                raise ValueError("Feishu rejected the alert")
                        self.delivered += 1
                        break
                    except (OSError, ValueError):
                        self.webhook_errors += 1
                        if attempt < 2:
                            time.sleep(0.2 * (attempt + 1))
            finally:
                self.alerts.task_done()

    def _write_loop(self):
        stream = None
        day = None
        size = 0
        housekeeping = 0.0
        heartbeat = time.monotonic()
        pending_flush = False
        flush_deadline = 0.0
        try:
            while True:
                self.writer_heartbeat = time.monotonic()
                try:
                    timeout = (
                        min(1.0, max(0.0, flush_deadline - time.monotonic()))
                        if pending_flush
                        else 1.0
                    )
                    record = self.records.get(timeout=timeout)
                except queue.Empty:
                    record = None
                if record is _STOP:
                    self.records.task_done()
                    break
                if isinstance(record, _Flush):
                    try:
                        if stream:
                            stream.flush()
                        pending_flush = False
                    except OSError as exc:
                        record.error = exc
                        self.io_errors += 1
                        self.notify(
                            "log_io_failed", "critical", {"errors": self.io_errors}
                        )
                    finally:
                        self.records.task_done()
                        record.done.set()
                    continue
                try:
                    if record is not None:
                        console = record.pop("_console", False)
                        created_ns = record.pop("_created_ns")
                        seconds, nanos = divmod(created_ns, 1_000_000_000)
                        record["timestamp_unix_ns"] = created_ns
                        record["timestamp"] = (
                            datetime.fromtimestamp(seconds, timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%S."
                            )
                            + f"{nanos // 1_000_000:03d}Z"
                        )
                        data = (
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        ).encode()
                        if console:
                            with suppress(OSError):
                                os.write(1, data)
                    if record is not None and self.directory:
                        now = datetime.now(timezone.utc)
                        if (
                            stream is None
                            or day != now.date()
                            or (size and size + len(data) > self.max_bytes)
                        ):
                            if stream:
                                stream.close()
                            self.directory.mkdir(parents=True, exist_ok=True)
                            filename = (
                                now.strftime("vllm.%Y%m%dT%H%M%S.%fZ.")
                                + f"{self.pid}.{uuid.uuid4().hex[:8]}.jsonl"
                            )
                            stream = (self.directory / filename).open(
                                "xb", buffering=64 * 1024
                            )
                            day, size = now.date(), 0
                            pending_flush = False
                        view = memoryview(data)
                        while view:
                            count = stream.write(view)
                            if not count:
                                raise OSError("zero byte log write")
                            view = view[count:]
                        size += len(data)
                        if not pending_flush:
                            flush_deadline = time.monotonic() + self.flush_interval
                        pending_flush = True
                    if (
                        stream is not None
                        and pending_flush
                        and (
                            time.monotonic() >= flush_deadline
                            or (
                                record is not None
                                and record.get("level") in ("ERROR", "CRITICAL")
                            )
                        )
                    ):
                        stream.flush()
                        pending_flush = False
                    if (
                        self.retention > 0
                        and self.directory
                        and time.monotonic() - housekeeping > 60
                    ):
                        housekeeping = time.monotonic()
                        if self.directory.exists():
                            for file in self.directory.glob("vllm.*.jsonl"):
                                if (
                                    not file.is_symlink()
                                    and file.stat().st_mtime
                                    < time.time() - self.retention
                                ):
                                    file.unlink()
                    if time.monotonic() - heartbeat >= self.interval:
                        heartbeat = time.monotonic()
                        log_event(
                            "observability_status",
                            log_queue=self.records.qsize(),
                            log_dropped=self.dropped,
                            log_io_errors=self.io_errors,
                            webhook_errors=self.webhook_errors,
                            webhook_delivered=self.delivered,
                        )
                except OSError:
                    self.io_errors += 1
                    self.notify("log_io_failed", "critical", {"errors": self.io_errors})
                    if stream:
                        with suppress(OSError):
                            stream.close()
                        stream = None
                    pending_flush = False
                finally:
                    if record is not None:
                        self.records.task_done()
        finally:
            if stream:
                stream.close()

    def flush(self, timeout=5):
        """Flush prior records outside the serving path."""
        if self.closed:
            return
        marker = _Flush()
        self.records.put(marker, timeout=timeout)
        if not marker.done.wait(timeout):
            raise TimeoutError("native log flush timed out")
        if marker.error is not None:
            raise marker.error

    def close(self):
        if self.closed or self.pid != os.getpid():
            return
        self.closed = True
        for channel, worker in (
            (self.records, self.writer),
            (self.alerts, self.sender),
        ):
            try:
                channel.put(_STOP, timeout=0.2)
            except queue.Full:
                continue
            if worker is not threading.current_thread():
                worker.join(timeout=3.5)


class NativeLogHandler(logging.Handler):
    def emit(self, record):
        try:
            state = runtime()
            text = record.getMessage()
            if "EngineDeadError" in text or (
                record.exc_info and record.exc_info[0].__name__ == "EngineDeadError"
            ):
                return
            if record.exc_info:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            for secret in state.secrets:
                text = text.replace(secret, "<redacted>")
            payload = {
                "_created_ns": int(record.created * 1_000_000_000),
                "request_id": (_request_context.get() or {}).get("request_id"),
                "level": record.levelname,
                "component": "vllm",
                "service": state.service,
                "node": socket.gethostname(),
                "role": state.role,
                "pid": os.getpid(),
                "logger": record.name,
                "message": text,
                **getattr(record, "observability", {}),
            }
            state.put(payload)
            if not record.name.startswith("vllm.observability"):
                for kind, pattern in (
                    ("cuda_oom", r"CUDA out of memory|OutOfMemoryError"),
                    (
                        "gpu_fault",
                        r"illegal memory access|device-side assert|uncorrectable ECC",
                    ),
                    (
                        "kv_transport_failed",
                        r"(?:NCCL|NIXL|NVSHMEM|KV.*transfer).*(?:error|fail|timed out)",
                    ),
                ):
                    if re.search(pattern, text, re.I):
                        state.notify(
                            kind, "critical", {"logger": record.name, "error": text}
                        )
                        break
        except Exception:
            self.handleError(record)


def configure_logging():
    if not enabled():
        return
    logger = logging.getLogger("vllm")
    if not any(isinstance(h, NativeLogHandler) for h in logger.handlers):
        logger.addHandler(NativeLogHandler())
    # Fail at startup for invalid configuration instead of silently missing files.
    runtime()


def log_event(event, *, _timestamp_ns=None, **fields):
    if enabled():
        state = runtime()
        state.put(
            {
                "_created_ns": time.time_ns()
                if _timestamp_ns is None
                else _timestamp_ns,
                "event_seq": next(state.sequence),
                "level": "INFO",
                "component": "vllm",
                "service": state.service,
                "node": state.node,
                "role": state.role,
                "pid": state.pid,
                "event": event,
                "request_id": None,
                "_console": state.console,
                **{
                    k: v
                    for k, v in fields.items()
                    if k not in ("observation_id", "dispatch_ns")
                },
            }
        )


def context_fields(headers, engine_request_id):
    headers = headers or {}
    return {
        "request_id": headers.get("x-native-request-id", engine_request_id),
        "client_request_id": headers.get("x-native-client-id", ""),
        "backend_request_id": headers.get("x-native-backend-id", ""),
        "engine_request_id": engine_request_id,
        "phase": headers.get("x-native-phase", "engine"),
    }


def bind_engine_request(request):
    if not enabled():
        return
    context = _request_context.get()
    headers = dict(request.trace_headers or {})
    if context:
        for key, name in (
            ("request_id", "request-id"),
            ("client_request_id", "client-id"),
            ("backend_request_id", "backend-id"),
            ("phase", "phase"),
        ):
            headers["x-native-" + name] = context[key]
        request.trace_headers = headers
    fields = context_fields(headers, request.request_id)
    external_id = getattr(request, "external_req_id", None) or request.request_id
    _contexts.setdefault(external_id, deque()).append(fields)
    log_event("engine_submit", **fields)


def queued_request(request):
    if enabled():
        log_event(
            "engine_queued", **context_fields(request.trace_headers, request.request_id)
        )


def annotate_schedule(output, requests):
    if not enabled():
        return
    entries = []
    for request_id, count in output.num_scheduled_tokens.items():
        request = requests.get(request_id)
        if request is None:
            continue
        before = request.num_computed_tokens
        prefill = before < request.num_prompt_tokens
        # Most iterations are ordinary decode. Do not rewrite per-request flags
        # or build stage metadata after the first decode dispatch.
        if not prefill and getattr(request, "_native_decode_started", False):
            continue
        if prefill:
            started = not getattr(request, "_native_prefill_started", False)
            final = before + count >= request.num_prompt_tokens and not getattr(
                request, "_native_prefill_final", False
            )
            request._native_prefill_started = True
            if final:
                request._native_prefill_final = True
        else:
            started = not getattr(request, "_native_decode_started", False)
            final = False
            request._native_decode_started = True
        if started or final:
            fields = context_fields(request.trace_headers, request_id)
            phase = "prefill" if prefill else "decode"
            if prefill and fields["phase"] == "decode":
                phase = "decode_recompute"
            entries.append(
                {
                    **fields,
                    "compute_phase": phase,
                    "start": started,
                    "final": final,
                    "scheduled_tokens": count,
                    "computed_before": before,
                    "prompt_tokens": request.num_prompt_tokens,
                    "step_id": f"{os.getpid()}-{time.monotonic_ns()}",
                }
            )
    if entries:
        output._native_observations = entries


def dispatch_workers(output):
    if not enabled():
        return
    for fields in getattr(output, "_native_observations", ()):
        fields["dispatch_ns"] = time.monotonic_ns()
        if fields["start"]:
            log_event("worker_dispatched", **fields)


def worker_execution(output, rank):
    if not enabled():
        return
    for fields in getattr(output, "_native_observations", ()):
        if fields["start"]:
            log_event(
                fields["compute_phase"] + "_compute_start", **fields, worker_rank=rank
            )


def compute_output_ready(output):
    if not enabled():
        return
    # One pair of timestamps per completed batch, not one record per token.
    output._native_output_ready_ns = time.time_ns()
    output._native_output_ready_monotonic_ns = time.monotonic_ns()
    for fields in getattr(output, "_native_observations", ()):
        if fields["compute_phase"] == "decode" and fields["start"]:
            event = "decode_first_output_ready"
        elif fields["final"]:
            event = fields["compute_phase"] + "_output_ready"
        else:
            continue
        log_event(
            event,
            **fields,
            _timestamp_ns=output._native_output_ready_ns,
            dispatch_to_result_seconds=(
                output._native_output_ready_monotonic_ns - fields["dispatch_ns"]
            )
            / 1e9,
        )


def decode_request_finished(request, output):
    if not enabled() or getattr(request, "_native_last_decode_logged", False):
        return
    fields = context_fields(request.trace_headers, request.request_id)
    if fields["phase"] == "prefill" or not request.num_output_tokens:
        return
    stamp = getattr(output, "_native_output_ready_ns", None)
    if stamp is None:
        return
    entry = next(
        (
            f
            for f in getattr(output, "_native_observations", ())
            if f["engine_request_id"] == request.request_id
        ),
        None,
    )
    request._native_last_decode_logged = True
    log_event(
        "decode_last_output_ready",
        **fields,
        _timestamp_ns=stamp,
        compute_phase=entry["compute_phase"] if entry else "decode",
        num_generation_tokens=request.num_output_tokens,
    )


def kv_scheduler_event(event, request, **details):
    if enabled():
        timing = {}
        if event == "kv_wait_started":
            request._native_kv_wait_ns = time.monotonic_ns()
        elif event == "kv_ready_for_decode":
            start = getattr(request, "_native_kv_wait_ns", None)
            timing["duration_seconds"] = (
                (time.monotonic_ns() - start) / 1e9 if start is not None else None
            )
        log_event(
            event,
            **context_fields(request.trace_headers, request.request_id),
            computed_tokens=request.num_computed_tokens,
            **timing,
            **details,
        )


def kv_meta_context(meta, request):
    if enabled():
        meta._native_context = context_fields(request.trace_headers, request.request_id)
        remote = getattr(meta, "remote", None)
        if remote is not None:
            meta._native_context.update(
                remote_engine_id=remote.engine_id,
                remote_request_id=remote.request_id,
                remote_host=remote.host,
                remote_port=remote.port,
            )
        # BlockIds is a list of block lists (one list per KV cache group).
        groups = getattr(meta, "local_block_ids", ())
        meta._native_context["local_block_count"] = sum(len(group) for group in groups)
        meta._native_context["kv_cache_group_count"] = len(groups)
        log_event("kv_pull_queued", **meta._native_context)


def kv_pull_started(meta, request_id, rank):
    if enabled():
        meta._native_started_ns = time.monotonic_ns()
        log_event(
            "kv_pull_started",
            **getattr(meta, "_native_context", {"request_id": request_id}),
            worker_rank=rank,
        )


def kv_pull_finished(meta, request_id, rank, failed):
    if enabled():
        fields = getattr(meta, "_native_context", {"request_id": request_id})
        start = getattr(meta, "_native_started_ns", None)
        log_event(
            "kv_pull_failed" if failed else "kv_pull_complete",
            **fields,
            worker_rank=rank,
            duration_seconds=(time.monotonic_ns() - start) / 1e9 if start else None,
        )
        if failed:
            runtime().notify("kv_pull_failed", "critical", fields)


def record_engine_finished(stats, engine_idx):
    if not enabled():
        return
    fields = {
        name: getattr(stats, name)
        for name in (
            "request_id",
            "e2e_latency",
            "queued_time",
            "prefill_time",
            "inference_time",
            "decode_time",
            "mean_time_per_output_token",
            "num_prompt_tokens",
            "num_generation_tokens",
            "num_cached_tokens",
            "is_corrupted",
        )
    }
    fields.update(engine_idx=engine_idx, finish_reason=str(stats.finish_reason))
    contexts = _contexts.get(stats.request_id)
    fields.update(
        contexts.popleft() if contexts else context_fields(None, stats.request_id)
    )
    if contexts is not None and not contexts:
        _contexts.pop(stats.request_id, None)
    log_event("engine_request_finished", **fields)
    if stats.is_corrupted:
        runtime().notify("corrupted_generation", "critical", fields)


def api_metrics():
    global _metrics
    if _metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        metrics = {
            "completed": Counter(
                "vllm:observability_requests",
                "Real HTTP response body outcomes",
                ["route", "outcome"],
            ),
            "e2e": Histogram(
                "vllm:observability_response_complete_seconds",
                "Actual ASGI response completion",
                ["route"],
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
            ),
            "ttft": Histogram(
                "vllm:observability_first_output_seconds",
                "First nonempty SSE output",
                ["route"],
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10, 30, 60, 120),
            ),
        }
        for key in ("log_dropped", "log_io_errors", "webhook_errors", "log_writer_lag"):
            metrics[key] = Gauge(
                "vllm:observability_" + key, "Native observability current " + key
            )
        _metrics = metrics
    return _metrics


class RequestObservabilityMiddleware:
    def __init__(self, app):
        self.app = app
        self.metrics = api_metrics()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in _INFERENCE:
            return await self.app(scope, receive, send)
        headers = {k.lower(): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = headers.get(b"x-request-id") or uuid.uuid4().hex
        if b"x-request-id" not in headers:
            scope = dict(
                scope,
                headers=[*scope["headers"], (b"x-request-id", request_id.encode())],
            )
        observation_id = headers.get(b"x-observability-request-id") or uuid.uuid4().hex
        context = {
            "request_id": observation_id,
            "backend_request_id": request_id,
            "client_request_id": headers.get(b"x-client-request-id", request_id),
            "route": scope["path"],
            "phase": headers.get(b"x-observability-phase", "api"),
        }
        context_token = _request_context.set(context)
        started = time.monotonic()
        first = last = None
        status = 0
        streaming = done = complete = protocol_error = False
        pending = b""
        chunks = 0
        gap_max = 0.0
        log_event("request_received", **context)

        async def observe_send(message):
            nonlocal \
                first, \
                last, \
                status, \
                streaming, \
                pending, \
                done, \
                chunks, \
                gap_max, \
                complete, \
                protocol_error
            if message["type"] == "http.response.start":
                message = dict(
                    message,
                    headers=[
                        *message.get("headers", []),
                        (b"x-observability-request-id", observation_id.encode()),
                    ],
                )
                status = message["status"]
                streaming = any(
                    k.lower() == b"content-type" and b"text/event-stream" in v
                    for k, v in message.get("headers", [])
                )
                log_event(
                    "response_headers",
                    **context,
                    status=status,
                    elapsed_seconds=time.monotonic() - started,
                )
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if streaming:
                    pending += body
                    while b"\n" in pending:
                        raw, pending = pending.split(b"\n", 1)
                        if not raw.startswith(b"data:"):
                            continue
                        data = raw[5:].strip()
                        if data == b"[DONE]":
                            done = True
                            continue
                        try:
                            event = json.loads(data)
                            kind = event.get("type", "")
                            if kind in (
                                "response.completed",
                                "response.incomplete",
                                "message_stop",
                            ):
                                done = True
                            if event.get("error") is not None or kind in (
                                "response.failed",
                                "error",
                            ):
                                protocol_error = True
                                log_event("stream_error_event", **context)
                            visible = any(
                                c.get("delta", {}).get("content")
                                or c.get("delta", {}).get("reasoning_content")
                                or c.get("delta", {}).get("tool_calls")
                                or c.get("text")
                                for c in event.get("choices", [])
                            )
                            delta = event.get("delta")
                            visible = (
                                visible
                                or (isinstance(delta, str) and bool(delta))
                                or (
                                    isinstance(delta, dict)
                                    and any(
                                        delta.get(key)
                                        for key in ("text", "thinking", "partial_json")
                                    )
                                )
                            )
                        except (ValueError, AttributeError, TypeError):
                            visible = False
                            protocol_error = True
                        if visible:
                            now = time.monotonic()
                            if first is None:
                                first = now
                                self.metrics["ttft"].labels(context["route"]).observe(
                                    now - started
                                )
                                log_event(
                                    "first_output",
                                    **context,
                                    ttft_seconds=now - started,
                                )
                            if last is not None:
                                gap_max = max(gap_max, now - last)
                            last, chunks = now, chunks + 1
                    if len(pending) > 1024 * 1024:
                        pending = b""
                        log_event("sse_observation_buffer_exceeded", **context)
                complete = not message.get("more_body", False)
            await send(message)

        outcome = "error"
        terminal_reason = "incomplete_response"
        error_type = None
        try:
            await self.app(scope, receive, observe_send)
            outcome = (
                "success"
                if complete
                and 0 < status < 400
                and (not streaming or done)
                and not protocol_error
                else ("client_error" if 400 <= status < 500 else "error")
            )
            terminal_reason = (
                "http_5xx"
                if status >= 500
                else "http_4xx"
                if status >= 400
                else "stream_protocol_error"
                if protocol_error
                else "stream_truncated"
                if streaming and complete and not done
                else "stream_completed"
                if outcome == "success" and streaming
                else "response_completed"
                if outcome == "success"
                else "incomplete_response"
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            terminal_reason = "cancelled"
            raise
        except Exception as exc:
            terminal_reason = "request_exception"
            error_type = type(exc).__name__
            log_event("request_exception", **context, error_type=error_type)
            raise
        finally:
            elapsed = time.monotonic() - started
            self.metrics["completed"].labels(context["route"], outcome).inc()
            self.metrics["e2e"].labels(context["route"]).observe(elapsed)
            log_event(
                "request_finished",
                **context,
                outcome=outcome,
                status=status,
                e2e_seconds=elapsed,
                ttft_seconds=first - started if first else None,
                output_chunks=chunks,
                max_chunk_gap_seconds=gap_max,
                protocol_done=done,
                streaming=streaming,
                response_complete=complete,
                terminal_reason=terminal_reason,
                error_type=error_type,
            )
            _request_context.reset(context_token)
