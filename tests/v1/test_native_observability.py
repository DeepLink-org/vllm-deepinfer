# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import logging
import os
import pickle
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vllm.observability as obs


def wait_for(predicate, seconds=5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for native event")


class NativeTests(unittest.TestCase):
    def setUp(self):
        if obs._runtime:
            obs._runtime.close()
        obs._runtime = None
        obs._enabled = None
        obs.setting.cache_clear()
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {
                "VLLM_OBSERVABILITY_LOG_DIR": self.temp.name,
                "OBSERVABILITY_ROLE": "native-test",
                "OBSERVABILITY_SERVICE": "native-tests",
                "OBSERVABILITY_LOG_INTERVAL_S": ".02",
                "OBSERVABILITY_LOG_MAX_MB": ".002",
                "OBSERVABILITY_TTFT_MS": "30",
                "OBSERVABILITY_E2E_MS": "30000",
                "OBSERVABILITY_CHUNK_GAP_MS": "1000",
            },
        )
        self.env.start()
        os.environ.pop("OBSERVABILITY_WEBHOOK_URL_FILE", None)
        os.environ.pop("OBSERVABILITY_ALERT_CONFIG", None)
        logging.getLogger("vllm").setLevel(logging.INFO)
        obs.configure_logging()

    def tearDown(self):
        obs.runtime().close()
        obs._enabled = None
        obs.setting.cache_clear()
        self.env.stop()
        self.temp.cleanup()
        obs._runtime = None

    def records(self):
        obs.runtime().flush()
        rows = [
            json.loads(line)
            for p in Path(self.temp.name).rglob("vllm.*.jsonl")
            for line in p.read_text().splitlines()
        ]
        return sorted(
            rows,
            key=lambda row: (
                row.get("timestamp", ""),
                row.get("pid", 0),
                row.get("event_seq", 0),
            ),
        )

    def test_native_size_rotation_and_exception(self):
        for n in range(20):
            logging.getLogger("vllm.native-test").info("record %s %s", n, "x" * 200)
        try:
            raise RuntimeError("native traceback fixture")
        except RuntimeError:
            logging.getLogger("vllm.native-test").exception("fixture")
        records = self.records()
        self.assertGreater(len(list(Path(self.temp.name).rglob("vllm.*.jsonl"))), 1)
        self.assertEqual(sum("record " in row["message"] for row in records), 20)
        self.assertTrue(any("Traceback" in row["message"] for row in records))

    def test_retention_removes_only_native_prefix(self):
        obs.runtime().close()
        obs._runtime = None
        os.environ["OBSERVABILITY_LOG_RETENTION_DAYS"] = "7"
        obs.setting.cache_clear()
        directory = obs.runtime().directory
        directory.mkdir(parents=True, exist_ok=True)
        old = directory / "vllm.old.jsonl"
        other = directory / "manual.log"
        for path in (old, other):
            path.write_text("old")
            os.utime(path, (time.time() - 8 * 86400,) * 2)
        logging.getLogger("vllm.native-test").info("retention fixture")
        wait_for(lambda: not old.exists())
        self.assertTrue(other.exists())

    def test_default_and_zero_retention_preserve_old_daily_files(self):
        os.environ.pop("OBSERVABILITY_LOG_RETENTION_DAYS", None)
        obs.setting.cache_clear()
        self.assertEqual(obs.runtime().retention, 0)
        directory = obs.runtime().directory
        directory.mkdir(parents=True, exist_ok=True)
        old = directory / "vllm.20000101T000000.0.jsonl"
        old.write_text("{}\n")
        os.utime(old, (time.time() - 30 * 86400,) * 2)
        logging.getLogger("vllm.native-test").info("manual retention fixture")
        self.records()
        self.assertTrue(old.exists())
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                os.environ["OBSERVABILITY_LOG_RETENTION_DAYS"] = value
                obs.setting.cache_clear()
                if value == "0":
                    self.assertEqual(obs.setting("LOG_RETENTION_DAYS", 0), 0)
                else:
                    with self.assertRaises(ValueError):
                        obs.setting("LOG_RETENTION_DAYS", 0)

    def test_fork_reinitializes_writer(self):
        parent = os.getpid()
        child = os.fork()
        if child == 0:
            logging.getLogger("vllm.native-test").info("child fork record")
            obs.runtime().close()
            os._exit(0)
        _, status = os.waitpid(child, 0)
        self.assertEqual(status, 0)
        records = self.records()
        self.assertTrue(
            any(
                r["pid"] == child and "child fork record" in r["message"]
                for r in records
            )
        )
        self.assertEqual(obs.runtime().pid, parent)

    def run_app(self, events, delay=0.0, request_headers=None):
        sent = []

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for i, body in enumerate(events):
                if delay:
                    await asyncio.sleep(delay)
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": i < len(events) - 1,
                    }
                )

        middleware = obs.RequestObservabilityMiddleware(app)

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b""}

        asyncio.run(
            middleware(
                {
                    "type": "http",
                    "path": "/v1/chat/completions",
                    "headers": request_headers
                    or [
                        (b"x-request-id", b"backend-id"),
                        (b"x-client-request-id", b"client-id"),
                    ],
                },
                receive,
                send,
            )
        )
        return sent

    def test_complete_sse_and_ids_and_split_packets(self):
        chunks = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"hi"}}',
            b"]}\n\n",
            b"data: [DONE]\n\n",
        ]
        sent = self.run_app(chunks)
        finished = [r for r in self.records() if r.get("event") == "request_finished"][
            -1
        ]
        self.assertEqual(finished["outcome"], "success")
        self.assertEqual(finished["terminal_reason"], "stream_completed")
        self.assertTrue(finished["streaming"])
        self.assertEqual(finished["output_chunks"], 1)
        self.assertEqual(finished["backend_request_id"], "backend-id")
        self.assertTrue(finished["request_id"])
        self.assertEqual(finished["client_request_id"], "client-id")
        self.assertEqual(b"".join(m.get("body", b"") for m in sent), b"".join(chunks))

    def test_truncated_and_error_event_are_not_success(self):
        for chunks in (
            [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'],
            [b'data: {"error":{"message":"fixture"}}\n\n', b"data: [DONE]\n\n"],
        ):
            self.run_app(chunks)
            last = [r for r in self.records() if r.get("event") == "request_finished"][
                -1
            ]
            self.assertEqual(last["outcome"], "error")
            self.assertIn(
                last["terminal_reason"], ("stream_truncated", "stream_protocol_error")
            )

    def test_response_and_anthropic_terminal_events(self):
        for chunks in (
            [
                b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n',
                b'data: {"type":"response.completed"}\n\n',
            ],
            [
                b'data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n',
                b'data: {"type":"message_stop"}\n\n',
            ],
        ):
            self.run_app(chunks)
            last = [r for r in self.records() if r.get("event") == "request_finished"][
                -1
            ]
            self.assertEqual(last["outcome"], "success")

    def test_backend_does_not_duplicate_router_latency_alerts(self):
        notifications = []
        obs.runtime().notify = lambda *args, **kwargs: notifications.append(args)
        self.run_app(
            [
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            delay=0.08,
        )
        self.assertEqual(notifications, [])
        self.assertFalse(
            any(r.get("event", "").endswith("_slow") for r in self.records())
        )
        self.assertIsNone(obs._request_context.get())

    def test_disabled_fast_path_has_no_metadata(self):
        with patch.dict(os.environ, {"OBSERVABILITY_ENABLED": "0"}):
            obs._enabled = None
            request = SimpleNamespace(request_id="engine-off", trace_headers=None)
            output = SimpleNamespace(num_scheduled_tokens={"engine-off": 8})
            obs.bind_engine_request(request)
            obs.annotate_schedule(output, {"engine-off": request})
            self.assertIsNone(request.trace_headers)
            self.assertFalse(hasattr(output, "_native_observations"))
        obs._enabled = None

    def test_lifecycle_context_preserves_trace_and_pickle(self):
        context = {
            "request_id": "canonical",
            "client_request_id": "client",
            "backend_request_id": "connector-id",
            "phase": "prefill",
        }
        request = SimpleNamespace(
            request_id="engine-internal",
            trace_headers={"traceparent": "kept"},
            num_computed_tokens=0,
            num_prompt_tokens=32,
        )
        token = obs._request_context.set(context)
        try:
            obs.bind_engine_request(request)
        finally:
            obs._request_context.reset(token)
        self.assertEqual(request.trace_headers["traceparent"], "kept")
        obs.queued_request(request)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 32})
        obs.annotate_schedule(output, {request.request_id: request})
        obs.dispatch_workers(output)
        copied = pickle.loads(pickle.dumps(output))
        obs.worker_execution(copied, 0)
        obs.compute_output_ready(output)
        meta = SimpleNamespace()
        obs.kv_meta_context(meta, request)
        meta = pickle.loads(pickle.dumps(meta))
        obs.kv_pull_started(meta, request.request_id, 0)
        obs.kv_pull_finished(meta, request.request_id, 0, False)
        records = sorted(
            (
                row
                for row in self.records()
                if row.get("engine_request_id") == request.request_id
            ),
            key=lambda row: row["event_seq"],
        )
        self.assertTrue(all(row["request_id"] == "canonical" for row in records))
        self.assertEqual(
            [r["event"] for r in records],
            [
                "engine_submit",
                "engine_queued",
                "worker_dispatched",
                "prefill_compute_start",
                "prefill_output_ready",
                "kv_pull_queued",
                "kv_pull_started",
                "kv_pull_complete",
            ],
        )

    def test_kv_source_block_counts_and_wait_duration(self):
        request = SimpleNamespace(
            request_id="kv-engine", trace_headers=None, num_computed_tokens=7
        )
        meta = SimpleNamespace(
            local_block_ids=([1, 2], [3, 4, 5]),
            remote=SimpleNamespace(
                engine_id="producer",
                request_id="remote-request",
                host="10.0.0.1",
                port=5557,
            ),
        )
        obs.kv_meta_context(meta, request)
        self.assertEqual(meta._native_context["local_block_count"], 5)
        self.assertEqual(meta._native_context["kv_cache_group_count"], 2)
        self.assertEqual(meta._native_context["remote_engine_id"], "producer")
        with patch.object(
            obs.time, "monotonic_ns", side_effect=[1000000000, 3500000000]
        ):
            obs.kv_scheduler_event("kv_wait_started", request)
            obs.kv_scheduler_event("kv_ready_for_decode", request)
        ready = [r for r in self.records() if r.get("event") == "kv_ready_for_decode"][
            -1
        ]
        self.assertEqual(ready["duration_seconds"], 2.5)
        obs.kv_pull_finished(meta, request.request_id, 0, False)
        finished = [r for r in self.records() if r.get("event") == "kv_pull_complete"][
            -1
        ]
        self.assertIsNone(finished["duration_seconds"])

    def test_exception_and_cancel_have_terminal_reason(self):
        for exception, reason in (
            (ValueError, "request_exception"),
            (asyncio.CancelledError, "cancelled"),
        ):

            async def app(scope, receive, send, _exception=exception):
                raise _exception()

            async def send(message):
                pass

            async def receive():
                return {"type": "http.request", "body": b""}

            middleware = obs.RequestObservabilityMiddleware(app)
            with self.assertRaises(exception):
                asyncio.run(
                    middleware(
                        {"type": "http", "path": "/v1/completions", "headers": []},
                        receive,
                        send,
                    )
                )
            finished = [
                r for r in self.records() if r.get("event") == "request_finished"
            ][-1]
            self.assertEqual(finished["terminal_reason"], reason)
            self.assertFalse(finished["response_complete"])
            self.assertEqual(
                finished["error_type"],
                "ValueError" if exception is ValueError else None,
            )

    def test_runtime_alarm_classification(self):
        notifications = []
        obs.runtime().notify = (
            lambda kind, severity, fields, status="firing": notifications.append(
                (kind, severity)
            )
        )
        logger = logging.getLogger("vllm.native-alarm-test")
        for message in (
            "CUDA out of memory",
            "GPU illegal memory access",
            "EngineDeadError: engine dead",
            "NIXL transfer failed",
            "unclassified runtime failure",
        ):
            logger.error(message)
        self.assertEqual(
            notifications,
            [
                ("cuda_oom", "critical"),
                ("gpu_fault", "critical"),
                ("kv_transport_failed", "critical"),
            ],
        )
        self.assertFalse(
            any("EngineDeadError" in r.get("message", "") for r in self.records())
        )

    def test_http_5xx_and_engine_statistics_alarms(self):
        notifications = []
        obs.runtime().notify = (
            lambda kind, severity, fields, status="firing": notifications.append(
                (kind, severity)
            )
        )

        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error":"fixture"}',
                    "more_body": False,
                }
            )

        async def send(message):
            pass

        async def receive():
            return {"type": "http.request", "body": b""}

        middleware = obs.RequestObservabilityMiddleware(app)
        asyncio.run(
            middleware(
                {"type": "http", "path": "/v1/completions", "headers": []},
                receive,
                send,
            )
        )
        stats = SimpleNamespace(
            request_id="stats-fixture",
            e2e_latency=5.0,
            queued_time=3.0,
            prefill_time=0.1,
            inference_time=1.0,
            decode_time=0.9,
            mean_time_per_output_token=0.2,
            num_prompt_tokens=32,
            num_generation_tokens=8,
            num_cached_tokens=0,
            is_corrupted=True,
            finish_reason="error",
        )
        obs.record_engine_finished(stats, 0)
        self.assertNotIn(("request_http_5xx", "critical"), notifications)
        self.assertNotIn(("engine_queue_slow", "warning"), notifications)
        self.assertNotIn(("engine_tpot_slow", "warning"), notifications)
        self.assertIn(("corrupted_generation", "critical"), notifications)
        finished = [r for r in self.records() if r.get("event") == "request_finished"][
            -1
        ]
        self.assertEqual(finished["terminal_reason"], "http_5xx")

    def test_sparse_record_flushes_within_interval(self):
        obs.log_event("sparse_flush_fixture", request_id="sparse")

        def visible():
            return any(
                "sparse_flush_fixture" in p.read_text()
                for p in Path(self.temp.name).rglob("vllm.*.jsonl")
            )

        wait_for(visible, seconds=0.5)

    def test_close_flushes_pending_records(self):
        for index in range(10):
            obs.log_event("close_flush_fixture", request_id="close", index=index)
        obs.runtime().close()
        rows = [
            json.loads(line)
            for p in Path(self.temp.name).rglob("vllm.*.jsonl")
            for line in p.read_text().splitlines()
        ]
        self.assertEqual(sum(r.get("event") == "close_flush_fixture" for r in rows), 10)

    def test_settings_are_cached_until_restart(self):
        first = obs.setting("LOG_MAX_MB", 256)
        with patch.dict(os.environ, {"OBSERVABILITY_LOG_MAX_MB": "999"}):
            self.assertEqual(obs.setting("LOG_MAX_MB", 256), first)

    def test_decode_fast_path_keeps_stage_boundaries(self):
        request = SimpleNamespace(
            request_id="decode-flow",
            trace_headers=None,
            num_computed_tokens=0,
            num_prompt_tokens=32,
        )
        entries = []
        for before, count in [(0, 16), (16, 16), (32, 1), (33, 1), (34, 1)]:
            request.num_computed_tokens = before
            output = SimpleNamespace(num_scheduled_tokens={request.request_id: count})
            obs.annotate_schedule(output, {request.request_id: request})
            entries.append(getattr(output, "_native_observations", []))
        self.assertEqual(
            [
                (e[0]["compute_phase"], e[0]["start"], e[0]["final"])
                for e in entries[:3]
            ],
            [
                ("prefill", True, False),
                ("prefill", False, True),
                ("decode", True, False),
            ],
        )
        self.assertEqual(entries[3:], [[], []])

    def test_direct_webhook_retry_and_recovery_contract(self):
        batches = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                batches.append(payload)
                self.send_response(500 if len(batches) == 1 else 200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        obs.runtime().close()
        obs._runtime = None
        path = Path(self.temp.name) / "private.url"
        path.write_text(f"http://127.0.0.1:{server.server_port}/alerts")
        os.environ["OBSERVABILITY_WEBHOOK_URL_FILE"] = str(path)
        try:
            obs.runtime().notify("log_io_failed", "critical", {"elapsed_seconds": 0.5})
            wait_for(lambda: obs.runtime().delivered == 1)
            obs.runtime().notify("log_io_failed", "critical", {}, status="resolved")
            wait_for(lambda: obs.runtime().delivered == 2)
            self.assertGreaterEqual(obs.runtime().webhook_errors, 1)
            self.assertEqual(batches[-1]["status"], "resolved")
            self.assertEqual(
                batches[0]["alerts"][0]["startsAt"],
                batches[-1]["alerts"][0]["startsAt"],
            )
            self.assertTrue(
                all(
                    isinstance(value, str)
                    for batch in batches
                    for value in batch["commonAnnotations"].values()
                )
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_unified_timestamps_for_events_and_framework_logs(self):
        ns = 1793081349784357766
        with patch.object(obs.time, "time_ns", return_value=ns):
            obs.log_event("timestamp_fixture", request_id="clock")
        logging.getLogger("vllm.clock").info("framework clock fixture")
        rows = self.records()
        event = next(r for r in rows if r.get("event") == "timestamp_fixture")
        self.assertEqual(event["timestamp"], "2026-10-27T06:09:09.784Z")
        self.assertEqual(event["timestamp_unix_ns"], ns)
        for row in rows:
            self.assertTrue(row["timestamp"].endswith("Z"))
            expected = (
                datetime.fromtimestamp(
                    row["timestamp_unix_ns"] // 10**9, timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{row['timestamp_unix_ns'] // 10**6 % 1000:03d}Z"
            )
            self.assertEqual(row["timestamp"], expected)

    def test_common_directory_and_compact_fields(self):
        obs.runtime().close()
        obs._runtime = None
        with patch.dict(os.environ, {"OBSERVABILITY_LOG_DIR": self.temp.name}):
            obs.log_event(
                "compact_fixture",
                request_id="compact",
                observation_id="duplicate",
                dispatch_ns=1,
            )
            row = next(r for r in self.records() if r.get("event") == "compact_fixture")
            self.assertNotIn("observation_id", row)
            self.assertNotIn("dispatch_ns", row)
            self.assertEqual(
                obs.runtime().directory,
                Path(self.temp.name) / "vllm" / obs.runtime().node / "native-test",
            )

    def test_feishu_signature_business_error_retry_and_noise_filter(self):
        import hashlib
        import hmac

        import pybase64 as base64

        batches = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                batches.append(payload)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps({"code": 19021 if len(batches) == 1 else 0}).encode()
                )

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        obs.runtime().close()
        obs._runtime = None
        path = Path(self.temp.name) / "alert.json"
        path.write_text(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{server.server_port}/hook",
                    "secret": "fixture-signing-secret",
                }
            )
        )
        os.environ["OBSERVABILITY_ALERT_CONFIG"] = str(path)
        try:
            state = obs.runtime()
            state.notify("engine_queue_slow", "warning", {})
            state.notify("runtime_exception", "warning", {})
            self.assertTrue(state.alerts.empty())
            state.notify(
                "cuda_oom", "critical", {"request_id": "req-feishu", "error": "fixture"}
            )
            wait_for(lambda: state.delivered == 1)
            self.assertEqual(len(batches), 2)
            body = batches[-1]
            expected = base64.b64encode(
                hmac.new(
                    (body["timestamp"] + "\nfixture-signing-secret").encode(),
                    b"",
                    hashlib.sha256,
                ).digest()
            ).decode()
            self.assertEqual(body["sign"], expected)
            self.assertEqual(body["msg_type"], "text")
            self.assertIn("推理告警", body["content"]["text"])
            self.assertIn("req-feishu", body["content"]["text"])
            self.assertNotIn("fixture-signing-secret", body["content"]["text"])
            self.assertGreaterEqual(state.webhook_errors, 1)
            state.notify("cuda_oom", "critical", {})
            time.sleep(0.1)
            self.assertEqual(len(batches), 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_first_and_last_decode_boundaries_without_per_token_logs(self):
        request = SimpleNamespace(
            request_id="decode-boundary",
            trace_headers={"x-native-phase": "decode"},
            num_computed_tokens=32,
            num_prompt_tokens=32,
            num_output_tokens=1,
        )
        first = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        obs.annotate_schedule(first, {request.request_id: request})
        obs.dispatch_workers(first)
        obs.worker_execution(first, 0)
        first_ns = 1793081349784357766
        with patch.object(obs.time, "time_ns", return_value=first_ns):
            obs.compute_output_ready(first)
        for before in range(33, 38):
            request.num_computed_tokens = before
            batch = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
            obs.annotate_schedule(batch, {request.request_id: request})
            obs.dispatch_workers(batch)
            obs.worker_execution(batch, 0)
            obs.compute_output_ready(batch)
        final_ns = first_ns + 123456789
        with patch.object(obs.time, "time_ns", return_value=final_ns):
            obs.compute_output_ready(batch)
        request.num_output_tokens = 8
        obs.decode_request_finished(request, batch)
        obs.decode_request_finished(request, batch)
        rows = [r for r in self.records() if r.get("request_id") == request.request_id]
        events = [r["event"] for r in rows]
        self.assertEqual(events.count("decode_compute_start"), 1)
        self.assertEqual(events.count("decode_first_output_ready"), 1)
        self.assertEqual(events.count("decode_last_output_ready"), 1)
        end = next(r for r in rows if r["event"] == "decode_first_output_ready")
        last = next(r for r in rows if r["event"] == "decode_last_output_ready")
        self.assertEqual(end["timestamp_unix_ns"], first_ns)
        self.assertEqual(last["timestamp_unix_ns"], final_ns)
        self.assertEqual(last["num_generation_tokens"], 8)
        self.assertLess(end["timestamp_unix_ns"], last["timestamp_unix_ns"])

    def test_last_output_can_be_in_recompute_and_prefill_is_excluded(self):
        request = SimpleNamespace(
            request_id="short-decode",
            trace_headers={"x-native-phase": "decode"},
            num_computed_tokens=31,
            num_prompt_tokens=32,
            num_output_tokens=1,
        )
        batch = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        obs.annotate_schedule(batch, {request.request_id: request})
        obs.dispatch_workers(batch)
        obs.compute_output_ready(batch)
        obs.decode_request_finished(request, batch)
        rows = [r for r in self.records() if r.get("request_id") == request.request_id]
        self.assertFalse(any(r["event"] == "decode_first_output_ready" for r in rows))
        self.assertEqual(
            next(r for r in rows if r["event"] == "decode_last_output_ready")[
                "compute_phase"
            ],
            "decode_recompute",
        )
        producer = SimpleNamespace(
            request_id="producer-only",
            trace_headers={"x-native-phase": "prefill"},
            num_output_tokens=1,
        )
        obs.decode_request_finished(producer, batch)
        self.assertFalse(
            any(r.get("request_id") == "producer-only" for r in self.records())
        )


if __name__ == "__main__":
    unittest.main()
