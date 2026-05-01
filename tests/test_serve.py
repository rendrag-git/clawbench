import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from openclaw_bench.models import ModelSpec
from openclaw_bench.serve import _serve_env, serve_model


class ServeTests(unittest.TestCase):
    def test_serve_env_adds_cuda_defaults_when_cuda_exists(self):
        with patch("openclaw_bench.serve.Path.exists", return_value=True):
            with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
                env = _serve_env()
        self.assertEqual(env["CUDA_HOME"], "/usr/local/cuda")
        self.assertIn("/usr/local/cuda/bin", env["PATH"])
        self.assertIn("/home/ubuntu/.venvs/vllm/bin", env["PATH"])
        self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/local/cuda/lib64")

    def test_local_model_without_probe_or_explicit_assumption_fails_readiness(self):
        model = ModelSpec(model_id="local", served_model_name="local")
        with serve_model(model, timeout_s=1) as state:
            self.assertEqual(state.load_success, False)
            self.assertEqual(state.failure_type, "model_load_failed")

    def test_process_without_health_check_fails_readiness(self):
        model = ModelSpec(
            model_id="local",
            served_model_name="local",
            serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
        )
        with serve_model(model, timeout_s=1) as state:
            self.assertEqual(state.load_success, False)
            self.assertEqual(state.failure_type, "model_load_failed")

    def test_non_2xx_health_check_fails_readiness(self):
        server = HTTPServer(("127.0.0.1", 0), _NotReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="local",
                serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
                health_check_url=f"http://127.0.0.1:{server.server_port}/health",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, False)
                self.assertEqual(state.failure_type, "server_timeout")
        finally:
            server.shutdown()
            server.server_close()

    def test_existing_health_endpoint_without_serve_command_passes_readiness(self):
        server = HTTPServer(("127.0.0.1", 0), _ReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="local",
                health_check_url=f"http://127.0.0.1:{server.server_port}/health",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, True)
                self.assertEqual(state.failure_type, None)
                self.assertEqual(state.started, False)
        finally:
            server.shutdown()
            server.server_close()

    def test_health_check_uses_declared_api_env_bearer_token(self):
        _AuthReadyHandler.authorization = None
        server = HTTPServer(("127.0.0.1", 0), _AuthReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="local",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_env="VLLM_API_KEY",
            )
            with patch.dict("os.environ", {"VLLM_API_KEY": "test-token"}):
                with serve_model(model, timeout_s=2) as state:
                    self.assertEqual(state.load_success, True)
                    self.assertEqual(state.failure_type, None)
                    self.assertEqual(_AuthReadyHandler.authorization, "Bearer test-token")
        finally:
            server.shutdown()
            server.server_close()

    def test_api_base_chat_smoke_passes_after_health(self):
        _VllmLikeHandler.requested_model = None
        _VllmLikeHandler.post_count = 0
        server = HTTPServer(("127.0.0.1", 0), _VllmLikeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="local-smoke",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, True)
                self.assertEqual(state.failure_type, None)
                self.assertEqual(_VllmLikeHandler.requested_model, "local-smoke")
                self.assertIn("Route smoke returned HTTP 200", state.notes)
                self.assertEqual(state.route_probe["success"], True)
                self.assertEqual(state.route_probe["completion_tokens"], 5)
                self.assertEqual(state.route_probe["sample_count"], 3)
                self.assertEqual(len(state.route_probe["samples"]), 3)
                self.assertGreater(state.route_probe["tokens_per_s_p50"], 0)
                self.assertGreater(state.route_probe["tokens_per_s_p95"], 0)
                self.assertGreater(state.route_probe["prompt_chars"], 200)
                self.assertEqual(_VllmLikeHandler.post_count, 4)
        finally:
            server.shutdown()
            server.server_close()

    def test_api_base_serve_probe_failure_marks_probe_failed(self):
        _ProbeFailureHandler.post_count = 0
        server = HTTPServer(("127.0.0.1", 0), _ProbeFailureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="probe-failure",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, False)
                self.assertEqual(state.failure_type, "serve_probe_failed")
                self.assertIn("Serve probe returned HTTP 500", state.notes)
                self.assertEqual(_ProbeFailureHandler.post_count, 2)
        finally:
            server.shutdown()
            server.server_close()

    def test_api_base_chat_smoke_failure_marks_route_failed(self):
        server = HTTPServer(("127.0.0.1", 0), _BrokenVllmRouteHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="missing-model",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, False)
                self.assertEqual(state.failure_type, "model_route_failed")
                self.assertIn("not found", state.notes)
        finally:
            server.shutdown()
            server.server_close()

    def test_api_base_tool_parser_failure_is_classified(self):
        server = HTTPServer(("127.0.0.1", 0), _ToolParserFailureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="tool-parser-missing",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, False)
                self.assertEqual(state.failure_type, "tool_parser_missing")
        finally:
            server.shutdown()
            server.server_close()

    def test_api_base_context_window_probe_failure_is_classified(self):
        server = HTTPServer(("127.0.0.1", 0), _ContextWindowProbeFailureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ModelSpec(
                model_id="local",
                served_model_name="context-too-small",
                health_check_url=f"http://127.0.0.1:{server.server_port}/v1/models",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
            )
            with serve_model(model, timeout_s=2) as state:
                self.assertEqual(state.load_success, False)
                self.assertEqual(state.failure_type, "context_window_exceeded")
        finally:
            server.shutdown()
            server.server_close()


class _NotReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class _ReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        return


class _AuthReadyHandler(BaseHTTPRequestHandler):
    authorization = None

    def do_GET(self):
        type(self).authorization = self.headers.get("Authorization")
        if type(self).authorization != "Bearer test-token":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def log_message(self, format, *args):
        return


class _VllmLikeHandler(BaseHTTPRequestHandler):
    requested_model = None
    post_count = 0

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        type(self).post_count += 1
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requested_model = payload["model"]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"choices":[{"message":{"content":"ok OC_BENCH_PROBE"}}],"usage":{"completion_tokens":5,"total_tokens":42}}')

    def log_message(self, format, *args):
        return


class _BrokenVllmRouteHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'{"error":"model not found"}')

    def log_message(self, format, *args):
        return


class _ProbeFailureHandler(BaseHTTPRequestHandler):
    post_count = 0

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        type(self).post_count += 1
        if type(self).post_count == 1:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')
            return
        self.send_response(500)
        self.end_headers()
        self.wfile.write(b'{"error":"probe failed"}')

    def log_message(self, format, *args):
        return


class _ToolParserFailureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'{"error":"\\"auto\\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"}')

    def log_message(self, format, *args):
        return


class _ContextWindowProbeFailureHandler(BaseHTTPRequestHandler):
    post_count = 0

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        type(self).post_count += 1
        if type(self).post_count == 1:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')
            return
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'{"error":"This model maximum context length is 4096 tokens, but the request exceeds that limit"}')

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    unittest.main()
