import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from openclaw_bench.providers.probes import LocalProbe, ProbeResult


class _Handler(BaseHTTPRequestHandler):
    payload = {"object": "list", "data": []}

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):  # silence stderr noise
        return


def _serve_once():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class LocalProbeTests(unittest.TestCase):
    def test_http_get_returns_success_result_with_body(self):
        server = _serve_once()
        try:
            host, port = server.server_address
            url = f"http://{host}:{port}/v1/models"
            result = LocalProbe().http_get(url, timeout_s=2.0)
        finally:
            server.shutdown()
            server.server_close()

        self.assertIsInstance(result, ProbeResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(json.loads(result.body), {"object": "list", "data": []})
        self.assertEqual(result.probe_name, "host")
        self.assertIsNone(result.error)


class LocalProbeTimeoutTests(unittest.TestCase):
    def test_http_get_returns_failure_when_endpoint_unreachable(self):
        # Reserved test address per RFC 5737; routes nowhere fast enough to hit timeout.
        result = LocalProbe().http_get("http://192.0.2.1:18080/v1/models", timeout_s=0.5)
        self.assertFalse(result.ok)
        self.assertIsNone(result.status_code)
        self.assertIsNotNone(result.error)


class _404Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"error": "not found"}).encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):  # silence stderr noise
        return


class LocalProbeHTTPErrorTests(unittest.TestCase):
    def test_http_get_returns_failure_for_4xx_response(self):
        server = HTTPServer(("127.0.0.1", 0), _404Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            url = f"http://{host}:{port}/v1/models"
            result = LocalProbe().http_get(url, timeout_s=2.0)
        finally:
            server.shutdown()
            server.server_close()

        self.assertFalse(result.ok)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.error, "http_404")
        self.assertEqual(json.loads(result.body), {"error": "not found"})
        self.assertEqual(result.probe_name, "host")
