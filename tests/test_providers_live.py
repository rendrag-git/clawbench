import os
import unittest

from openclaw_bench.providers import (
    IncusExecProbe,
    LocalProbe,
    run_detection,
)


@unittest.skipUnless(os.environ.get("OC_BENCH_LIVE") == "1", "OC_BENCH_LIVE!=1; skipping live test")
class LiveGptOssDetectionTests(unittest.TestCase):
    def test_detection_finds_gpt_oss_via_oc_stack(self):
        from pathlib import Path

        # GPT-OSS 20B binds to 10.68.198.1:8000 (incusbr0 bridge address), not
        # loopback.  Pass both so the cascade finds it on this machine and from
        # inside oc-stack via the same bridge address.
        report = run_detection(
            providers=["vllm"],
            probes=[LocalProbe(), IncusExecProbe("oc-stack")],
            home=Path.home(),
            probe_hosts=["127.0.0.1", "10.68.198.1"],
        )
        vllm_candidates = [c for c in report.candidates if c.provider == "vllm"]
        self.assertTrue(vllm_candidates, msg=f"no vllm candidate; report={report}")
        candidate = vllm_candidates[0]
        self.assertIn("gpt-oss-20b", candidate.models)
        host_result = candidate.probe_results.get("host")
        runtime_result = candidate.probe_results.get("incus:oc-stack")
        self.assertTrue(host_result and host_result.ok)
        self.assertTrue(runtime_result and runtime_result.ok)
        self.assertFalse(
            any(f.startswith("reachable_from_host_not_runtime") for f in report.findings),
            msg=f"unexpected mismatch finding: {report.findings}",
        )
