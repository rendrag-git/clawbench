import unittest

from openclaw_bench.telemetry import GpuTelemetry, GpuSample, parse_nvidia_smi


class TelemetryTests(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self):
        samples = parse_nvidia_smi(
            "0, NVIDIA RTX A4000, 16376, 1024, 12\n"
            "1, NVIDIA RTX PRO 5000 Blackwell, 48935, 2048, 34\n"
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].index, 0)
        self.assertEqual(samples[1].memory_used_mb, 2048)

    def test_gpu_telemetry_peak_values(self):
        telemetry = GpuTelemetry(
            available=True,
            samples=[
                GpuSample(index=0, name="a", memory_total_mb=100, memory_used_mb=10, utilization_pct=5),
                GpuSample(index=0, name="a", memory_total_mb=100, memory_used_mb=25, utilization_pct=80),
            ],
        )

        self.assertEqual(telemetry.peak_vram_mb, 25)
        self.assertEqual(telemetry.max_gpu_utilization_pct, 80)

    def test_gpu_telemetry_row_includes_device_inventory(self):
        telemetry = GpuTelemetry(
            available=True,
            samples=[
                GpuSample(index=1, name="gpu-b", memory_total_mb=200, memory_used_mb=20, utilization_pct=10),
                GpuSample(index=0, name="gpu-a", memory_total_mb=100, memory_used_mb=30, utilization_pct=20),
            ],
        )

        row = telemetry.to_row()

        self.assertEqual(row["devices"][0]["index"], 0)
        self.assertEqual(row["devices"][1]["name"], "gpu-b")
        self.assertEqual(row["peak_vram_mb"], 30)


if __name__ == "__main__":
    unittest.main()
