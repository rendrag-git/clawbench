from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status_code: int | None
    body: str
    probe_name: str
    error: str | None


class Probe(Protocol):
    name: str

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult: ...


class LocalProbe:
    name = "host"

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
                return ProbeResult(
                    ok=True,
                    status_code=response.status,
                    body=body,
                    probe_name=self.name,
                    error=None,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return ProbeResult(
                ok=False,
                status_code=exc.code,
                body=body,
                probe_name=self.name,
                error=f"http_{exc.code}",
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ProbeResult(
                ok=False,
                status_code=None,
                body="",
                probe_name=self.name,
                error=str(exc),
            )
