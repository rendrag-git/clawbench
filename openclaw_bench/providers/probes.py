from __future__ import annotations

import subprocess
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

    def http_get(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> ProbeResult: ...


class LocalProbe:
    name = "host"

    def http_get(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> ProbeResult:
        request = urllib.request.Request(url, method="GET")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
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


def _curl_args(
    url: str,
    timeout_s: float,
    *,
    headers: dict[str, str] | None = None,
) -> list[str]:
    # Use a leading newline in --write-out so the status marker always starts
    # on its own line, even when the response body does not end with a newline.
    args = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        str(int(max(1, round(timeout_s)))),
        "--write-out",
        "\\nHTTP_STATUS:%{http_code}\\n",
    ]
    for key, value in (headers or {}).items():
        args += ["-H", f"{key}: {value}"]
    args.append(url)
    return args


def _parse_curl_stdout(stdout: str) -> tuple[int | None, str]:
    status: int | None = None
    body_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("HTTP_STATUS:"):
            try:
                status = int(line.removeprefix("HTTP_STATUS:").strip())
            except ValueError:
                status = None
            continue
        body_lines.append(line)
    return status, "\n".join(body_lines)


def _result_from_curl(name: str, completed: subprocess.CompletedProcess) -> ProbeResult:
    if completed.returncode != 0:
        return ProbeResult(
            ok=False,
            status_code=None,
            body="",
            probe_name=name,
            error=(completed.stderr or completed.stdout or f"curl exit {completed.returncode}").strip(),
        )
    status, body = _parse_curl_stdout(completed.stdout)
    ok = status is not None and 200 <= status < 400
    return ProbeResult(
        ok=ok,
        status_code=status,
        body=body,
        probe_name=name,
        error=None if ok else f"http_{status}" if status else "missing_status_line",
    )


class IncusExecProbe:
    def __init__(self, instance: str) -> None:
        self.instance = instance
        self.name = f"incus:{instance}"

    def http_get(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> ProbeResult:
        cmd = ["incus", "exec", self.instance, "--"] + _curl_args(
            url, timeout_s, headers=headers
        )
        return self._run(cmd, timeout_s)

    def _run(self, cmd: list[str], timeout_s: float) -> ProbeResult:
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)


class DockerExecProbe:
    def __init__(self, container: str) -> None:
        self.container = container
        self.name = f"docker:{container}"

    def http_get(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> ProbeResult:
        cmd = ["docker", "exec", self.container] + _curl_args(
            url, timeout_s, headers=headers
        )
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)


class SSHProbe:
    def __init__(self, target: str) -> None:
        self.target = target
        self.name = f"ssh:{target}"

    def http_get(
        self,
        url: str,
        *,
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> ProbeResult:
        cmd = ["ssh", "-o", "BatchMode=yes", self.target] + _curl_args(
            url, timeout_s, headers=headers
        )
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)
