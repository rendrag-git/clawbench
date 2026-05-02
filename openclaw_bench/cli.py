from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from .certification import certify_run_dirs, render_certification_text
from .backend import make_backend
from .providers import ProviderCandidate, derive_probes_for_profile, run_detection
from .container import DEFAULT_GATEWAY_PORT, DEFAULT_OPENCLAW_IMAGE, ensure_openclaw_container
from .manifest import load_model_manifest_scope, load_model_specs, load_suite
from .models import ModelSpec
from .preflight import PreflightCheck, check_openclaw_version, ensure_openclaw_gateway, render_text, run_preflight, run_verification_gates, stop_openclaw_gateway
from .quickstart import (
    DEFAULT_AGENT,
    DEFAULT_PROFILE,
    default_bench_root,
    init_quickstart,
    prompt_provider_selection,
    quickstart_run_args,
    start_benchclaw_gateway,
    stop_benchclaw_gateway,
)
from .runner import BenchmarkRunner, RunConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openclaw-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create an isolated benchclaw quickstart profile")
    _add_quickstart_init_args(init_parser)
    start_parser = subparsers.add_parser("start", help="Start only the benchclaw OpenClaw gateway")
    _add_quickstart_lifecycle_args(start_parser)
    stop_parser = subparsers.add_parser("stop", help="Stop only the benchclaw OpenClaw gateway")
    _add_quickstart_lifecycle_args(stop_parser)
    quickstart_parser = subparsers.add_parser("quickstart", help="Initialize benchclaw, preflight, and run a starter benchmark")
    _add_quickstart_init_args(quickstart_parser)
    quickstart_parser.add_argument("--run-id")
    quickstart_parser.add_argument("--timeout", type=int, default=300)
    quickstart_parser.add_argument("--smoke-timeout", type=int, default=60)
    quickstart_parser.add_argument("--openclaw-gateway-timeout", type=int, default=60)
    quickstart_parser.add_argument("--stop-after", action="store_true", help="Stop only the benchclaw gateway after the starter run")
    quickstart_parser.add_argument(
        "--backend",
        choices=["openclaw", "simulator"],
        default="openclaw",
        help="Use simulator only for harness validation; quickstart defaults to the benchclaw OpenClaw gateway",
    )
    run_parser = subparsers.add_parser("run", help="Run an OpenClaw benchmark suite")
    _add_common_run_args(run_parser)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--timeout", type=int, default=300)
    run_parser.add_argument(
        "--openclaw-smoke-timeout",
        type=int,
        default=60,
        help="Seconds to wait for the OpenClaw route smoke before running task attempts",
    )
    preflight_parser = subparsers.add_parser("preflight", help="Check whether a benchmark run can execute safely")
    _add_common_run_args(preflight_parser)
    preflight_parser.add_argument("--json", action="store_true", help="Print machine-readable preflight output")
    preflight_parser.add_argument("--smoke-turn", action="store_true", help="Run a tiny OpenClaw model route turn for eligible models")
    preflight_parser.add_argument(
        "--agent-smoke-turn",
        action="store_true",
        help="Run a tiny OpenClaw agent turn that mirrors benchmark agent routing for eligible models",
    )
    preflight_parser.add_argument("--smoke-timeout", type=int, default=60, help="Seconds to wait for each smoke turn")
    provider_preflight_parser = subparsers.add_parser(
        "provider-preflight",
        help="Run the four verification gates against an OpenClaw profile + provider route.",
    )
    provider_preflight_parser.add_argument("--profile", required=True)
    provider_preflight_parser.add_argument("--provider", required=True, choices=["vllm", "ollama", "llamacpp", "lmstudio"])
    provider_preflight_parser.add_argument("--base-url", required=True)
    provider_preflight_parser.add_argument("--route-model", required=True)
    provider_preflight_parser.add_argument("--container", default=None)
    provider_preflight_parser.add_argument("--timeout-s", type=int, default=60)
    provider_preflight_parser.set_defaults(handler=provider_preflight_command)
    certify_parser = subparsers.add_parser("certify", help="Audit benchmark result directories against the certification objective")
    certify_parser.add_argument("run_dirs", nargs="+", help="One or more benchmark result directories")
    certify_parser.add_argument("--json", action="store_true", help="Print machine-readable certification output")
    certify_parser.add_argument("--failures-only", action="store_true", help="Print only failing and warning certification checks in text output")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
        if args.command == "preflight":
            return preflight_command(args)
        if args.command == "certify":
            return certify_command(args)
        if args.command == "init":
            return init_command(args)
        if args.command == "start":
            return start_command(args)
        if args.command == "stop":
            return stop_command(args)
        if args.command == "quickstart":
            return quickstart_command(args)
        if args.command == "provider-preflight":
            return provider_preflight_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default=str(PROJECT_ROOT / "manifests" / "openclaw-agent-core.json"))
    parser.add_argument("--models", default="simulated-model")
    parser.add_argument("--model-config", help="JSON file with model/provider definitions")
    parser.add_argument("--kv", help="Comma-separated KV modes. Overrides model-config kv_modes when set.")
    parser.add_argument("--concurrency", help="Comma-separated concurrency levels. Overrides model-config concurrency when set.")
    parser.add_argument("--contexts", help="Comma-separated context limits. Overrides model-config contexts when set.")
    parser.add_argument("--out", default="/home/ubuntu/openclaw-bench/results")
    parser.add_argument("--workspace-root")
    parser.add_argument("--fixtures-root", default=str(PROJECT_ROOT / "fixtures"))
    parser.add_argument("--backend", choices=["simulator", "openclaw"], default="openclaw")
    parser.add_argument("--openclaw-profile", default="bench")
    parser.add_argument("--openclaw-agent", default="main")
    parser.add_argument("--openclaw-local", action="store_true")
    parser.add_argument("--openclaw-container", help="Run OpenClaw CLI commands via docker exec in this container")
    parser.add_argument(
        "--ensure-openclaw-container",
        dest="ensure_openclaw_container",
        action="store_true",
        default=True,
        help="Create/start --openclaw-container with the bench profile when it is missing or stopped",
    )
    parser.add_argument(
        "--no-ensure-openclaw-container",
        dest="ensure_openclaw_container",
        action="store_false",
        help="Do not create or start --openclaw-container before checking the gateway",
    )
    parser.add_argument("--openclaw-container-image", default=DEFAULT_OPENCLAW_IMAGE)
    parser.add_argument("--openclaw-container-home", help="Host directory mounted as /home/ubuntu in the bench container")
    parser.add_argument("--openclaw-container-gateway-port", type=int, default=DEFAULT_GATEWAY_PORT)
    parser.add_argument("--openclaw-container-token", help="Gateway token for a newly created bench container")
    parser.add_argument(
        "--openclaw-gateway-timeout",
        type=int,
        default=60,
        help="Seconds to wait while auto-starting the selected OpenClaw gateway",
    )
    parser.add_argument(
        "--ensure-openclaw-gateway",
        dest="ensure_openclaw_gateway",
        action="store_true",
        default=True,
        help="Start the selected OpenClaw gateway/profile when a non-local OpenClaw run needs it",
    )
    parser.add_argument(
        "--no-ensure-openclaw-gateway",
        dest="ensure_openclaw_gateway",
        action="store_false",
        help="Only check the selected OpenClaw gateway instead of auto-starting it",
    )
    parser.add_argument(
        "--openclaw-workspace-agents",
        dest="openclaw_workspace_agents",
        action="store_true",
        default=None,
        help="Create/use one configured OpenClaw agent per attempt so agent cwd/model come from config instead of per-call overrides",
    )
    parser.add_argument(
        "--no-openclaw-workspace-agents",
        dest="openclaw_workspace_agents",
        action="store_false",
        help="Disable configured benchmark agents; only valid for local OpenClaw runs",
    )
    parser.add_argument("--thinking")


def _add_quickstart_init_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--providers", choices=["local", "api", "both"], help="Provider mode. Omit for the interactive wizard.")
    parser.add_argument("--bench-root", default=str(default_bench_root()))
    parser.add_argument("--config-home", help=argparse.SUPPRESS)
    parser.add_argument("--openclaw-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--openclaw-agent", default=DEFAULT_AGENT)
    parser.add_argument("--gateway-port", type=int, help="Gateway port for the isolated benchclaw profile")
    parser.add_argument("--vllm-base-url", help="OpenAI-compatible vLLM API base URL for local-provider quickstart")
    parser.add_argument("--vllm-model", help="Served model name exposed by the vLLM API")
    parser.add_argument(
        "--vllm-context",
        type=int,
        default=32768,
        help="Benchmark context window to record for the external vLLM endpoint; generated OpenClaw routes use at least 16000 tokens",
    )
    parser.add_argument("--vllm-max-tokens", type=int, default=256, help="Max output tokens to configure for the vLLM route")
    parser.add_argument("--force", action="store_true", help="Overwrite the generated isolated benchclaw config")
    parser.add_argument("--no-validate", action="store_true", help="Write files without running openclaw config validate")
    parser.add_argument("--no-detect", action="store_true", help="Skip provider auto-detection; use --vllm-* flags or env vars.")
    parser.add_argument("--oc-runtime", default=None, help="Override OpenClaw runtime probe target (e.g., ssh:user@host).")


def _add_quickstart_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--openclaw-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--timeout", type=int, default=60)


def init_command(args: argparse.Namespace) -> int:
    providers = args.providers or prompt_provider_selection()
    home = Path(args.config_home) if getattr(args, "config_home", None) else None

    detected: ProviderCandidate | None = None
    if not getattr(args, "no_detect", False) and providers in {"local", "both"}:
        probe_home = home or Path.home()
        probes = derive_probes_for_profile(
            args.openclaw_profile,
            home=probe_home,
            oc_runtime_override=getattr(args, "oc_runtime", None),
        )
        report = run_detection(
            providers=["vllm", "ollama", "llamacpp", "lmstudio"],
            probes=probes,
            home=probe_home,
        )
        for finding in report.findings:
            print(f"finding: {finding}")
        if not report.candidates:
            print(
                "no local provider detected; pass --no-detect with --vllm-base-url/--vllm-model "
                "to specify one explicitly, or start a model server first"
            )
            return 2
        detected = next((c for c in report.candidates if c.provider == "vllm"), report.candidates[0])

    result = init_quickstart(
        providers=providers,
        project_root=PROJECT_ROOT,
        bench_root=Path(args.bench_root),
        home=home,
        profile=args.openclaw_profile,
        agent=args.openclaw_agent,
        port=args.gateway_port,
        force=args.force,
        validate=not args.no_validate,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_context=args.vllm_context,
        vllm_max_tokens=args.vllm_max_tokens,
        detected_candidate=detected,
    )
    print(f"profile={result.profile}")
    print(f"providers={result.providers}")
    print(f"gateway_port={result.port}")
    print(f"config={result.paths.config_path}")
    print(f"suite={result.paths.suite_path}")
    print(f"model_config={result.paths.model_config_path}")
    print(f"results={result.paths.results_root}")
    if result.providers in {"local", "both"}:
        print(f"vllm_base_url={result.vllm.base_url}")
        print(f"vllm_model={result.vllm.model}")
    if result.existing_profiles:
        print(f"existing_profiles={','.join(result.existing_profiles)}")
    if result.validation is not None:
        print(f"{result.validation.name}={result.validation.status} {result.validation.notes}")
    print("oauth=bring-your-own-auth; configure OAuth providers in the benchclaw profile before running them")
    return 0


def start_command(args: argparse.Namespace) -> int:
    check = start_benchclaw_gateway(args.openclaw_profile, timeout_s=args.timeout)
    print(f"{check.name}={check.status} {check.notes}")
    return 0 if check.status != "fail" else 1


def stop_command(args: argparse.Namespace) -> int:
    check = stop_benchclaw_gateway(args.openclaw_profile, timeout_s=args.timeout)
    print(f"{check.name}={check.status} {check.notes}")
    return 0 if check.status != "fail" else 1


def quickstart_command(args: argparse.Namespace) -> int:
    providers = args.providers or prompt_provider_selection()
    init_result = init_quickstart(
        providers=providers,
        project_root=PROJECT_ROOT,
        bench_root=Path(args.bench_root),
        home=Path(args.config_home) if getattr(args, "config_home", None) else None,
        profile=args.openclaw_profile,
        agent=args.openclaw_agent,
        port=args.gateway_port,
        force=args.force,
        reuse_existing=True,
        validate=not args.no_validate,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_context=args.vllm_context,
        vllm_max_tokens=args.vllm_max_tokens,
    )
    print(f"profile={init_result.profile}")
    print(f"config={init_result.paths.config_path}")
    print(f"model_config={init_result.paths.model_config_path}")

    started = False
    run_code = 1
    try:
        _prepare_quickstart_env(providers)
        if args.backend == "openclaw":
            start_check = start_benchclaw_gateway(args.openclaw_profile, timeout_s=args.openclaw_gateway_timeout)
            print(f"{start_check.name}={start_check.status} {start_check.notes}")
            if start_check.status == "fail":
                return 1
            started = True

        common = quickstart_run_args(init_result.paths, profile=args.openclaw_profile, agent=args.openclaw_agent)
        preflight_args = _quickstart_preflight_namespace(args, common, providers)
        preflight_code = preflight_command(preflight_args)
        if preflight_code != 0:
            return preflight_code

        run_args = _quickstart_run_namespace(args, common)
        run_code = run_command(run_args)
        out_dir = Path(common["out"]) / run_args.run_id
        print(f"result_path={out_dir}")
        return run_code
    finally:
        if args.stop_after and started:
            stop_check = stop_benchclaw_gateway(args.openclaw_profile, timeout_s=30)
            print(f"{stop_check.name}={stop_check.status} {stop_check.notes}")


def _quickstart_preflight_namespace(args: argparse.Namespace, common: dict[str, str], providers: str) -> argparse.Namespace:
    model_config = None if args.backend == "simulator" else common["model_config"]
    return argparse.Namespace(
        suite=common["suite"],
        models="simulated-model",
        model_config=model_config,
        kv=None,
        concurrency=None,
        contexts=None,
        out=common["out"],
        workspace_root=common["workspace_root"],
        fixtures_root=common["fixtures_root"],
        backend=args.backend,
        openclaw_profile=common["openclaw_profile"],
        openclaw_agent=common["openclaw_agent"],
        openclaw_local=False,
        openclaw_container=None,
        ensure_openclaw_container=False,
        openclaw_container_image=DEFAULT_OPENCLAW_IMAGE,
        openclaw_container_home=None,
        openclaw_container_gateway_port=DEFAULT_GATEWAY_PORT,
        openclaw_container_token=None,
        openclaw_gateway_timeout=args.openclaw_gateway_timeout,
        ensure_openclaw_gateway=False,
        openclaw_workspace_agents=True,
        json=False,
        smoke_turn=providers in {"api", "both"} and args.backend == "openclaw",
        agent_smoke_turn=False,
        smoke_timeout=args.smoke_timeout,
        thinking=None,
    )


def _prepare_quickstart_env(providers: str) -> None:
    if providers in {"local", "both"}:
        os.environ.setdefault("VLLM_API_KEY", "vllm-local")


def _quickstart_run_namespace(args: argparse.Namespace, common: dict[str, str]) -> argparse.Namespace:
    model_config = None if args.backend == "simulator" else common["model_config"]
    return argparse.Namespace(
        suite=common["suite"],
        models="simulated-model",
        model_config=model_config,
        kv=None,
        concurrency="1",
        contexts=None,
        out=common["out"],
        workspace_root=common["workspace_root"],
        fixtures_root=common["fixtures_root"],
        backend=args.backend,
        openclaw_profile=common["openclaw_profile"],
        openclaw_agent=common["openclaw_agent"],
        openclaw_local=False,
        openclaw_container=None,
        ensure_openclaw_container=False,
        openclaw_container_image=DEFAULT_OPENCLAW_IMAGE,
        openclaw_container_home=None,
        openclaw_container_gateway_port=DEFAULT_GATEWAY_PORT,
        openclaw_container_token=None,
        openclaw_gateway_timeout=args.openclaw_gateway_timeout,
        ensure_openclaw_gateway=False,
        openclaw_workspace_agents=True,
        thinking=None,
        timeout=args.timeout,
        openclaw_smoke_timeout=args.smoke_timeout,
        run_id=args.run_id or datetime.now(timezone.utc).strftime("quickstart-%Y%m%dT%H%M%SZ"),
    )


def run_command(args: argparse.Namespace) -> int:
    _validate_run_args(args)
    suite, models, kv_modes, contexts, concurrencies = _load_run_inputs(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / run_id if Path(args.out).name != run_id else Path(args.out)
    bench_root = out_dir.parent.parent if out_dir.parent.name == "results" else out_dir.parent
    workspace_root = Path(args.workspace_root) if args.workspace_root else bench_root / "workspaces" / run_id
    _ensure_container_for_run(args, models, bench_root, workspace_root, emit=True, raise_on_fail=True)
    gateway_ensure = _ensure_gateway_for_run(args)
    effective_gateway_ensure = args.backend == "openclaw" and not args.openclaw_local and args.ensure_openclaw_gateway
    effective_workspace_agents = _workspace_agents_enabled(args)
    config = RunConfig(
        run_id=run_id,
        suite=suite,
        models=models,
        kv_modes=kv_modes,
        contexts=contexts,
        concurrencies=concurrencies,
        out_dir=out_dir,
        workspace_root=workspace_root,
        fixtures_root=Path(args.fixtures_root),
        backend_name=args.backend,
        suite_path=Path(args.suite),
        model_config_path=Path(args.model_config) if args.model_config else None,
        openclaw_profile=args.openclaw_profile,
        openclaw_agent=args.openclaw_agent,
        openclaw_local=args.openclaw_local,
        openclaw_container=args.openclaw_container,
        ensure_openclaw_gateway=effective_gateway_ensure,
        openclaw_gateway_ensure=gateway_ensure,
        openclaw_gateway_timeout_s=args.openclaw_gateway_timeout,
        openclaw_workspace_agents=effective_workspace_agents,
        thinking=args.thinking,
        timeout_s=args.timeout,
        openclaw_smoke_timeout_s=args.openclaw_smoke_timeout,
    )
    backend = make_backend(
        args.backend,
        profile=args.openclaw_profile,
        agent=args.openclaw_agent,
        local=args.openclaw_local,
        thinking=args.thinking,
        workspace_agents=effective_workspace_agents,
        container=args.openclaw_container,
    )
    runner = BenchmarkRunner(backend)
    stop_started_gateway = _started_local_foreground_gateway(args, gateway_ensure)
    try:
        results = runner.run(config)
    finally:
        if stop_started_gateway:
            stop_check = stop_openclaw_gateway(args.openclaw_profile, timeout_s=args.openclaw_gateway_timeout)
            print(f"{stop_check.name}={stop_check.status} {stop_check.notes}")
    failures = sum(1 for result in results if result.status != "pass")
    print(f"run_id={run_id}")
    print(f"out={out_dir}")
    print(f"attempts={len(results)}")
    print(f"failures={failures}")
    return 0 if failures == 0 else 1


def _validate_run_args(args: argparse.Namespace) -> None:
    if args.backend == "openclaw" and not args.openclaw_local and not _workspace_agents_enabled(args):
        raise ValueError(
            "gateway OpenClaw benchmark runs require --openclaw-workspace-agents so the model is bound on the agent "
            "instead of sent as an unauthorized per-call --model override; use --openclaw-local for embedded runs"
        )


def _workspace_agents_enabled(args: argparse.Namespace) -> bool:
    requested = getattr(args, "openclaw_workspace_agents", None)
    if requested is not None:
        return bool(requested)
    return args.backend == "openclaw" and not args.openclaw_local


def _ensure_gateway_for_run(args: argparse.Namespace) -> dict[str, str] | None:
    if args.backend != "openclaw" or args.openclaw_local or not args.ensure_openclaw_gateway:
        return None
    version = check_openclaw_version(args.openclaw_container)
    if version.status == "fail":
        raise ValueError(version.notes)
    check = ensure_openclaw_gateway(args.openclaw_profile, args.openclaw_container, timeout_s=args.openclaw_gateway_timeout)
    if check.status == "fail":
        raise ValueError(f"OpenClaw gateway is not ready: {check.notes}")
    print(f"{check.name}={check.status} {check.notes}")
    return check.to_row()


def _started_local_foreground_gateway(args: argparse.Namespace, gateway_ensure: dict[str, str] | None) -> bool:
    if args.backend != "openclaw" or args.openclaw_local or args.openclaw_container:
        return False
    if not gateway_ensure:
        return False
    notes = gateway_ensure.get("notes", "")
    return "started bench gateway" in notes


def preflight_command(args: argparse.Namespace) -> int:
    suite, models, _, _, _ = _load_run_inputs(args)
    out_dir = Path(args.out)
    bench_root = out_dir.parent if out_dir.name == "results" else out_dir
    workspace_root = Path(args.workspace_root) if args.workspace_root else bench_root / "workspaces" / "preflight"
    container_ensure = _ensure_container_for_run(args, models, bench_root, workspace_root, emit=False, raise_on_fail=False)
    result = run_preflight(
        suite=suite,
        models=models,
        backend_name=args.backend,
        out_dir=out_dir,
        workspace_root=workspace_root,
        fixtures_root=Path(args.fixtures_root),
        openclaw_profile=args.openclaw_profile,
        openclaw_agent=args.openclaw_agent,
        openclaw_local=args.openclaw_local,
        openclaw_container=args.openclaw_container,
        ensure_gateway=args.ensure_openclaw_gateway,
        gateway_timeout_s=args.openclaw_gateway_timeout,
        openclaw_workspace_agents=_workspace_agents_enabled(args),
        smoke_turn=args.smoke_turn,
        agent_smoke_turn=args.agent_smoke_turn,
        smoke_timeout_s=args.smoke_timeout,
    )
    if container_ensure is not None:
        result.checks.insert(0, container_ensure)
    portability_check = _model_config_portability_check(args)
    if portability_check is not None:
        result.checks.insert(0, portability_check)
    print(result.to_json() if args.json else render_text(result))
    return 0 if result.ok else 1


def provider_preflight_command(args: argparse.Namespace) -> int:
    report = run_verification_gates(
        profile=args.profile,
        provider=args.provider,
        base_url=args.base_url,
        route_model=args.route_model,
        container=args.container,
        timeout_s=args.timeout_s,
    )
    for check in report.checks:
        status = "PASS" if check.status == "pass" else "FAIL"
        print(f"{status}\t{check.name}\t{check.notes}")
    return 0 if report.ok else 1


def certify_command(args: argparse.Namespace) -> int:
    result = certify_run_dirs([Path(path) for path in args.run_dirs])
    if args.json:
        print(result.to_json())
    else:
        print(render_certification_text(result, failures_only=args.failures_only))
    return 0 if result.ok else 1


def _load_run_inputs(args: argparse.Namespace) -> tuple:
    suite = load_suite(Path(args.suite))
    model_specs = load_model_specs(Path(args.model_config)) if args.model_config else []
    kv_modes = _parse_csv(args.kv) if args.kv else _default_kv_modes(model_specs)
    contexts = [int(item) for item in _parse_csv(args.contexts)] if args.contexts else _default_contexts(model_specs)
    concurrencies = [int(item) for item in _parse_csv(args.concurrency)] if args.concurrency else _default_concurrency(model_specs)
    models = _build_models(args, model_specs, kv_modes, contexts)
    return suite, models, kv_modes, contexts, concurrencies


def _model_config_portability_check(args: argparse.Namespace) -> PreflightCheck | None:
    if not args.model_config:
        return None
    scope = load_model_manifest_scope(Path(args.model_config))
    if scope.get("portability") != "host_specific":
        return None
    notes = scope.get("notes")
    if not isinstance(notes, str) or not notes:
        notes = "model config is marked host_specific; copy and adapt before using on another host"
    return PreflightCheck("model_config_portability", "warn", notes)


def _ensure_container_for_run(
    args: argparse.Namespace,
    models: list[ModelSpec],
    bench_root: Path,
    workspace_root: Path,
    *,
    emit: bool,
    raise_on_fail: bool,
):
    if args.backend != "openclaw" or not args.openclaw_container or not args.ensure_openclaw_container:
        return None
    check = ensure_openclaw_container(
        container=args.openclaw_container,
        image=args.openclaw_container_image,
        profile=args.openclaw_profile,
        project_root=PROJECT_ROOT,
        bench_root=bench_root,
        workspace_root=workspace_root,
        container_home=Path(args.openclaw_container_home) if args.openclaw_container_home else None,
        gateway_port=args.openclaw_container_gateway_port,
        gateway_token=args.openclaw_container_token,
        models=models,
        timeout_s=args.openclaw_gateway_timeout,
    )
    if check.status == "fail":
        if raise_on_fail:
            raise ValueError(f"OpenClaw container is not ready: {check.notes}")
        return check
    if emit:
        print(f"{check.name}={check.status} {check.notes}")
    return check


def _build_models(args: argparse.Namespace, model_specs: list[dict], kv_modes: list[str], contexts: list[int]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    if args.model_config:
        for spec in model_specs:
            if _override_changes_local_serve_cell(args, spec, kv_modes, contexts):
                raise ValueError(
                    "--kv/--contexts overrides are not safe for local model-config entries with serve_command; "
                    "use a manifest whose served_model_name, openclaw_model_name, serve_command, and --max-model-len match the desired cell"
                )
            spec_kv_modes = kv_modes if args.kv else spec.get("kv_modes", kv_modes)
            spec_contexts = contexts if args.contexts else spec.get("contexts", contexts)
            support = spec.get("kv_support", {})
            model_data = {
                key: value
                for key, value in spec.items()
                if key not in {"kv_modes", "contexts", "concurrency", "kv_support"}
            }
            for kv in spec_kv_modes:
                for context in spec_contexts:
                    status = support.get(kv, spec.get("support_status", "unknown"))
                    models.append(
                        ModelSpec.from_mapping(
                            {**model_data, "support_status": status},
                            kv_cache_dtype=kv,
                            context_limit=int(context),
                        )
                    )
        return models
    for alias in _parse_csv(args.models):
        for kv in kv_modes:
            for context in contexts:
                models.append(ModelSpec.from_alias(alias, kv_cache_dtype=kv, context_limit=context))
    return models


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _override_changes_local_serve_cell(args: argparse.Namespace, spec: dict, kv_modes: list[str], contexts: list[int]) -> bool:
    if spec.get("provider_type", "local") != "local" or not spec.get("serve_command"):
        return False
    if args.kv:
        declared_modes = {str(mode) for mode in spec.get("kv_modes", [])}
        command_kv = _serve_arg_value(spec.get("serve_command", []), "--kv-cache-dtype")
        if command_kv:
            declared_modes.add(command_kv)
        if declared_modes and set(kv_modes) != declared_modes:
            return True
    if args.contexts:
        declared_contexts = {int(context) for context in spec.get("contexts", [])}
        max_model_len = _serve_arg_int(spec.get("serve_command", []), "--max-model-len")
        if max_model_len is not None:
            declared_contexts.add(max_model_len)
        if declared_contexts and not set(contexts) <= declared_contexts:
            return True
    return False


def _serve_arg_value(command: list, flag: str) -> str | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return str(command[index + 1])
        if isinstance(item, str) and item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _serve_arg_int(command: list, flag: str) -> int | None:
    value = _serve_arg_value(command, flag)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _default_kv_modes(model_specs: list[dict]) -> list[str]:
    if not model_specs:
        return ["fp8"]
    modes: list[str] = []
    for spec in model_specs:
        for mode in spec.get("kv_modes", ["fp8"]):
            if mode not in modes:
                modes.append(mode)
    return modes


def _default_contexts(model_specs: list[dict]) -> list[int]:
    if not model_specs:
        return [4096]
    contexts: list[int] = []
    for spec in model_specs:
        for context in spec.get("contexts", [4096]):
            numeric = int(context)
            if numeric not in contexts:
                contexts.append(numeric)
    return contexts


def _default_concurrency(model_specs: list[dict]) -> list[int]:
    if not model_specs:
        return [1]
    levels: list[int] = []
    for spec in model_specs:
        for level in spec.get("concurrency", [1]):
            numeric = int(level)
            if numeric not in levels:
                levels.append(numeric)
    return levels
