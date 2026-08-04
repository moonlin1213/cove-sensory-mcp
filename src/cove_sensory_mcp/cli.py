from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .config.paths import AppPaths
from .config.schema import AppConfig, ProviderConfig
from .config.secrets import KeyringSecretStore
from .config.store import ConfigStore
from .errors import SensoryError
from .models import Modality
from .server import run_stdio
from .services import AppServices
from .tools.setup import sensory_self_test

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

_MINIMAX_BASE_URLS = {
    "cn": "https://api.minimaxi.com/v1",
    "global": "https://api.minimax.io/v1",
}
_SAFE_PROVIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_CANCEL_VALUES = frozenset({"cancel", "quit", "q"})


class _ConfigurationCancelled(Exception):
    """Stop the local wizard before either persistent store is changed."""


def _optional_environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _build_services() -> AppServices:
    paths = AppPaths.for_system(
        platform.system(),
        home=Path.home(),
        roaming=_optional_environment_path("APPDATA"),
        local=_optional_environment_path("LOCALAPPDATA"),
    )
    return AppServices(
        config_store=ConfigStore(paths.config_file),
        secret_store=KeyringSecretStore(),
    )


def _clean_answer(input_fn: InputFn, prompt: str) -> str:
    answer = input_fn(prompt).strip()
    if _is_cancelled(answer):
        raise _ConfigurationCancelled
    return answer


def _is_cancelled(value: str) -> bool:
    return value.lower() in _CANCEL_VALUES


def _validated_identifier(value: str) -> str:
    if _SAFE_PROVIDER_ID.fullmatch(value) is None:
        raise ValueError("invalid identifier")
    return value


def _validated_model(value: str) -> str:
    if not value or "\n" in value or "\r" in value:
        raise ValueError("invalid model")
    return value


def _validated_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _port = parsed.port
    except ValueError:
        raise ValueError("invalid endpoint") from None
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid endpoint")
    return value.rstrip("/")


def _declared_capabilities(value: str) -> dict[Modality, bool]:
    selected = {item.strip().lower() for item in value.split(",") if item.strip()}
    allowed = {modality.value for modality in Modality}
    if not selected or not selected <= allowed:
        raise ValueError("invalid capabilities")
    return {modality: modality.value in selected for modality in Modality}


def _provider_from_answers(
    provider_choice: str,
    input_fn: InputFn,
) -> tuple[str, str, ProviderConfig]:
    if provider_choice == "gemini":
        credential_ref = _clean_answer(input_fn, "Credential reference: ")
        model = _validated_model(_clean_answer(input_fn, "Gemini model: "))
        provider = ProviderConfig(
            adapter="gemini",
            model=model,
            credential_ref=credential_ref,
        )
        return "gemini", credential_ref, provider

    if provider_choice == "minimax-m3":
        credential_ref = _clean_answer(input_fn, "Credential reference: ")
        region = _clean_answer(input_fn, "MiniMax region [cn/global/custom]: ").lower()
        if region == "custom":
            base_url = _validated_base_url(_clean_answer(input_fn, "MiniMax base URL: "))
        elif region in _MINIMAX_BASE_URLS:
            base_url = _MINIMAX_BASE_URLS[region]
        else:
            raise ValueError("invalid region")
        model = _validated_model(_clean_answer(input_fn, "MiniMax model: "))
        provider = ProviderConfig(
            adapter="minimax-m3",
            base_url=base_url,
            model=model,
            credential_ref=credential_ref,
        )
        return "minimax-m3", credential_ref, provider

    if provider_choice == "custom":
        provider_id = _validated_identifier(_clean_answer(input_fn, "Provider identifier: "))
        credential_ref = _clean_answer(input_fn, "Credential reference: ")
        base_url = _validated_base_url(_clean_answer(input_fn, "HTTPS base URL: "))
        model = _validated_model(_clean_answer(input_fn, "Model: "))
        declared = _declared_capabilities(
            _clean_answer(
                input_fn,
                "Declared capabilities (comma-separated image, video_visual, "
                "video_audio, audio, music): ",
            )
        )
        provider = ProviderConfig(
            adapter="openai-compatible",
            base_url=base_url,
            model=model,
            credential_ref=credential_ref,
            declared_capabilities=declared,
        )
        return provider_id, credential_ref, provider

    raise ValueError("invalid provider")


def run_configure(
    services: AppServices,
    input_fn: InputFn,
    secret_input_fn: InputFn,
    output: OutputFn,
) -> int:
    """Collect one provider locally and persist only its credential reference."""
    output(
        "Capabilities: Gemini may understand image, video, audio, and music; "
        "MiniMax-M3 is an image/native-video eye by default; custom capabilities "
        "are declarations only and remain unverified."
    )
    output(
        "Privacy: enter API keys only in this local wizard. Keys are stored in the "
        "operating-system credential store and must never be pasted into chat."
    )
    try:
        config = services.config_store.load()
        provider_choice = _clean_answer(
            input_fn,
            "Provider [gemini/minimax-m3/custom/cancel]: ",
        ).lower()
        provider_id, credential_ref, provider = _provider_from_answers(
            provider_choice,
            input_fn,
        )
        if provider_id in config.providers or any(
            configured.credential_ref == credential_ref
            for configured in config.providers.values()
        ):
            output("Configuration was not saved: use a new provider and credential reference.")
            return 1
        try:
            occupied = services.secret_store.exists(credential_ref)
        except SensoryError:
            output("Configuration was not saved: local credential storage is unavailable.")
            return 1
        if occupied:
            output("Configuration was not saved: the local credential reference is occupied.")
            return 1
        secret = secret_input_fn("API key (local input, hidden): ")
    except (EOFError, KeyboardInterrupt, StopIteration, _ConfigurationCancelled):
        output("Configuration cancelled; nothing was saved.")
        return 1
    except (SensoryError, ValueError):
        output("Configuration was not saved: one or more settings are invalid.")
        return 1

    try:
        services.secret_store.set(credential_ref, secret)
    except SensoryError:
        output("Configuration was not saved: local credential storage is unavailable.")
        return 1

    updated = config.model_copy(deep=True)
    updated.providers[provider_id] = provider
    try:
        services.config_store.save(updated)
    except (OSError, SensoryError):
        try:
            services.secret_store.delete(credential_ref)
        except SensoryError:
            output(
                "Configuration failed and credential rollback also failed; remove the "
                "new entry from the operating-system credential store."
            )
            return 1
        output("Configuration was not saved; the newly stored credential was removed.")
        return 1

    output(f"Configured provider: {provider_id}")
    output("Credential: stored locally (value and reference are hidden).")
    try:
        _clean_answer(input_fn, "Configure another provider later? [y/N]: ")
    except (EOFError, KeyboardInterrupt, StopIteration, _ConfigurationCancelled):
        pass
    return 0


def _credential_available(services: AppServices, provider_id: str, provider: ProviderConfig) -> bool:
    try:
        services.secret_store.get(
            provider.credential_ref or provider_id,
            env_name=provider.api_key_env,
        )
    except SensoryError:
        return False
    return True


def run_status(services: AppServices, output: OutputFn) -> int:
    """Print local foundation status without exposing credential data."""
    try:
        config = services.config_store.load()
    except SensoryError:
        output("Configuration: invalid")
        output("Foundation provider perception: unavailable")
        return 1

    output("Configuration: readable")
    if not config.providers:
        output("Providers: none")
    for provider_id, provider in sorted(config.providers.items()):
        state = "available" if _credential_available(services, provider_id, provider) else "missing"
        output(f"Provider {provider_id}: credential {state}")
    output("Foundation provider perception: unavailable (no working provider adapter yet)")
    return 0


def _probe_cache_directory() -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="cove-sensory-mcp-doctor-") as root:
            probe = Path(root) / "cache-probe"
            probe.mkdir()
            probe.rmdir()
            return not probe.exists()
    except OSError:
        return False


def run_doctor(services: AppServices, output: OutputFn) -> int:
    """Run local-only configuration, credential, cache, and FFmpeg diagnostics."""
    healthy = True
    try:
        config: AppConfig | None = services.config_store.load()
    except SensoryError:
        config = None
        healthy = False
        output("Config: invalid")
    else:
        output("Config: ok")

    if config is None or not config.providers:
        output("Credentials: missing (no provider configured)")
        healthy = False
    else:
        for provider_id, provider in sorted(config.providers.items()):
            available = _credential_available(services, provider_id, provider)
            output(f"Credential for {provider_id}: {'ok' if available else 'missing'}")
            healthy = healthy and available

    cache_ok = _probe_cache_directory()
    output(f"Cache create/remove: {'ok' if cache_ok else 'failed'}")
    healthy = healthy and cache_ok

    ffmpeg_available = shutil.which("ffmpeg") is not None
    output(f"FFmpeg: {'available' if ffmpeg_available else 'missing'}")
    healthy = healthy and ffmpeg_available
    output("Provider network checks: skipped in the foundation milestone")
    return 0 if healthy else 1


def run_self_test(services: AppServices, output: OutputFn) -> int:
    """Report the local-only foundation verifier result without provider I/O."""
    result = asyncio.run(sensory_self_test(services, list(Modality)))
    output(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result.get("status") == "error" else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cove-sensory-mcp")
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the local MCP server over stdio.")
    subparsers.add_parser("configure", help="Configure one provider locally.")
    subparsers.add_parser("status", help="Show redacted local setup status.")
    subparsers.add_parser("doctor", help="Run local-only setup diagnostics.")
    subparsers.add_parser("self-test", help="Run the foundation capability self-test.")
    args = parser.parse_args(argv)
    if args.version:
        print(f"cove-sensory-mcp {__version__}")
        return 0
    if args.command == "serve":
        run_stdio(_build_services())
        return 0
    if args.command == "configure":
        return run_configure(
            _build_services(),
            input_fn=input,
            secret_input_fn=getpass.getpass,
            output=print,
        )
    if args.command == "status":
        return run_status(_build_services(), output=print)
    if args.command == "doctor":
        return run_doctor(_build_services(), output=print)
    if args.command == "self-test":
        return run_self_test(_build_services(), output=print)
    parser.print_help()
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
