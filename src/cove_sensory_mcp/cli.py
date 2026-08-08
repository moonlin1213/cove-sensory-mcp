from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import TypeAdapter

from . import __version__
from .config.paths import AppPaths
from .config.schema import AdapterOptions, AppConfig, ProviderConfig
from .config.secrets import KeyringSecretStore
from .config.store import ConfigStore
from .errors import ErrorCode, SensoryError
from .models import Modality, ProviderId, ProviderRef, RouteConfig
from .providers.openai_compatible import media_part_mode_capabilities
from .server import run_stdio
from .services import AppServices
from .tools.setup import sensory_self_test, verify_provider_capabilities

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
ConfigureVerifyFn = Callable[
    [AppServices, str, list[Modality]],
    Awaitable[dict[str, object]],
]

_MINIMAX_BASE_URLS = {
    "cn": "https://api.minimaxi.com",
    "global": "https://api.minimax.io",
}
CustomMediaPartMode = Literal[
    "image_url_data_uri",
    "input_audio_base64",
    "video_url_data_uri",
    "anthropic_base64_media",
]
_PROVIDER_ID_ADAPTER = TypeAdapter(ProviderId)
_CANCEL_VALUES = frozenset({"cancel", "quit", "q"})
_CONFIG_CHANGED_MESSAGE = "The configuration changed before the update completed."


class _ConfigurationCancelled(Exception):
    """Stop the local wizard before either persistent store is changed."""


class _CustomWireModeMismatch(ValueError):
    """Require separate Provider IDs for incompatible custom wire contracts."""


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
        config_store=ConfigStore(paths.config_file, jobs_dir=paths.jobs_dir),
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
    return _PROVIDER_ID_ADAPTER.validate_python(value)


def _validated_model(value: str) -> str:
    if not value or "\n" in value or "\r" in value:
        raise ValueError("invalid model")
    return value


def _credential_source(value: str) -> tuple[str | None, str | None]:
    """Interpret env:NAME without changing the existing credential-prompt position."""
    if value.startswith("env:"):
        return None, value.removeprefix("env:")
    return value, None


def _declared_capabilities(value: str) -> dict[Modality, bool]:
    selected = {item.strip().lower() for item in value.split(",") if item.strip()}
    allowed = {modality.value for modality in Modality}
    if not selected or not selected <= allowed:
        raise ValueError("invalid capabilities")
    return {modality: modality.value in selected for modality in Modality}


def _default_capabilities(*modalities: Modality) -> dict[Modality, bool]:
    selected = frozenset(modalities)
    return {modality: modality in selected for modality in Modality}


def _provider_from_answers(
    provider_choice: str,
    input_fn: InputFn,
) -> tuple[str, ProviderConfig]:
    if provider_choice == "gemini":
        credential_ref, api_key_env = _credential_source(
            _clean_answer(input_fn, "Credential reference (or env:VARIABLE_NAME): ")
        )
        model = _validated_model(_clean_answer(input_fn, "Gemini model: "))
        provider = ProviderConfig(
            adapter="gemini",
            model=model,
            credential_ref=credential_ref,
            api_key_env=api_key_env,
            declared_capabilities=_default_capabilities(*Modality),
        )
        return "gemini", provider

    if provider_choice == "minimax-m3":
        credential_ref, api_key_env = _credential_source(
            _clean_answer(input_fn, "Credential reference (or env:VARIABLE_NAME): ")
        )
        region = _clean_answer(input_fn, "MiniMax region [cn/global/custom]: ").lower()
        if region == "custom":
            base_url = _clean_answer(input_fn, "MiniMax base URL: ")
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
            api_key_env=api_key_env,
            declared_capabilities=_default_capabilities(
                Modality.IMAGE,
                Modality.VIDEO_VISUAL,
            ),
        )
        return "minimax-m3", provider

    if provider_choice == "custom":
        provider_id = _validated_identifier(
            _clean_answer(input_fn, "Provider identifier: ")
        )
        credential_ref, api_key_env = _credential_source(
            _clean_answer(input_fn, "Credential reference (or env:VARIABLE_NAME): ")
        )
        base_url = _clean_answer(input_fn, "HTTPS base URL: ")
        endpoint_path = _clean_answer(input_fn, "Relative endpoint path: ")
        media_part_mode_value = _clean_answer(
            input_fn,
            "Media part mode [image_url_data_uri/input_audio_base64/"
            "video_url_data_uri/anthropic_base64_media]: ",
        ).lower()
        supported_capabilities = media_part_mode_capabilities(media_part_mode_value)
        if supported_capabilities is None:
            raise ValueError("invalid media part mode")
        media_part_mode = cast(CustomMediaPartMode, media_part_mode_value)
        model = _validated_model(_clean_answer(input_fn, "Model: "))
        declared = _declared_capabilities(
            _clean_answer(
                input_fn,
                "Declared capabilities (comma-separated image, video_visual, "
                "video_audio, audio, music): ",
            )
        )
        selected = frozenset(
            modality for modality, enabled in declared.items() if enabled
        )
        if not selected <= supported_capabilities:
            raise _CustomWireModeMismatch
        provider = ProviderConfig(
            adapter="openai-compatible",
            base_url=base_url,
            model=model,
            credential_ref=credential_ref,
            api_key_env=api_key_env,
            declared_capabilities=declared,
            adapter_options=AdapterOptions(
                endpoint_path=endpoint_path,
                media_part_mode=media_part_mode,
            ),
        )
        return provider_id, provider

    raise ValueError("invalid provider")


def _optional_yes(input_fn: InputFn, prompt: str) -> bool:
    """Read one explicit authorization, defaulting safely when scripted input ends."""
    try:
        answer = _clean_answer(input_fn, prompt).lower()
    except StopIteration:
        return False
    if answer in {"y", "yes"}:
        return True
    if answer in {"", "n", "no"}:
        return False
    raise ValueError("invalid yes/no answer")


def _role_modalities(*, eye: bool, ear: bool) -> list[Modality]:
    modalities: list[Modality] = []
    if eye:
        modalities.extend((Modality.IMAGE, Modality.VIDEO_VISUAL))
    if ear:
        modalities.extend((Modality.VIDEO_AUDIO, Modality.AUDIO, Modality.MUSIC))
    return modalities


def _write_verified_routes(
    services: AppServices,
    provider_id: str,
    selected_modalities: Sequence[Modality],
    input_fn: InputFn,
) -> None:
    """Collect authorizations, then merge routes in one short locked transaction."""
    snapshot = services.config_store.load()
    primary = snapshot.providers[provider_id]
    original_primary = primary.model_copy(deep=True)
    planned: dict[
        Modality,
        tuple[RouteConfig | None, dict[str, ProviderConfig], list[str]],
    ] = {}
    for modality in selected_modalities:
        if not primary.verified_capabilities.get(modality, False):
            continue
        eligible: dict[str, ProviderConfig] = {}
        authorized_ids: list[str] = []
        for fallback_id, fallback in sorted(snapshot.providers.items()):
            if fallback_id == provider_id or not fallback.verified_capabilities.get(
                modality,
                False,
            ):
                continue
            eligible[fallback_id] = fallback.model_copy(deep=True)
            authorized = _optional_yes(
                input_fn,
                f"Authorize {fallback_id} as cross-Provider fallback for "
                f"{modality.value}? [y/N]: ",
            )
            if authorized:
                authorized_ids.append(fallback_id)
        planned[modality] = (
            getattr(snapshot.routes, modality.value).model_copy(deep=True)
            if getattr(snapshot.routes, modality.value) is not None
            else None,
            eligible,
            authorized_ids,
        )
    if not planned:
        return

    def merge_routes(latest: AppConfig) -> None:
        current_primary = latest.providers.get(provider_id)
        if current_primary != original_primary:
            raise SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_CHANGED_MESSAGE)
        for modality, (original_route, eligible, authorized_ids) in planned.items():
            if getattr(latest.routes, modality.value) != original_route:
                raise SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_CHANGED_MESSAGE)
            if not current_primary.verified_capabilities.get(modality, False):
                raise SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_CHANGED_MESSAGE)
            fallbacks: list[ProviderRef] = []
            for fallback_id in authorized_ids:
                current_fallback = latest.providers.get(fallback_id)
                if current_fallback != eligible[
                    fallback_id
                ] or not current_fallback.verified_capabilities.get(modality, False):
                    raise SensoryError(
                        ErrorCode.CONFIG_INVALID, _CONFIG_CHANGED_MESSAGE
                    )
                fallbacks.append(ProviderRef(provider=fallback_id, authorized=True))
            setattr(
                latest.routes,
                modality.value,
                RouteConfig(primary=provider_id, fallbacks=fallbacks),
            )

    services.config_store.update(merge_routes)


def run_configure(
    services: AppServices,
    input_fn: InputFn,
    secret_input_fn: InputFn,
    output: OutputFn,
    verify_fn: ConfigureVerifyFn = verify_provider_capabilities,
) -> int:
    """Collect one provider locally and persist only its credential reference."""
    output(
        "Roles: Gemini can be both eye and ear for image, video, audio, and music. "
        "MiniMax-M3 is an image/native-video eye by default and is not an ear by "
        "default. Custom capabilities are "
        "declarations only until each selected modality passes verification."
    )
    output(
        "Privacy: use env:VARIABLE_NAME for an environment-only credential, or enter "
        "an API key for the local operating-system credential store. Never paste keys "
        "into chat."
    )
    try:
        config = services.config_store.load()
        provider_choice = _clean_answer(
            input_fn,
            "Provider [gemini/minimax-m3/custom/cancel]: ",
        ).lower()
        provider_id, provider = _provider_from_answers(
            provider_choice,
            input_fn,
        )
        credential_ref = provider.credential_ref
        if provider_id in config.providers or (
            credential_ref is not None
            and any(
                configured.credential_ref == credential_ref
                for configured in config.providers.values()
            )
        ):
            output(
                "Configuration was not saved: use a new provider and credential reference."
            )
            return 1
        secret: str | None = None
        if credential_ref is not None:
            try:
                occupied = services.secret_store.exists(credential_ref)
            except SensoryError:
                output(
                    "Configuration was not saved: local credential storage is unavailable."
                )
                return 1
            if occupied:
                output(
                    "Configuration was not saved: the local credential reference is occupied."
                )
                return 1
            secret = secret_input_fn("API key (local input, hidden): ")
        eye = _optional_yes(input_fn, "Use this Provider as the eye? [y/N]: ")
        ear = False
        if provider.adapter != "minimax-m3":
            ear = _optional_yes(input_fn, "Use this Provider as the ear? [y/N]: ")
    except (EOFError, KeyboardInterrupt, StopIteration, _ConfigurationCancelled):
        output("Configuration cancelled; nothing was saved.")
        return 1
    except _CustomWireModeMismatch:
        output(
            "Configuration was not saved: one custom wire mode cannot carry every "
            "declared capability. Create multiple Provider IDs for incompatible wire "
            "modes."
        )
        return 1
    except (SensoryError, ValueError):
        output("Configuration was not saved: one or more settings are invalid.")
        return 1

    stored_credential_ref: str | None = None
    if credential_ref is not None and secret is not None:
        try:
            services.secret_store.set(credential_ref, secret)
        except SensoryError:
            output(
                "Configuration was not saved: local credential storage is unavailable."
            )
            return 1
        stored_credential_ref = credential_ref

    try:

        def add_provider(latest: AppConfig) -> None:
            if provider_id in latest.providers or (
                credential_ref is not None
                and any(
                    configured.credential_ref == credential_ref
                    for configured in latest.providers.values()
                )
            ):
                raise SensoryError(ErrorCode.CONFIG_INVALID, _CONFIG_CHANGED_MESSAGE)
            latest.providers[provider_id] = provider

        services.config_store.update(add_provider)
    except (OSError, SensoryError):
        if stored_credential_ref is None:
            output("Configuration was not saved.")
            return 1
        try:
            services.secret_store.delete(stored_credential_ref)
        except SensoryError:
            output(
                "Configuration failed and credential rollback also failed; remove the "
                "new entry from the operating-system credential store."
            )
            return 1
        output("Configuration was not saved; the newly stored credential was removed.")
        return 1

    output(f"Configured provider: {provider_id}")
    if provider.api_key_env is not None:
        state = (
            "available"
            if _credential_available(services, provider_id, provider)
            else "missing"
        )
        output(f"Credential: environment variable {state} (name and value are hidden).")
    else:
        output("Credential: stored locally (value and reference are hidden).")

    selected_modalities = [
        modality
        for modality in _role_modalities(eye=eye, ear=ear)
        if provider.declared_capabilities.get(modality, False)
    ]
    if selected_modalities:
        output(
            "Verification notice: this sends tiny test media to the selected Provider "
            "and may use a small amount of Provider quota."
        )
        try:
            verify_now = _optional_yes(
                input_fn,
                "Run capability verification now? [y/N]: ",
            )
        except (EOFError, KeyboardInterrupt, _ConfigurationCancelled, ValueError):
            output(
                "Verification was not started; the Provider remains saved without new routes."
            )
            return 0
        if verify_now:

            async def run_verification() -> dict[str, object]:
                return await verify_fn(services, provider_id, selected_modalities)

            result: dict[str, object] = asyncio.run(run_verification())
            if result.get("status") == "error":
                output("Verification did not pass; no unverified route was written.")
            try:
                _write_verified_routes(
                    services,
                    provider_id,
                    selected_modalities,
                    input_fn,
                )
            except (
                EOFError,
                KeyboardInterrupt,
                SensoryError,
                _ConfigurationCancelled,
                ValueError,
            ):
                output(
                    "Fallback authorization stopped; no implicit fallback was added."
                )
    return 0


def _credential_available(
    services: AppServices, provider_id: str, provider: ProviderConfig
) -> bool:
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
        output("Verified sensory routes: unavailable")
        return 1

    output("Configuration: readable")
    if not config.providers:
        output("Providers: none")
    for provider_id, provider in sorted(config.providers.items()):
        state = (
            "available"
            if _credential_available(services, provider_id, provider)
            else "missing"
        )
        output(f"Provider {provider_id}: credential {state}")
    verified_routes = 0
    for modality in Modality:
        route = getattr(config.routes, modality.value)
        if route is None:
            continue
        route_provider = config.providers.get(route.primary)
        if route_provider is None or not route_provider.verified_capabilities.get(
            modality, False
        ):
            continue
        output(f"Verified {modality.value}: {route.primary}")
        verified_routes += 1
    if verified_routes == 0:
        output("Verified sensory routes: none")
    return 0


def _probe_jobs_directory(jobs_dir: Path) -> bool:
    """Create and remove only one unique child of the configured jobs root."""
    probe: Path | None = None
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="cove-sensory-mcp-doctor-",
            dir=jobs_dir,
        ) as root:
            probe = Path(root)
            if probe.parent != jobs_dir or not probe.is_dir():
                return False
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

    cache_ok = _probe_jobs_directory(services.config_store.jobs_dir)
    output(f"Cache create/remove: {'ok' if cache_ok else 'failed'}")
    healthy = healthy and cache_ok

    ffmpeg_available = shutil.which("ffmpeg") is not None
    output(f"FFmpeg: {'available' if ffmpeg_available else 'missing'}")
    healthy = healthy and ffmpeg_available
    output("Provider network checks: skipped in the foundation milestone")
    return 0 if healthy else 1


def run_self_test(
    services: AppServices,
    output: OutputFn,
    input_fn: InputFn = input,
    *,
    yes: bool = False,
) -> int:
    """Run a quota-bearing Provider self-test only after explicit confirmation."""
    output(
        "Usage notice: this sends tiny test media to configured Providers and may use "
        "a small amount of Provider quota."
    )
    if not yes:
        try:
            confirmed = (
                input_fn("Continue with the sensory self-test? [y/N]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            output("Self-test cancelled; no Provider call was made.")
            return 1
        if confirmed not in {"y", "yes"}:
            output("Self-test declined; no Provider call was made.")
            return 1
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
    self_test_parser = subparsers.add_parser(
        "self-test",
        help="Run the Provider capability self-test.",
    )
    self_test_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm Provider quota use without an interactive prompt.",
    )
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
        return run_self_test(
            _build_services(),
            output=print,
            input_fn=input,
            yes=args.yes,
        )
    parser.print_help()
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
