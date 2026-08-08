import subprocess
import sys
import tempfile
from importlib.metadata import requires, version
from pathlib import Path

import keyring
import pytest

from cove_sensory_mcp import __version__, cli
from cove_sensory_mcp.cli import (
    main,
    run_configure,
    run_doctor,
    run_self_test,
    run_status,
)
from cove_sensory_mcp.config.paths import AppPaths
from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.errors import ErrorCode, SensoryError
from cove_sensory_mcp.models import Modality, RouteConfig
from cove_sensory_mcp.services import AppServices


def test_self_test_refusal_makes_no_provider_call_or_config_write(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quota-bearing self-test requires confirmation before any Provider work."""
    messages: list[str] = []
    called = False

    async def unexpected_self_test(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "ok", "results": []}

    def refuse_after_notice(prompt: str) -> str:
        notice = " ".join(messages).lower()
        assert "tiny test media" in notice
        assert "provider quota" in notice
        assert "continue" in prompt.lower()
        return "no"

    monkeypatch.setattr(cli, "sensory_self_test", unexpected_self_test)

    assert (
        run_self_test(
            tmp_services,
            output=messages.append,
            input_fn=refuse_after_notice,
            yes=False,
        )
        == 1
    )
    assert called is False
    assert not tmp_services.config_store.path.exists()


def test_self_test_yes_is_explicit_noninteractive_confirmation(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The --yes path must print the quota notice but never block for stdin."""
    messages: list[str] = []
    calls = 0

    async def successful_self_test(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "ok", "results": []}

    monkeypatch.setattr(cli, "sensory_self_test", successful_self_test)

    assert (
        run_self_test(
            tmp_services,
            output=messages.append,
            input_fn=lambda _: pytest.fail("--yes prompted for confirmation"),
            yes=True,
        )
        == 0
    )
    assert calls == 1
    report = " ".join(messages).lower()
    assert "tiny test media" in report
    assert "provider quota" in report


def test_main_self_test_yes_reaches_noninteractive_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing to register --yes would make supported automation impossible."""
    captured: dict[str, object] = {}

    def capture_runner(
        services: AppServices,
        output,
        input_fn,
        *,
        yes: bool,
    ) -> int:
        captured.update(services=services, output=output, input_fn=input_fn, yes=yes)
        return 0

    monkeypatch.setattr(cli, "_build_services", lambda: object())
    monkeypatch.setattr(cli, "run_self_test", capture_runner)

    assert main(["self-test", "--yes"]) == 0
    assert captured["yes"] is True


def test_configure_eye_and_ear_steps_write_only_capabilities_verified_in_batch(
    tmp_services: AppServices,
) -> None:
    """Role selection must not turn a merely declared Provider into a working route."""
    answers = iter(
        [
            "gemini",
            "gemini-main",
            "gemini-test-model",
            "yes",  # eye
            "yes",  # ear
            "yes",  # tiny-media verification
        ]
    )
    messages: list[str] = []

    async def verify_selected(
        services: AppServices,
        provider_id: str,
        modalities: list[Modality],
    ) -> dict[str, object]:
        assert provider_id == "gemini"
        assert modalities == list(Modality)
        config = services.config_store.load()
        provider = config.providers[provider_id]
        provider.verified_capabilities = {
            Modality.IMAGE: True,
            Modality.AUDIO: True,
        }
        services.config_store.save(config)
        return {"status": "partial", "results": []}

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "local-gemini-secret",
            output=messages.append,
            verify_fn=verify_selected,
        )
        == 0
    )

    config = tmp_services.config_store.load()
    assert config.routes.image is not None
    assert config.routes.image.primary == "gemini"
    assert config.routes.audio is not None
    assert config.routes.audio.primary == "gemini"
    assert config.routes.video_visual is None
    assert config.routes.video_audio is None
    assert config.routes.music is None
    notice = " ".join(messages).lower()
    assert "gemini" in notice and "eye" in notice and "ear" in notice
    assert "minimax-m3" in notice and "native-video eye" in notice
    assert "not" in notice and "ear" in notice


@pytest.mark.parametrize(
    ("authorization", "expected_fallbacks"), [("no", []), ("yes", ["ear-two"])]
)
def test_configure_cross_provider_fallback_requires_explicit_authorization(
    tmp_services: AppServices,
    authorization: str,
    expected_fallbacks: list[str],
) -> None:
    """A second capable Provider must never become a fallback through inference."""
    existing = ProviderConfig(
        adapter="gemini",
        model="ear-model",
        credential_ref="ear-two-ref",
        declared_capabilities={Modality.AUDIO: True},
        verified_capabilities={Modality.AUDIO: True},
    )
    tmp_services.config_store.save(AppConfig(providers={"ear-two": existing}))
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    tmp_services.secret_store.set("ear-two-ref", "existing-ear-secret")
    answers = iter(
        [
            "gemini",
            "gemini-main",
            "gemini-test-model",
            "no",  # eye
            "yes",  # ear
            "yes",  # verification
            authorization,  # ear-two fallback for audio
        ]
    )

    async def verify_audio_only(
        services: AppServices,
        provider_id: str,
        modalities: list[Modality],
    ) -> dict[str, object]:
        assert modalities == [Modality.VIDEO_AUDIO, Modality.AUDIO, Modality.MUSIC]
        config = services.config_store.load()
        config.providers[provider_id].verified_capabilities = {Modality.AUDIO: True}
        services.config_store.save(config)
        return {"status": "partial", "results": []}

    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    assert (
        run_configure(
            tmp_services,
            input_fn=answer,
            secret_input_fn=lambda _: "new-gemini-secret",
            output=lambda _: None,
            verify_fn=verify_audio_only,
        )
        == 0
    )

    route = tmp_services.config_store.load().routes.audio
    assert route is not None
    assert [fallback.provider for fallback in route.fallbacks] == expected_fallbacks
    assert all(fallback.authorized is True for fallback in route.fallbacks)
    fallback_prompts = [prompt for prompt in prompts if "fallback" in prompt.lower()]
    assert len(fallback_prompts) == 1
    assert "audio" in fallback_prompts[0].lower()


def test_route_prompt_collects_decision_before_short_update_and_rejects_late_conflict(
    tmp_services: AppServices,
) -> None:
    """A route changed during user input must survive instead of being overwritten."""
    primary = ProviderConfig(
        adapter="gemini",
        model="primary-model",
        credential_ref="primary-ref",
        declared_capabilities={Modality.AUDIO: True},
        verified_capabilities={Modality.AUDIO: True},
    )
    fallback = ProviderConfig(
        adapter="gemini",
        model="fallback-model",
        credential_ref="fallback-ref",
        declared_capabilities={Modality.AUDIO: True},
        verified_capabilities={Modality.AUDIO: True},
    )
    tmp_services.config_store.save(
        AppConfig(providers={"primary": primary, "fallback": fallback})
    )
    external = ConfigStore(tmp_services.config_store.path)

    def change_route_while_prompted(_: str) -> str:
        def mutate(config: AppConfig) -> None:
            config.routes.audio = RouteConfig(primary="fallback")
            config.allowed_media_roots.append("late-setting")

        external.update(mutate)
        return "yes"

    with pytest.raises(SensoryError) as caught:
        cli._write_verified_routes(
            tmp_services,
            "primary",
            [Modality.AUDIO],
            change_route_while_prompted,
        )

    assert caught.value.code is ErrorCode.CONFIG_INVALID
    saved = tmp_services.config_store.load()
    assert saved.routes.audio == RouteConfig(primary="fallback")
    assert saved.allowed_media_roots == ["late-setting"]


def test_configure_declined_verification_saves_provider_but_no_routes(
    tmp_services: AppServices,
) -> None:
    """Users may save local configuration and defer every quota-bearing self-test."""
    answers = iter(
        [
            "minimax-m3",
            "minimax-main",
            "global",
            "MiniMax-M3",
            "yes",  # eye
            "no",  # verification
        ]
    )
    called = False

    async def unexpected_verify(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "ok"}

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "minimax-secret",
            output=lambda _: None,
            verify_fn=unexpected_verify,
        )
        == 0
    )
    config = tmp_services.config_store.load()
    assert set(config.providers) == {"minimax-m3"}
    assert config.routes.image is None
    assert config.routes.video_visual is None
    assert called is False


def test_cli_status_lists_only_verified_routed_modalities_without_foundation_claim(
    tmp_services: AppServices,
) -> None:
    """The CLI must not hide a working route or advertise a stale unverified one."""
    provider = ProviderConfig(
        adapter="gemini",
        model="test-model",
        credential_ref="test-ref",
        declared_capabilities={
            Modality.IMAGE: True,
            Modality.VIDEO_VISUAL: True,
        },
        verified_capabilities={Modality.IMAGE: True},
    )
    tmp_services.config_store.save(
        AppConfig(
            providers={"vision": provider},
            routes={
                "image": {"primary": "vision"},
                "video_visual": {"primary": "vision"},
            },
        )
    )
    messages: list[str] = []

    assert run_status(tmp_services, output=messages.append) == 0

    report = "\n".join(messages).lower()
    assert "verified image: vision" in report
    assert "verified video_visual" not in report
    assert "foundation provider perception" not in report


@pytest.fixture
def tmp_services(tmp_path: Path) -> AppServices:
    """Compose local-only services for CLI behavior tests."""
    return AppServices(
        config_store=ConfigStore(tmp_path / "config.yaml"),
        secret_store=MemorySecretStore(),
    )


def test_version_is_semver() -> None:
    major, minor, patch = __version__.split(".")
    assert major.isdigit() and minor.isdigit() and patch.isdigit()


def test_version_command_prints_only_version(capsys) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"cove-sensory-mcp {__version__}\n"


def test_version_does_not_compose_runtime_services(monkeypatch, capsys) -> None:
    """Composing paths or keyring for --version would add machine side effects."""

    def fail_composition(*args: object, **kwargs: object) -> None:
        raise AssertionError("--version must not compose runtime services")

    monkeypatch.setattr(cli.AppPaths, "for_system", fail_composition)
    monkeypatch.setattr(cli, "ConfigStore", fail_composition)
    monkeypatch.setattr(cli, "KeyringSecretStore", fail_composition)

    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"cove-sensory-mcp {__version__}\n"


def test_status_command_composes_services_only_after_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Losing the composed service return would break every non-version local command."""
    monkeypatch.setattr(
        cli.AppPaths,
        "for_system",
        lambda *args, **kwargs: AppPaths(
            config_file=tmp_path / "config.yaml",
            jobs_dir=tmp_path / "jobs",
        ),
    )
    monkeypatch.setattr(cli, "KeyringSecretStore", MemorySecretStore)

    assert main(["status"]) == 0
    assert "Configuration: readable" in capsys.readouterr().out


def test_installed_metadata_pins_sdk_used_by_version_entrypoint() -> None:
    """A broad MCP requirement can install an SDK that cannot import the CLI server API."""
    project_requirements = requires("cove-sensory-mcp") or []

    assert "mcp==2.0.0" in project_requirements
    assert version("mcp") == "2.0.0"

    completed = subprocess.run(
        [sys.executable, "-m", "cove_sensory_mcp", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"cove-sensory-mcp {__version__}\n"
    assert completed.stderr == ""


def test_configure_saves_reference_but_not_secret(
    tmp_services: AppServices, capsys: pytest.CaptureFixture[str]
) -> None:
    """Saving the key in YAML or printing it would breach the credential boundary."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "n"])
    code = run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "test-secret-that-must-not-print",
        output=lambda message: print(message),
    )

    assert code == 0
    saved = tmp_services.config_store.load()
    assert saved.providers["gemini"].credential_ref == "gemini-main"
    assert saved.providers["gemini"].model == "gemini-test-model"
    captured = capsys.readouterr()
    assert "test-secret-that-must-not-print" not in captured.out + captured.err
    assert "test-secret-that-must-not-print" not in (
        tmp_services.config_store.path.read_text(encoding="utf-8")
    )


def test_configure_stops_after_one_provider_and_its_separate_role_steps(
    tmp_services: AppServices,
) -> None:
    """Role selection must not restore an unbounded add-another-provider loop."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "no", "no"])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(answers)
        except StopIteration:
            pytest.fail("configure asked an extra question after both role decisions")

    assert (
        run_configure(
            tmp_services,
            input_fn=answer,
            secret_input_fn=lambda _: "test-secret-that-stays-local",
            output=lambda _: None,
        )
        == 0
    )
    assert any("eye" in prompt.lower() for prompt in prompts)
    assert any("ear" in prompt.lower() for prompt in prompts)
    assert all("another provider" not in prompt.lower() for prompt in prompts)


def test_configure_explains_capabilities_and_privacy_before_accepting_key(
    tmp_services: AppServices,
) -> None:
    """Prompting for a key before disclosure would deny informed local configuration."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "n"])
    messages: list[str] = []

    def secret_after_notice(_: str) -> str:
        notice = " ".join(messages).lower()
        assert "image" in notice and "audio" in notice
        assert "local" in notice and "never" in notice and "chat" in notice
        return "test-secret-that-stays-local"

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=secret_after_notice,
            output=messages.append,
        )
        == 0
    )


def test_configure_environment_reference_never_prompts_for_or_stores_key(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating env: as a keyring reference would prompt for and persist an unwanted key."""
    environment_name = "TEST_GEMINI_API_KEY"
    environment_secret = "environment-secret-must-not-print"
    monkeypatch.setenv(environment_name, environment_secret)
    answers = iter(["gemini", f"env:{environment_name}", "gemini-test-model", "n"])
    messages: list[str] = []

    def unexpected_secret_prompt(_: str) -> str:
        pytest.fail("environment mode must not prompt for a secret")

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=unexpected_secret_prompt,
            output=messages.append,
        )
        == 0
    )

    provider = tmp_services.config_store.load().providers["gemini"]
    assert provider.api_key_env == environment_name
    assert provider.credential_ref is None
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}
    report = "\n".join(messages)
    assert "available" in report.lower()
    assert environment_name not in report
    assert environment_secret not in report


def test_configure_environment_only_succeeds_when_keyring_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consulting keyring in environment mode would block supported CI/server setup."""
    environment_name = "TEST_SERVER_API_KEY"
    monkeypatch.setenv(environment_name, "environment-secret-value")

    def unavailable_keyring(*args: object, **kwargs: object) -> None:
        raise RuntimeError("keyring backend unavailable")

    monkeypatch.setattr(keyring, "get_password", unavailable_keyring)
    monkeypatch.setattr(keyring, "set_password", unavailable_keyring)
    monkeypatch.setattr(keyring, "delete_password", unavailable_keyring)

    services = AppServices(
        config_store=ConfigStore(tmp_path / "config.yaml"),
        secret_store=cli.KeyringSecretStore(),
    )
    answers = iter(["gemini", f"env:{environment_name}", "gemini-test-model", "n"])

    assert (
        run_configure(
            services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: pytest.fail(
                "environment mode prompted for a key"
            ),
            output=lambda _: None,
        )
        == 0
    )
    assert (
        services.config_store.load().providers["gemini"].api_key_env == environment_name
    )


def test_configure_missing_environment_reference_saves_and_reports_only_absence(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requiring an environment value at save time would make staged server setup impossible."""
    environment_name = "TEST_LATER_API_KEY"
    monkeypatch.delenv(environment_name, raising=False)
    answers = iter(["gemini", f"env:{environment_name}", "gemini-test-model", "n"])
    messages: list[str] = []

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: pytest.fail(
                "environment mode prompted for a key"
            ),
            output=messages.append,
        )
        == 0
    )

    report = "\n".join(messages)
    assert "missing" in report.lower()
    assert environment_name not in report


@pytest.mark.parametrize(
    "environment_name",
    ["", "1PRIVATE_KEY", "PRIVATE-KEY", "PRIVATE.KEY", "PRIVATE\nKEY", "P" * 129],
)
def test_configure_rejects_invalid_environment_reference_without_persistence(
    tmp_services: AppServices, environment_name: str
) -> None:
    """Accepting a nonportable env: form would defer failure until another platform runs it."""
    answers = iter(["gemini", f"env:{environment_name}", "gemini-test-model", "n"])

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: pytest.fail(
                "invalid environment mode prompted for a key"
            ),
            output=lambda _: None,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


@pytest.mark.parametrize(
    ("region", "expected_base_url"),
    [
        ("cn", "https://api.minimaxi.com/v1"),
        ("global", "https://api.minimax.io/v1"),
    ],
)
def test_configure_minimax_region_selects_non_secret_base_url(
    tmp_services: AppServices, region: str, expected_base_url: str
) -> None:
    """Mapping a region to the wrong host would send later media to the wrong endpoint."""
    answers = iter(["minimax-m3", "minimax-main", region, "MiniMax-M3", "n"])

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "minimax-test-secret",
            output=lambda _: None,
        )
        == 0
    )

    provider = tmp_services.config_store.load().providers["minimax-m3"]
    assert provider.adapter == "minimax-m3"
    assert provider.base_url == expected_base_url


def test_configure_minimax_custom_endpoint_rejects_secret_bearing_url(
    tmp_services: AppServices,
) -> None:
    """A credential-bearing custom URL would bypass keyring-only secret storage."""
    answers = iter(
        [
            "minimax-m3",
            "minimax-main",
            "custom",
            "https://api.example.test/v1?api_key=must-not-save",
            "MiniMax-M3",
            "n",
        ]
    )

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "minimax-test-secret",
            output=lambda _: None,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


def test_configure_minimax_custom_endpoint_with_numeric_port_is_saved(
    tmp_services: AppServices,
) -> None:
    """Rejecting a valid explicit HTTPS port would block legitimate custom deployments."""
    answers = iter(
        [
            "minimax-m3",
            "minimax-main",
            "custom",
            "https://api.example.test:8443/v1",
            "MiniMax-M3",
            "n",
        ]
    )

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "minimax-test-secret",
            output=lambda _: None,
        )
        == 0
    )
    assert (
        tmp_services.config_store.load().providers["minimax-m3"].base_url
        == "https://api.example.test:8443/v1"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.test:not-a-port/v1",
        "https://api.example.test:70000/v1",
    ],
)
def test_configure_minimax_custom_endpoint_rejects_malformed_port_before_persistence(
    tmp_services: AppServices, base_url: str
) -> None:
    """Deferring malformed-port failure would save an unusable endpoint and credential."""
    answers = iter(
        ["minimax-m3", "minimax-main", "custom", base_url, "MiniMax-M3", "n"]
    )

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "minimax-test-secret",
            output=lambda _: None,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


def test_configure_custom_provider_saves_only_declared_supported_capabilities(
    tmp_services: AppServices,
) -> None:
    """Treating declarations as verified or accepting unknown modalities would overclaim access."""
    answers = iter(
        [
            "custom",
            "studio-sense",
            "studio-sense-key",
            "https://api.example.test/v1",
            "sense-model",
            "image,audio,music",
            "n",
        ]
    )

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "custom-provider-secret",
            output=lambda _: None,
        )
        == 0
    )

    provider = tmp_services.config_store.load().providers["studio-sense"]
    assert provider.adapter == "openai-compatible"
    assert provider.declared_capabilities == {
        "image": True,
        "video_visual": False,
        "video_audio": False,
        "audio": True,
        "music": True,
    }
    assert "verified_capabilities" not in tmp_services.config_store.path.read_text(
        encoding="utf-8"
    )


def test_configure_rejects_unknown_custom_capability_without_saving(
    tmp_services: AppServices,
) -> None:
    """Silently accepting an unknown capability would create an unsafe future route claim."""
    answers = iter(
        [
            "custom",
            "studio-sense",
            "studio-sense-key",
            "https://api.example.test/v1",
            "sense-model",
            "image,telepathy",
            "n",
        ]
    )

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "custom-provider-secret",
            output=lambda _: None,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


@pytest.mark.parametrize("cancel_value", ["cancel", "quit", "q"])
@pytest.mark.parametrize(
    "answer_prefix",
    [
        pytest.param([], id="provider-choice"),
        pytest.param(["gemini"], id="gemini-credential-reference"),
        pytest.param(["gemini", "gemini-main"], id="gemini-model"),
        pytest.param(["minimax-m3"], id="minimax-credential-reference"),
        pytest.param(["minimax-m3", "minimax-main"], id="minimax-region"),
        pytest.param(
            ["minimax-m3", "minimax-main", "custom"],
            id="minimax-custom-url",
        ),
        pytest.param(
            ["minimax-m3", "minimax-main", "global"],
            id="minimax-model",
        ),
        pytest.param(["custom"], id="custom-provider-id"),
        pytest.param(["custom", "studio-sense"], id="custom-credential-reference"),
        pytest.param(
            ["custom", "studio-sense", "studio-sense-key"],
            id="custom-base-url",
        ),
        pytest.param(
            [
                "custom",
                "studio-sense",
                "studio-sense-key",
                "https://api.example.test/v1",
            ],
            id="custom-model",
        ),
        pytest.param(
            [
                "custom",
                "studio-sense",
                "studio-sense-key",
                "https://api.example.test/v1",
                "sense-model",
            ],
            id="custom-capabilities",
        ),
    ],
)
def test_configure_cancel_at_any_plaintext_prompt_does_not_persist(
    tmp_services: AppServices, answer_prefix: list[str], cancel_value: str
) -> None:
    """Continuing after an explicit cancellation could prompt for or persist a credential."""
    answers = iter([*answer_prefix, cancel_value])
    secret_prompted = False

    def supplied_answer(_: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            pytest.fail("wizard prompted again after cancellation")

    def unexpected_secret_prompt(_: str) -> str:
        nonlocal secret_prompted
        secret_prompted = True
        return "must-not-be-stored"

    messages: list[str] = []
    assert (
        run_configure(
            tmp_services,
            input_fn=supplied_answer,
            secret_input_fn=unexpected_secret_prompt,
            output=messages.append,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}
    assert secret_prompted is False
    assert "cancelled" in " ".join(messages).lower()


@pytest.mark.parametrize("cancel_exception", [EOFError, KeyboardInterrupt])
def test_configure_hidden_secret_prompt_cancels_only_on_terminal_interruption(
    tmp_services: AppServices, cancel_exception: type[BaseException]
) -> None:
    """A terminal cancellation at hidden input must stop before either persistent write."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model"])

    def interrupted_secret(_: str) -> str:
        raise cancel_exception

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=interrupted_secret,
            output=lambda _: None,
        )
        == 1
    )
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


def test_configure_does_not_parse_hidden_secret_text_as_a_cancel_command(
    tmp_services: AppServices,
) -> None:
    """Secret input must remain opaque even when its text contains a cancellation word."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "n"])

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "cancel-is-part-of-this-secret",
            output=lambda _: None,
        )
        == 0
    )
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {
        "gemini-main": "cancel-is-part-of-this-secret"
    }


def test_configure_refuses_occupied_secret_reference_without_overwrite_or_delete(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed save after overwriting an occupied ref must not destroy its original value."""
    original_secret = "original-secret-must-survive"
    replacement_secret = "replacement-secret-must-not-store"
    reference = "occupied-private-reference"
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    tmp_services.secret_store.set(reference, original_secret)
    answers = iter(["gemini", reference, "gemini-test-model", "n"])
    secret_prompted = False

    def supplied_secret(_: str) -> str:
        nonlocal secret_prompted
        secret_prompted = True
        return replacement_secret

    def fail_save(config: AppConfig) -> None:
        del config
        raise OSError("forced config save failure")

    monkeypatch.setattr(tmp_services.config_store, "save", fail_save)

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=supplied_secret,
            output=lambda message: print(message),
        )
        == 1
    )
    assert secret_prompted is False
    assert tmp_services.secret_store.values == {reference: original_secret}
    assert not tmp_services.config_store.path.exists()
    captured = capsys.readouterr()
    public_output = captured.out + captured.err
    assert reference not in public_output
    assert original_secret not in public_output
    assert replacement_secret not in public_output


def test_configure_deletes_just_created_secret_when_config_save_fails(
    tmp_services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed config write must not leave an orphan credential behind."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "n"])

    def fail_update(mutator: object) -> None:
        del mutator
        raise OSError("private path must not print")

    monkeypatch.setattr(tmp_services.config_store, "update", fail_update)

    assert (
        run_configure(
            tmp_services,
            input_fn=lambda _: next(answers),
            secret_input_fn=lambda _: "rollback-test-secret",
            output=lambda _: None,
        )
        == 1
    )
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


def test_status_reports_credential_presence_without_values_or_references(
    tmp_services: AppServices, capsys: pytest.CaptureFixture[str]
) -> None:
    """Status must not expose a key, its reference, or key metadata."""
    secret = "status-secret-that-must-not-print"
    tmp_services.secret_store.set("private-reference", secret)
    tmp_services.config_store.save(
        AppConfig(
            providers={
                "gemini": ProviderConfig(
                    adapter="gemini",
                    model="gemini-test-model",
                    credential_ref="private-reference",
                )
            }
        )
    )

    assert run_status(tmp_services, output=lambda message: print(message)) == 0

    captured_result = capsys.readouterr()
    captured = captured_result.out + captured_result.err
    assert "gemini" in captured
    assert "available" in captured.lower()
    assert secret not in captured
    assert "private-reference" not in captured
    assert str(len(secret)) not in captured


def test_status_rejects_invalid_provider_identifier_without_echo(
    tmp_services: AppServices,
) -> None:
    """Printing a rejected provider-map key would expose untrusted multiline config data."""
    private_identifier = "private\nprovider"
    tmp_services.config_store.path.write_text(
        "version: 1\nproviders:\n  ? |\n    private\n    provider\n  :\n"
        "    adapter: gemini\n"
        "    model: gemini-test\n"
        "    credential_ref: private-reference\n",
        encoding="utf-8",
    )
    messages: list[str] = []

    assert run_status(tmp_services, output=messages.append) == 1

    report = "\n".join(messages)
    assert report == "Configuration: invalid\nVerified sensory routes: unavailable"
    assert private_identifier not in report


def test_doctor_rejects_invalid_provider_identifier_without_echo(
    tmp_services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor must bound invalid config diagnostics before iterating printable provider IDs."""
    private_identifier = "private/provider"
    tmp_services.config_store.path.write_text(
        "version: 1\nproviders:\n"
        f"  {private_identifier}:\n"
        "    adapter: gemini\n"
        "    model: gemini-test\n"
        "    credential_ref: private-reference\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli.shutil, "which", lambda executable: None)
    messages: list[str] = []

    assert run_doctor(tmp_services, output=messages.append) == 1

    report = "\n".join(messages)
    assert "Config: invalid" in report
    assert private_identifier not in report


def test_doctor_reports_missing_media_runtime_without_crashing(
    tmp_services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent optional FFmpeg installation must produce a bounded diagnostic."""
    messages: list[str] = []
    monkeypatch.setattr(cli.shutil, "which", lambda executable: None)

    assert run_doctor(tmp_services, output=messages.append) == 1

    report = "\n".join(messages).lower()
    assert "config" in report and "ok" in report
    assert "credential" in report
    assert "cache" in report and "ok" in report
    assert "ffmpeg" in report and "missing" in report


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_doctor_probes_and_cleans_the_injected_platform_jobs_root(
    tmp_path: Path,
    system: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to a generic temp root would miss failures in the configured jobs cache."""
    roaming = tmp_path / "Roaming" if system == "Windows" else None
    local = tmp_path / "Local" if system == "Windows" else None
    paths = AppPaths.for_system(
        system,
        home=tmp_path / "Home",
        roaming=roaming,
        local=local,
    )
    monkeypatch.setattr(cli.AppPaths, "for_system", lambda *args, **kwargs: paths)
    monkeypatch.setattr(cli, "KeyringSecretStore", MemorySecretStore)
    monkeypatch.setattr(cli.shutil, "which", lambda executable: "/test/bin/ffmpeg")
    real_temporary_directory = tempfile.TemporaryDirectory
    probed_roots: list[Path | None] = []

    def recording_temporary_directory(*args: object, **kwargs: object):
        directory = kwargs.get("dir")
        probed_roots.append(
            Path(directory) if isinstance(directory, (str, Path)) else None
        )
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(
        cli.tempfile, "TemporaryDirectory", recording_temporary_directory
    )
    services = cli._build_services()
    assert isinstance(services.secret_store, MemorySecretStore)
    services.secret_store.set("gemini-main", "doctor-test-secret")
    services.config_store.save(
        AppConfig(
            providers={
                "gemini": ProviderConfig(
                    adapter="gemini",
                    model="gemini-test",
                    credential_ref="gemini-main",
                )
            }
        )
    )

    assert run_doctor(services, output=lambda _: None) == 0
    assert probed_roots == [paths.jobs_dir]
    assert paths.jobs_dir.is_dir()
    assert list(paths.jobs_dir.iterdir()) == []


def test_doctor_bounds_jobs_probe_failure_to_the_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed jobs-root probe must not switch to or remove another temporary directory."""
    paths = AppPaths.for_system(
        "Darwin",
        home=tmp_path / "Home",
        roaming=None,
        local=None,
    )
    paths.jobs_dir.parent.mkdir(parents=True)
    paths.jobs_dir.write_text(
        "leave this configured-root blocker intact", encoding="utf-8"
    )
    monkeypatch.setattr(cli.AppPaths, "for_system", lambda *args, **kwargs: paths)
    monkeypatch.setattr(cli, "KeyringSecretStore", MemorySecretStore)
    monkeypatch.setattr(cli.shutil, "which", lambda executable: "/test/bin/ffmpeg")
    messages: list[str] = []

    assert run_doctor(cli._build_services(), output=messages.append) == 1

    assert "Cache create/remove: failed" in messages
    assert paths.jobs_dir.read_text(encoding="utf-8") == (
        "leave this configured-root blocker intact"
    )


def test_confirmed_empty_self_test_reports_setup_required_without_provider_claim(
    tmp_services: AppServices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicitly confirmed empty self-test must still report truthful setup state."""
    monkeypatch.setattr(cli, "_build_services", lambda: tmp_services)

    assert main(["self-test", "--yes"]) == 1

    captured = capsys.readouterr()
    assert "SETUP_REQUIRED" in captured.out
    assert "tiny test media" in captured.out.lower()
    assert "provider quota" in captured.out.lower()
    assert "provider" in captured.out.lower()
    assert captured.err == ""
