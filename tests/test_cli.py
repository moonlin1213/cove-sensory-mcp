import subprocess
import sys
from importlib.metadata import requires, version
from pathlib import Path

import pytest

from cove_sensory_mcp import __version__, cli
from cove_sensory_mcp.cli import main, run_configure, run_doctor, run_status
from cove_sensory_mcp.config.paths import AppPaths
from cove_sensory_mcp.config.schema import AppConfig, ProviderConfig
from cove_sensory_mcp.config.secrets import MemorySecretStore
from cove_sensory_mcp.config.store import ConfigStore
from cove_sensory_mcp.services import AppServices


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

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=secret_after_notice,
        output=messages.append,
    ) == 0


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

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "minimax-test-secret",
        output=lambda _: None,
    ) == 0

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

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "minimax-test-secret",
        output=lambda _: None,
    ) == 1
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

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "custom-provider-secret",
        output=lambda _: None,
    ) == 0

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

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "custom-provider-secret",
        output=lambda _: None,
    ) == 1
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}


def test_configure_cancel_does_not_save_or_prompt_for_secret(tmp_services: AppServices) -> None:
    """Cancel must leave both config and credential storage untouched."""
    secret_prompted = False

    def unexpected_secret_prompt(_: str) -> str:
        nonlocal secret_prompted
        secret_prompted = True
        return "must-not-be-stored"

    assert run_configure(
        tmp_services,
        input_fn=lambda _: "cancel",
        secret_input_fn=unexpected_secret_prompt,
        output=lambda _: None,
    ) == 1
    assert not tmp_services.config_store.path.exists()
    assert isinstance(tmp_services.secret_store, MemorySecretStore)
    assert tmp_services.secret_store.values == {}
    assert secret_prompted is False


def test_configure_deletes_just_created_secret_when_config_save_fails(
    tmp_services: AppServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed config write must not leave an orphan credential behind."""
    answers = iter(["gemini", "gemini-main", "gemini-test-model", "n"])

    def fail_save(config: AppConfig) -> None:
        del config
        raise OSError("private path must not print")

    monkeypatch.setattr(tmp_services.config_store, "save", fail_save)

    assert run_configure(
        tmp_services,
        input_fn=lambda _: next(answers),
        secret_input_fn=lambda _: "rollback-test-secret",
        output=lambda _: None,
    ) == 1
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


def test_foundation_self_test_subcommand_stays_local_and_reports_setup_required(
    tmp_services: AppServices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Foundation self-test must not imply that an unimplemented provider was contacted."""
    monkeypatch.setattr(cli, "_build_services", lambda: tmp_services)

    assert main(["self-test"]) == 1

    captured = capsys.readouterr()
    assert "SETUP_REQUIRED" in captured.out
    assert "provider" in captured.out.lower()
    assert captured.err == ""
