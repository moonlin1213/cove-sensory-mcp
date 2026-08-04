import subprocess
import sys
from importlib.metadata import requires, version

from cove_sensory_mcp import __version__, cli
from cove_sensory_mcp.cli import main


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
