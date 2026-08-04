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
