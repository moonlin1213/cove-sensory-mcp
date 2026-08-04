from cove_sensory_mcp import __version__
from cove_sensory_mcp.cli import main


def test_version_is_semver() -> None:
    major, minor, patch = __version__.split(".")
    assert major.isdigit() and minor.isdigit() and patch.isdigit()


def test_version_command_prints_only_version(capsys) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"cove-sensory-mcp {__version__}\n"
