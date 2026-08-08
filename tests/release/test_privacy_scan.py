from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _scanner():
    spec = importlib.util.spec_from_file_location("privacy_scan", ROOT / "scripts/privacy_scan.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scanner_reports_rules_without_echoing_secret(tmp_path: Path) -> None:
    scanner = _scanner()
    (tmp_path / "bad.txt").write_text("/Users/privateperson/work\nsk-" + "A" * 32, encoding="utf-8")
    findings = scanner.scan(tmp_path)
    rendered = "\n".join(item.render() for item in findings)
    assert "PRIVATE_UNIX_USER_PATH" in rendered and "API_KEY" in rendered
    assert "privateperson" not in rendered and "A" * 20 not in rendered


def test_markdown_allows_only_generic_user_examples(tmp_path: Path) -> None:
    scanner = _scanner()
    (tmp_path / "README.md").write_text("/Users/alice/media\nC:\\Users\\Alice\\media\n", encoding="utf-8")
    assert scanner.scan(tmp_path) == []
    (tmp_path / ".env").write_text("KEY=value", encoding="utf-8")
    assert {item.rule for item in scanner.scan(tmp_path)} == {"FORBIDDEN_DATA_FILE"}
