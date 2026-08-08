# Contributing

Keep changes narrow and add tests for behavior and failure paths. Before submitting, run
Pytest, Ruff, mypy, the offline package build, and `scripts/privacy_scan.py`. Never add a
real credential, `.env`, user configuration, database, raw personal media, private path,
endpoint, Provider response, or copied copyrighted fixture. New Provider behavior must
link its official protocol/capability documentation, declare data transmission and cost,
normalize output, and pass project-owned tiny-media verification. Apache-2.0 applies to
contributions; third-party code and media need compatible provenance and notices.
