# Changelog

## [0.1.0] - release candidate

- Initial local stdio MCP with seven sensory tools.
- Gemini, MiniMax-M3, and bounded custom Provider adapters.
- Authorized local/direct-URL media preparation and deterministic evidence reports.
- Secret-safe local setup and verified capability advertising.

Release notes:

- Python wheel/sdist support macOS and Windows with Python 3.11 or later.
- Standalone targets are macOS Apple Silicon, macOS Intel, and Windows x64. Native CI
  evidence is required before the draft artifacts are published.
- Local candidates are unsigned and not notarized unless their native release job
  explicitly performs and verifies signing; the GitHub workflow creates drafts only.
- Gemini availability depends on the configured model/account. MiniMax-M3 is an
  image/native-video eye, not an audio ear. Custom capabilities are advertised only
  after a real tiny-media self-test.
- Version 0.1.0 accepts only explicitly supplied local files or direct HTTPS media URLs;
  it does not browse pages, capture devices, or continuously monitor media.
- Provider verification and perception calls may incur API charges and are governed by
  the chosen Provider's retention and privacy terms.
- Optional automatic FFmpeg download remains disabled while exact three-platform binary
  provenance is awaiting native audit. A working system FFmpeg remains supported.
