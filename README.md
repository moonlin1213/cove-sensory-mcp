# Cove Sensory MCP

**Cove Sensory MCP——给纯文本 LLM 一双眼睛和耳朵** is a local stdio MCP server
that lets an Agent ask configured multimodal providers to inspect an explicitly supplied
image, video, audio file, or music file. It is a sensory layer only: it does not provide
chat, memory, personality, playback, continuous monitoring, or a calling policy.

## Install and connect

Python 3.11+ users can install with `uvx` or `pipx`. Standalone macOS (Apple Silicon and
Intel) and Windows x64 archives are produced by release CI; the installers stay inside
the current user account and never require administrator rights.

The `0.1.0` release candidate is not public yet. Standalone archives must be treated as
unsigned unless the downloaded draft's native CI evidence explicitly confirms signing
(and, on macOS, notarization). The current local acceptance run does not substitute for
the three native CI jobs or clean-account installation journeys.

```console
uvx cove-sensory-mcp doctor
uvx cove-sensory-mcp print-config --client generic
```

Supported renderers are `generic`, `codex`, `claude-desktop`, and `claude-code`. Copy the
printed local stdio entry into your client, restart it, then ask the Agent to call
`sensory_setup_guide`. Official configuration references:
[Codex MCP](https://developers.openai.com/codex/mcp/),
[Claude Desktop](https://modelcontextprotocol.io/quickstart/user), and
[Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp).
Client surfaces change independently; if a surface does not support local stdio MCP,
run the server from a compatible host instead.

Do not paste an API key into chat. Run `cove-sensory-mcp configure` in a local terminal:
hidden input goes to the operating-system credential store, while `env:VARIABLE_NAME`
stores only the variable name. Add each readable local directory separately with
`cove-sensory-mcp configure paths`.

## Choose the eyes and ears

- Gemini can be an eye for images/video and an ear for video audio, ordinary audio, and
  music.
- MiniMax-M3 can be an image/native-video eye; it is not treated as an ear, so choose a
  separate audio provider for a video's soundtrack.
- A custom provider must declare capabilities and pass the tiny-media self-test before
  those capabilities are advertised. The self-test sends project-created media, can
  incur a small API cost, and requires confirmation.

Media is sent only to the selected, authorized Provider. Its privacy policy, retention,
region, and billing rules apply. Cross-provider fallback is never inferred. Local paths
must be absolute and within configured roots. URLs must be direct HTTPS media URLs;
redirects, credentials, private networks, localhost, and metadata endpoints are blocked.

## Tools

The seven public tools are:

- `sensory_status` — show redacted, verified capability status.
- `sensory_setup_guide` — explain missing local setup without requesting secrets.
- `sensory_self_test` — verify selected capabilities using tiny included media.
- `sense_image` — inspect an authorized image; e.g. “read the visible labels.”
- `sense_video` — inspect visuals and, when configured, its audio timeline.
- `sense_audio` — describe speech and non-speech events in an audio clip.
- `sense_music` — describe structure, rhythm, instrumentation, and key moments.

Run `cove-sensory-mcp doctor` for local configuration, credential-presence, cache, and
FFmpeg diagnostics. It never prints secret values or contacts a Provider. The optional
FFmpeg download stays disabled until each platform binary has audited provenance; a
working system FFmpeg can be used meanwhile.

## Uninstall and privacy

Remove the `cove-sensory-mcp` Python tool with the installer you used, or run the
standalone installer's uninstall command. Uninstall preserves configuration and OS
credentials unless you explicitly request data removal. Temporary derived media is
request-scoped and removed after completion; the original is never modified.

The project is Apache-2.0. Dependencies and FFmpeg keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Please read [SECURITY.md](SECURITY.md)
before processing untrusted media.

## Development

```console
uv sync --group dev
uv run pytest
uv run ruff check src tests scripts
uv run mypy src/cove_sensory_mcp
```
