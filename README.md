# Cove Sensory MCP

**Cove Sensory MCP——给纯文本 LLM 一双眼睛和耳朵** is a local stdio MCP server
that lets an Agent ask configured multimodal providers to inspect an explicitly supplied
image, video, audio file, or music file. It is a sensory layer only: it does not provide
chat, memory, personality, playback, continuous monitoring, or a calling policy.

## Install and connect from source

This repository is distributed as source code. It does not require a packaged desktop
app or a standalone installer. Install [Git](https://git-scm.com/), Python 3.11+, and
[uv](https://docs.astral.sh/uv/), then clone and set up the locked environment:

```console
git clone https://github.com/moonlin1213/cove-sensory-mcp.git
cd cove-sensory-mcp
uv sync --locked
uv run cove-sensory-mcp doctor
uv run cove-sensory-mcp print-config --client generic
```

Supported renderers are `generic`, `codex`, `claude-desktop`, and `claude-code`. Copy the
printed local stdio entry into your client, restart it, then ask the Agent to call
`sensory_setup_guide`. Official configuration references:
[Codex MCP](https://developers.openai.com/codex/mcp/),
[Claude Desktop](https://modelcontextprotocol.io/quickstart/user), and
[Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp).
Client surfaces change independently; if a surface does not support local stdio MCP,
run the server from a compatible host instead.

Run later commands from the cloned repository with `uv run cove-sensory-mcp ...`.
FFmpeg is optional for initial setup and image sensing, but video/audio preparation
requires a working system FFmpeg discoverable on `PATH` or configured locally.

### Linux

Linux is supported from source. Install Git and FFmpeg with the distribution package
manager, then follow the locked `uv sync` workflow above. For example, on Ubuntu or
Debian:

```console
sudo apt update
sudo apt install -y git ffmpeg
```

Configuration follows the XDG Base Directory convention. When the variables are not
set, the defaults are `~/.config/cove-sensory-mcp/config.yaml` for persistent config
and `~/.cache/cove-sensory-mcp/jobs` for transient media jobs. A desktop keyring may
store Provider credentials. On a headless Linux host without a keyring backend, choose
an `env:VARIABLE_NAME` credential reference and make that variable available to the MCP
client process. The repository directory and its `.venv` must remain available because
the MCP client launches `.venv/bin/cove-sensory-mcp` directly.

Do not paste an API key into chat. Run `cove-sensory-mcp configure` in a local terminal:
hidden input goes to the operating-system credential store, while `env:VARIABLE_NAME`
stores only the variable name. Add each readable local directory separately with
`cove-sensory-mcp configure paths`.

### Continue an interrupted setup

Run `uv run cove-sensory-mcp configure` again and select the same Provider. For a
custom Provider, select `custom` and enter its existing identifier. The wizard detects
the saved entry **before** asking for credentials and offers to continue setup. It
keeps the saved model, endpoint, and credential source; this is not a credential-edit
or overwrite operation.

Choose the eye and/or ear roles you want to enable. The wizard checks whether the
credential can be read locally, then requests permission to send tiny test media for
only the selected, still-unverified capabilities. Previously verified capabilities
can be enabled without another paid test. A capability becomes the default only after
verification succeeds. Replacing a different default requires a separate confirmation;
continuing an existing default preserves its authorized fallbacks. Declining or
cancelling keeps the Provider settings and any completed verification, so you can retry.

Use these local diagnostics when setup is incomplete:

```console
uv run cove-sensory-mcp --version
uv run cove-sensory-mcp status
uv run cove-sensory-mcp doctor
```

`status` distinguishes configured-but-not-verified capabilities from verified-but-not-
enabled defaults, and checks local credential availability separately. `sensory_status`
is a configuration-only MCP status tool: it does not read credentials or check Provider
connectivity. Do not manually mark capabilities verified to bypass a failed test.

With `env:VARIABLE_NAME`, the variable must actually be set in the process running the
MCP. Typing that reference does not set an API key. On Windows, a variable set only in
one terminal is not necessarily available to an already-running desktop client; make
it available to the MCP launch process and restart the terminal/client as needed.
Missing local credentials are not the same as a Provider rejecting a key. Verification
failures show bounded error codes, not secret values or raw Provider responses. Never
send API keys, environment dumps, or entire configuration files in chat.

You do not need all five capabilities enabled: MiniMax's built-in eye covers `image`
and `video_visual`, not hearing. In an MCP self-test, request only the capabilities
you have configured (for example, `modalities: ["image", "video_visual"]`). The CLI
`self-test` command still requests all five; use the configuration wizard for role-
specific testing and activation. A standalone self-test records verification but does
not choose your default Providers; return to `configure` to enable them.

## Choose the eyes and ears

- Gemini can be an eye for images/video and an ear for video audio, ordinary audio, and
  music.
- MiniMax-M3 can be an image/native-video eye; it is not treated as an ear, so choose a
  separate audio provider for a video's soundtrack.
- An OpenAI-compatible provider using the `audio_url_data_uri` media mode can serve as
  the ear on platforms that accept an `audio_url` content part (for example SiliconFlow's
  Qwen3-Omni-Instruct), covering audio, music, and video audio. One bounded retry covers
  transient 408, 429, 500, 502, 503, and 504 responses, connection/timeout failures, or
  malformed structured output, and may incur one additional Provider call or charge.
- A custom provider must declare capabilities and pass the tiny-media self-test before
  those capabilities are advertised. The self-test sends project-created media, can
  incur a small API cost, and requires confirmation. It verifies that the Provider call
  returns a valid structured observation; it does not score the model's wording or
  description style.

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
