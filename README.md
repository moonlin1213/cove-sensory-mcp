# Cove Sensory MCP

Cove Sensory MCP——给纯文本 LLM 一双眼睛和耳朵

A privacy-safe, cross-platform local stdio MCP package for Python 3.11 and later.

## Development

This repository currently implements the **foundation milestone only**. It has no
working perception Provider yet: Gemini, MiniMax-M3, and custom entries can be saved
locally, but they cannot inspect images, video, audio, or music until the provider
milestone is implemented.

Install the development dependencies and exercise the local foundation with:

```console
uv sync --group dev
uv run cove-sensory-mcp --version
uv run cove-sensory-mcp configure
uv run cove-sensory-mcp doctor
uv run cove-sensory-mcp serve
```

`status` reports redacted local configuration and credential presence. `self-test`
truthfully returns `SETUP_REQUIRED` in this foundation milestone and performs no
Provider network request:

```console
uv run cove-sensory-mcp status
uv run cove-sensory-mcp self-test
```

At the wizard's credential-reference prompt, enter an ordinary local reference to use
hidden key input and the operating-system credential store. For CI or servers, enter
`env:VARIABLE_NAME` instead; environment mode saves only the variable name, never
prompts for a key, and reports only whether a usable value is present. Portable variable
names start with an ASCII letter or underscore, contain only ASCII letters, digits, and
underscores, and are at most 128 characters.

Never paste a key into chat, an MCP tool argument, the YAML configuration file, a
command line, a test, or a snapshot. `status` and `doctor` never print key values,
references, lengths, environment-variable names, or endpoint headers. `doctor` checks
only local config readability, credential presence, temporary cache creation/removal,
and FFmpeg discovery; it does not contact a Provider.

## License

This project is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
