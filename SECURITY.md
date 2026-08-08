# Security policy

## Supported versions

Security fixes are provided for the latest published minor release. Pre-release builds
are for evaluation and should not process sensitive media.

## Reporting

Please use the repository host's private security-advisory channel. Do not open a public
issue containing a secret, private media, exploit payload, local path, or Provider
response. The public repository intentionally contains no personal contact address.

## Guarantees and threat model

Credentials are accepted only by local hidden input, an environment reference, or the
OS credential store. MCP arguments, status, diagnostics, logs, configuration, and
self-test reports do not reveal key values. Media parsers, Providers, remote URLs, and
archives are untrusted boundaries: local roots are allowlisted, URL resolution blocks
private networks and redirects are revalidated, processing is bounded, and temporary
artifacts are removed. Provider transmission is explicit and subject to that provider's
retention, jurisdiction, and billing.

This project does not sandbox the host OS, inspect arbitrary web pages, monitor devices,
guarantee a Provider's claims, or protect media after an authorized Provider receives it.
