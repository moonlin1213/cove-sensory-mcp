#!/usr/bin/env bash
set -euo pipefail

VERSION="0.1.0"
INSTALL_ROOT="${COVE_INSTALL_ROOT:-${HOME}/Library/Application Support/cove-sensory-mcp/bin}"
ARCHIVE=""
CHECKSUM=""
ACTION="install"
REMOVE_DATA="false"
CONFIRM_PATH="false"

while (($#)); do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --sha256) CHECKSUM="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --confirm-path) CONFIRM_PATH="true"; shift ;;
    --uninstall) ACTION="uninstall"; shift ;;
    --remove-data) REMOVE_DATA="true"; shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$INSTALL_ROOT" in
  ""|"/"|"${HOME}"|"${HOME}/"|"/Users"|"/Applications")
    printf 'Refusing an unsafe install root.\n' >&2; exit 2 ;;
esac
case "$INSTALL_ROOT" in
  "${HOME}"/*) ;;
  *) printf 'Install root must be inside the current user home.\n' >&2; exit 2 ;;
esac

CONFIG_ROOT="${HOME}/Library/Application Support/cove-sensory-mcp"
if [[ "$ACTION" == "uninstall" ]]; then
  [[ -d "$INSTALL_ROOT" ]] && rm -rf -- "$INSTALL_ROOT"
  if [[ "$REMOVE_DATA" == "true" && -d "$CONFIG_ROOT" ]]; then
    case "$CONFIG_ROOT" in "${HOME}"/*/cove-sensory-mcp) rm -rf -- "$CONFIG_ROOT" ;; esac
  fi
  printf 'Cove Sensory MCP executable removed.\n'
  exit 0
fi

[[ -f "$ARCHIVE" && "$CHECKSUM" =~ ^[0-9a-fA-F]{64}$ ]] || {
  printf 'Provide --archive and its 64-character --sha256.\n' >&2; exit 2;
}
machine="$(uname -m)"
case "$machine" in arm64|aarch64) platform_tag="macos-arm64" ;; x86_64) platform_tag="macos-x64" ;; *) exit 2 ;; esac
[[ "$(basename "$ARCHIVE")" == *"${platform_tag}"* ]] || {
  printf 'Archive does not match this Mac architecture.\n' >&2; exit 2;
}
actual="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
expected="$(printf '%s' "$CHECKSUM" | tr '[:upper:]' '[:lower:]')"
[[ "$actual" == "$expected" ]] || { printf 'Checksum mismatch.\n' >&2; exit 1; }
tar -tzf "$ARCHIVE" | while IFS= read -r member; do
  [[ "$member" != /* && "$member" != *"../"* ]] || exit 91
done

mkdir -p "$INSTALL_ROOT"
staging="$(mktemp -d "${INSTALL_ROOT}/.staging.XXXXXX")"
rollback="${INSTALL_ROOT}/rollback-${VERSION}"
cleanup() { [[ -d "$staging" ]] && rm -rf -- "$staging"; }
trap cleanup EXIT
tar -xzf "$ARCHIVE" -C "$staging"
candidate="${staging}/cove-sensory-mcp"
[[ -x "${candidate}/cove-sensory-mcp" ]] || { printf 'Executable missing.\n' >&2; exit 1; }
if [[ -d "${INSTALL_ROOT}/current" ]]; then
  [[ -d "$rollback" ]] && rm -rf -- "$rollback"
  mv "${INSTALL_ROOT}/current" "$rollback"
fi
if ! mv "$candidate" "${INSTALL_ROOT}/current"; then
  [[ -d "$rollback" ]] && mv "$rollback" "${INSTALL_ROOT}/current"
  exit 1
fi

if [[ "$CONFIRM_PATH" == "true" ]]; then
  printf 'Add this directory to PATH manually: %s\n' "${INSTALL_ROOT}/current"
fi
exe="${INSTALL_ROOT}/current/cove-sensory-mcp"
printf 'Installed for the current user. Next:\n'
printf '  %q configure\n' "$exe"
printf '  %q doctor\n' "$exe"
printf '  %q print-config --client generic\n' "$exe"
