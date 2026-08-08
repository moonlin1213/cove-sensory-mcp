[CmdletBinding()]
param(
  [string]$Archive,
  [string]$Sha256,
  [string]$Version = "0.1.0",
  [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "CoveSensoryMCP\bin"),
  [switch]$ConfirmPath,
  [switch]$Uninstall,
  [switch]$RemoveData
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$fullRoot = [IO.Path]::GetFullPath($InstallRoot)
$userRoot = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
if (($fullRoot -eq $userRoot) -or -not $fullRoot.StartsWith($userRoot + [IO.Path]::DirectorySeparatorChar)) {
  throw "Install root must be a specific directory beneath LOCALAPPDATA."
}
$dataRoot = Join-Path $env:LOCALAPPDATA "CoveSensoryMCP"

if ($Uninstall) {
  if (Test-Path -LiteralPath $fullRoot) { Remove-Item -LiteralPath $fullRoot -Recurse -Force }
  if ($RemoveData -and (Test-Path -LiteralPath $dataRoot)) {
    Remove-Item -LiteralPath $dataRoot -Recurse -Force
  }
  Write-Output "Cove Sensory MCP executable removed."
  exit 0
}
if (-not [Environment]::Is64BitOperatingSystem) { throw "Windows x64 is required." }
if (-not $Archive.EndsWith("windows-x64.zip")) { throw "Archive does not match Windows x64." }
if ($Sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "A SHA-256 is required." }
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256.ToLowerInvariant()) {
  throw "Checksum mismatch."
}

New-Item -ItemType Directory -Path $fullRoot -Force | Out-Null
$staging = Join-Path $fullRoot (".staging-" + [guid]::NewGuid().ToString("N"))
$rollback = Join-Path $fullRoot ("rollback-" + $Version)
try {
  New-Item -ItemType Directory -Path $staging | Out-Null
  $archiveEntries = [IO.Compression.ZipFile]::OpenRead($Archive)
  try {
    foreach ($entry in $archiveEntries.Entries) {
      $target = [IO.Path]::GetFullPath((Join-Path $staging $entry.FullName))
      if (-not $target.StartsWith($staging + [IO.Path]::DirectorySeparatorChar)) { throw "Archive traversal." }
    }
  } finally { $archiveEntries.Dispose() }
  Expand-Archive -LiteralPath $Archive -DestinationPath $staging
  $candidate = Join-Path $staging "cove-sensory-mcp"
  if (-not (Test-Path -LiteralPath (Join-Path $candidate "cove-sensory-mcp.exe"))) { throw "Executable missing." }
  $current = Join-Path $fullRoot "current"
  if (Test-Path -LiteralPath $current) {
    if (Test-Path -LiteralPath $rollback) { Remove-Item -LiteralPath $rollback -Recurse -Force }
    Move-Item -LiteralPath $current -Destination $rollback
  }
  try { Move-Item -LiteralPath $candidate -Destination $current }
  catch {
    if (Test-Path -LiteralPath $rollback) { Move-Item -LiteralPath $rollback -Destination $current }
    throw
  }
  if ($ConfirmPath) {
    Write-Output ("Add to the current-user PATH manually: " + $current)
  }
  $exe = Join-Path $current "cove-sensory-mcp.exe"
  Write-Output ('Installed. Next: & "{0}" configure' -f $exe)
  Write-Output ('Then: & "{0}" doctor' -f $exe)
  Write-Output ('Client config: & "{0}" print-config --client generic' -f $exe)
} finally {
  if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
