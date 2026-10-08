[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$VaeCheckpoint,
    [string]$FlowCheckpoint,
    [Alias("Activity")][string]$Class,
    [int]$Seed = 42,
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu",
    [string]$OutputPath,
    [string]$ExpectedVaeSha256,
    [string]$ExpectedFlowSha256,
    [switch]$DryRun
)

function Show-Usage {
    @"
Usage: run_lrf_imu.ps1 -VaeCheckpoint PATH -FlowCheckpoint PATH -Class NAME [options]

Thin wrapper around the canonical command: python -m lrf_imu generate-harth.

  -Seed N                    Random seed (default: 42)
  -Device cpu|cuda           Execution device (default: cpu)
  -Python PATH               Python launcher (default: python)
  -OutputPath PATH           Capture canonical JSON output at PATH
  -ExpectedVaeSha256 HASH    Verify VAE SHA-256 before execution
  -ExpectedFlowSha256 HASH   Verify Flow SHA-256 before execution
  -DryRun                    Validate inputs and print the canonical command
  -Help                      Show this help
"@
}

if ($Help) {
    Show-Usage
    exit 0
}
$ErrorActionPreference = "Stop"
if (-not $VaeCheckpoint -or -not $FlowCheckpoint -or -not $Class) {
    Show-Usage
    throw "VaeCheckpoint, FlowCheckpoint, and Class are required."
}
if (-not (Test-Path -LiteralPath $VaeCheckpoint -PathType Leaf)) { throw "VAE checkpoint not found: $VaeCheckpoint" }
if (-not (Test-Path -LiteralPath $FlowCheckpoint -PathType Leaf)) { throw "Flow checkpoint not found: $FlowCheckpoint" }

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$repoRoot/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
}
function Assert-Hash([string]$Path, [string]$Expected) {
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToUpperInvariant()) { throw "SHA-256 mismatch for $Path`: expected $($Expected.ToUpperInvariant()), got $actual" }
}

if ($ExpectedVaeSha256) { Assert-Hash $VaeCheckpoint $ExpectedVaeSha256 }
if ($ExpectedFlowSha256) { Assert-Hash $FlowCheckpoint $ExpectedFlowSha256 }
& $Python --version | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($DryRun) {
    Write-Output "dry_run=true"
    Write-Output "canonical=python -m lrf_imu generate-harth --flow-checkpoint $FlowCheckpoint --vae-checkpoint $VaeCheckpoint --activity $Class --seed $Seed --device $Device"
    exit 0
}
if ($Device -eq "cuda") {
    & $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the selected Python environment'"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Canonical command: python -m lrf_imu generate-harth.
$arguments = @(
    "-m", "lrf_imu", "generate-harth",
    "--flow-checkpoint", $FlowCheckpoint,
    "--vae-checkpoint", $VaeCheckpoint,
    "--activity", $Class,
    "--seed", $Seed,
    "--device", $Device
)
if ($OutputPath) {
    $parent = Split-Path -Parent $OutputPath
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $result = & $Python @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($OutputPath, (($result -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)
    $result
    if ($exitCode -ne 0) { exit $exitCode }
} else {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
