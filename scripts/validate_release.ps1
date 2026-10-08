[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" })
)

function Show-Usage {
    "Usage: validate_release.ps1 [-Python PATH]"
    "Runs bounded tests, syntax parsing, diff hygiene, targeted Ruff, and CLI help checks."
}

if ($Help) { Show-Usage; exit 0 }
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "$repoRoot/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

& $Python -B scripts/check_syntax.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m ruff check `
    src/lrf_imu/artifacts.py `
    src/lrf_imu/cli.py `
    scripts/check_syntax.py `
    tests/test_core_release_boundary.py `
    tests/test_production_artifacts.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONDONTWRITEBYTECODE = "1"
& $Python -B -m pytest -p no:cacheprovider -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$commands = @(
    "prepare-harth-data", "train-harth-vae", "train-harth-flow", "generate-harth",
    "evaluate-harth-vae", "evaluate-harth-flow", "fetch-production-artifacts",
    "verify-production-artifacts"
)
foreach ($command in $commands) {
    & $Python -m lrf_imu $command --help | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output "release validation passed"
