[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" })
)

function Show-Usage {
    "Usage: validate_release.ps1 [-Python PATH]"
    "Runs bounded tests, compilation, diff hygiene, targeted Ruff, and CLI help checks."
}

if ($Help) { Show-Usage; exit 0 }
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "$repoRoot/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

& $Python -m compileall -q src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m ruff check `
    src/lrf_imu/integration/dayforge.py `
    src/lrf_imu/integration/physical_state.py `
    src/lrf_imu/integration/dayforge_audit.py `
    src/lrf_imu/integration/__init__.py `
    src/lrf_imu/cli.py `
    tests/test_dayforge_handoff.py `
    tests/test_dayforge_fusion.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pytest -p no:cacheprovider -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$commands = @(
    "prepare-harth-data", "train-harth-vae", "train-harth-flow", "generate-harth",
    "evaluate-harth-vae", "evaluate-harth-flow", "map-dayforge-physical-states",
    "synthesize-dayforge"
)
foreach ($command in $commands) {
    & $Python -m lrf_imu $command --help | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output "release validation passed"
