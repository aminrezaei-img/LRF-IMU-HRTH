[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$DayforgeRoot,
    [string]$DerivedRoot,
    [string]$MappingConfig = "configs/paper/dayforge_harth_mapping.yaml",
    [string]$MappingOutput,
    [switch]$RunSynthesis,
    [string]$VaeCheckpoint,
    [string]$FlowCheckpoint,
    [string]$Normalization,
    [string]$OutputPath,
    [string]$Persona,
    [string]$Date,
    [int]$Seed = 42,
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu"
)

function Show-Usage {
    @"
Usage: run_paper3_dayforge.ps1 -DayforgeRoot PATH -MappingOutput PATH [options]

Runs Module B mapping. Add -RunSynthesis for one explicitly selected person-day.
The wrapper calls: python -m lrf_imu map-dayforge-physical-states and
python -m lrf_imu synthesize-dayforge.

  -DerivedRoot PATH       Optional in-bed handoff root
  -MappingConfig PATH     Mapping YAML
  -RunSynthesis           Invoke Module C for one selected person-day
  -VaeCheckpoint PATH     Required for synthesis
  -FlowCheckpoint PATH    Required for synthesis
  -Normalization PATH     Required for synthesis
  -OutputPath PATH        Module C output directory
  -Persona ID / -Date DAY Required for synthesis
  -Seed N / -Device NAME  Reproducibility controls
  -Help                   Show this help
"@
}

if ($Help) { Show-Usage; exit 0 }
$ErrorActionPreference = "Stop"
if (-not $DayforgeRoot) { Show-Usage; throw "DayforgeRoot is required." }
if (-not (Test-Path -LiteralPath $DayforgeRoot -PathType Container)) { throw "DayForge root not found: $DayforgeRoot" }
if ($DerivedRoot -and -not (Test-Path -LiteralPath $DerivedRoot -PathType Container)) { throw "Derived root not found: $DerivedRoot" }
if (-not (Test-Path -LiteralPath $MappingConfig -PathType Leaf)) { throw "Mapping config not found: $MappingConfig" }

$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$repoRoot/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
$mapArgs = @("-m", "lrf_imu", "map-dayforge-physical-states", "--dayforge-root", $DayforgeRoot, "--config", $MappingConfig, "--output-dir", $MappingOutput)
if ($DerivedRoot) { $mapArgs += @("--derived-root", $DerivedRoot) }
& $Python @mapArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RunSynthesis) {
    if (-not $VaeCheckpoint -or -not (Test-Path -LiteralPath $VaeCheckpoint -PathType Leaf)) { throw "VaeCheckpoint is required for synthesis." }
    if (-not $FlowCheckpoint -or -not (Test-Path -LiteralPath $FlowCheckpoint -PathType Leaf)) { throw "FlowCheckpoint is required for synthesis." }
    if (-not $Normalization -or -not (Test-Path -LiteralPath $Normalization -PathType Leaf)) { throw "Normalization is required for synthesis." }
    if (-not $OutputPath -or -not $Persona -or -not $Date) { throw "OutputPath, Persona, and Date are required for synthesis." }
    $synthesisArgs = @(
        "-m", "lrf_imu", "synthesize-dayforge",
        "--dayforge-root", $DayforgeRoot,
        "--mapping-root", $MappingOutput,
        "--vae-checkpoint", $VaeCheckpoint,
        "--flow-checkpoint", $FlowCheckpoint,
        "--normalization-metadata", $Normalization,
        "--output-dir", $OutputPath,
        "--persona", $Persona,
        "--date", $Date,
        "--seed", $Seed,
        "--device", $Device
    )
    & $Python @synthesisArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
