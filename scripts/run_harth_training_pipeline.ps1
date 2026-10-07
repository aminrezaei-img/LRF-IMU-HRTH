[CmdletBinding()]
param(
    [switch]$Help,
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$DataRoot,
    [string]$OutputRoot,
    [string]$Config = "configs/paper/harth_10class_160_40.yaml",
    [string]$HeldOutSubject = "harth:S006",
    [int]$Seed = 42,
    [int]$VaeEpochs = 0,
    [int]$FlowEpochs = 0,
    [int]$MaxTrainBatches = 0,
    [int]$MaxValBatches = 0,
    [switch]$DryRun
)

function Show-Usage {
    @"
Usage: run_harth_training_pipeline.ps1 -DataRoot PATH -OutputRoot PATH [options]

Runs the canonical model-only sequence:
  prepare-harth-data -> train-harth-vae -> train-harth-flow

  -Config PATH              Frozen HARTH configuration
  -HeldOutSubject ID        LOSO subject (default: harth:S006)
  -Seed N                   Global seed (default: 42)
  -VaeEpochs N              Optional VAE epoch override
  -FlowEpochs N             Optional Flow epoch override
  -MaxTrainBatches N        Optional bounded smoke override
  -MaxValBatches N          Optional bounded smoke override
  -Python PATH              Python launcher
  -DryRun                   Print commands without writing or training
  -Help                     Show this help
"@
}

# This wrapper delegates every scientific step to: python -m lrf_imu <command>.
if ($Help) { Show-Usage; exit 0 }
$ErrorActionPreference = "Stop"
if (-not $DataRoot -or -not $OutputRoot) {
    Show-Usage
    throw "DataRoot and OutputRoot are required."
}
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Data root not found: $DataRoot"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $repoRoot $Config }
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Configuration not found: $configPath"
}
$env:PYTHONPATH = "$repoRoot/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$vaeOutput = Join-Path $OutputRoot "vae"
$flowOutput = Join-Path $OutputRoot "flow"
$vaeCheckpoint = Join-Path $vaeOutput "vae_s3_z48.pt"

$prepare = @(
    "-m", "lrf_imu", "prepare-harth-data",
    "--data-root", $DataRoot,
    "--composition", "harth_walking_speed",
    "--held-out-subject", $HeldOutSubject,
    "--window-length", "160",
    "--hop-length", "40",
    "--seed", "$Seed"
)
$vae = @(
    "-m", "lrf_imu", "train-harth-vae",
    "--data-root", $DataRoot,
    "--composition", "harth_walking_speed",
    "--held-out-subject", $HeldOutSubject,
    "--config", $configPath,
    "--output-dir", $vaeOutput,
    "--seed", "$Seed"
)
$flow = @(
    "-m", "lrf_imu", "train-harth-flow",
    "--data-root", $DataRoot,
    "--composition", "harth_walking_speed",
    "--held-out-subject", $HeldOutSubject,
    "--config", $configPath,
    "--vae-checkpoint", $vaeCheckpoint,
    "--output-dir", $flowOutput,
    "--seed", "$Seed"
)
if ($VaeEpochs -gt 0) { $vae += @("--epochs", "$VaeEpochs") }
if ($FlowEpochs -gt 0) { $flow += @("--epochs", "$FlowEpochs") }
if ($MaxTrainBatches -gt 0) {
    $vae += @("--max-train-batches", "$MaxTrainBatches")
    $flow += @("--max-train-batches", "$MaxTrainBatches")
}
if ($MaxValBatches -gt 0) {
    $vae += @("--max-val-batches", "$MaxValBatches")
    $flow += @("--max-val-batches", "$MaxValBatches")
}

if ($DryRun) {
    Write-Output "dry_run=true"
    Write-Output ("prepare=" + ($prepare -join " "))
    Write-Output ("vae=" + ($vae -join " "))
    Write-Output ("flow=" + ($flow -join " "))
    exit 0
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
& $Python @prepare
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python @vae
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path -LiteralPath $vaeCheckpoint -PathType Leaf)) {
    throw "VAE training completed without the expected checkpoint: $vaeCheckpoint"
}
& $Python @flow
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$flowCheckpoint = Join-Path $flowOutput "flow_unet_best.pt"
if (-not (Test-Path -LiteralPath $flowCheckpoint -PathType Leaf)) {
    throw "Flow training completed without the expected checkpoint: $flowCheckpoint"
}
Write-Output "training_pipeline=complete"
Write-Output "vae_checkpoint=$vaeCheckpoint"
Write-Output "flow_checkpoint=$flowCheckpoint"
