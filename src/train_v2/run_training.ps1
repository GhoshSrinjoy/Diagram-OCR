# Runs training in the current console (launch.ps1 opens a new window running this file).
param([switch]$Smoke, [string]$Config = "", [string]$Module = "train_v2")

$python = "D:\envs\diagram\python.exe"
$src = Split-Path -Parent $PSScriptRoot
$title = if ($Smoke) { "Diagram IR training (smoke)" } else { "Diagram IR training" }
$host.UI.RawUI.WindowTitle = $title
Set-Location $src

$trainArgs = @("-m", $Module)
if ($Smoke) { $trainArgs += "--smoke" }
if ($Config) { $trainArgs += @("--config", $Config) }

& $python @trainArgs
$code = $LASTEXITCODE

# Keep the window open so the final dashboard and any error stay readable.
Write-Host ""
Write-Host "Training process exited with code $code. Logs and checkpoints: D:\data\diagram_extraction\runs"
Read-Host "Press Enter to close"
