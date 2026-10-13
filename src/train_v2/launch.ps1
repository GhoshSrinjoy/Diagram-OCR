# Open Tadpole training in its own terminal window (live dashboard + logs).
#   .\launch.ps1            full run (resumes automatically if a checkpoint exists)
#   .\launch.ps1 -Smoke     8-step pipeline check
#   .\launch.ps1 -Module train_v2.rl -Config train_v2\config_rl.yaml   RL stage
param([switch]$Smoke, [string]$Config = "", [string]$Module = "")

$runner = Join-Path $PSScriptRoot "run_training.ps1"
$runnerArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$runner`"")
if ($Smoke) { $runnerArgs += "-Smoke" }
if ($Config) { $runnerArgs += @("-Config", "`"$Config`"") }
if ($Module) { $runnerArgs += @("-Module", $Module) }
$title = if ($Smoke) { "Tadpole training (smoke)" } else { "Tadpole training" }

# Pass a single script file: wt.exe treats ';' in a command line as its own separator.
if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
    Start-Process wt.exe -ArgumentList (@("new-tab", "--title", "`"$title`"", "powershell.exe") + $runnerArgs)
} else {
    Start-Process powershell.exe -ArgumentList $runnerArgs
}
