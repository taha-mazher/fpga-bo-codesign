# Overnight search: GP BO, random and NSGA-II, 3 seeds, 40 designs each, then the report. Keeps Windows awake while it runs.
# Usage (PowerShell): .\run_bo.ps1        log: out\bo_run.log
param([int]$N = 40)
Set-Location $PSScriptRoot
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$log = "out\bo_run.log"
foreach ($seed in 0, 1, 2) {
    foreach ($m in "gp", "random", "nsga2") {
        "=== $m seed $seed  $(Get-Date -Format s)" | Add-Content $log
        cmd /c "python bo.py run $m $seed $N >> $log 2>&1"
    }
}
cmd /c "python bo.py report >> $log 2>&1"
"RUN_DONE $(Get-Date -Format s)" | Add-Content $log
[Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null
