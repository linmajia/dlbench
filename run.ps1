$ErrorActionPreference = "Stop"

# Set up ENV
$LogDir = Join-Path -Path $PSScriptRoot -ChildPath "logs"
$TimeStr = (Get-Date).toUniversalTime().ToString("yyyy-MM-dd-HH-mm-ss")
$LogDir = Join-Path -Path $LogDir -ChildPath "dlbench_$TimeStr"
$ConfigFile = Join-Path -Path $LogDir -ChildPath "benchmark.config"
New-Item -Path $LogDir -ItemType "directory" -Force | Out-Null
$BaseConfig = Join-Path -Path $PSScriptRoot -ChildPath "configs\benchmark.config"
$ConfigText = Get-Content -Path $BaseConfig
$Mode = Invoke-Expression -Command "python -m pip list" | Where { $_.StartsWith("tensorflow-gpu") }
if (($Mode -eq $Null) -or ($Mode.Length == 0)) {
    $ConfigText | where { -not $_.ToLower().StartsWith("gpu") } | Set-Content -Path $ConfigFile
} else {
    $ConfigText | where { -not $_.ToLower().StartsWith("cpu") } | Set-Content -Path $ConfigFile
}

# Run Python
$Env:PYTHONPATH = $PSScriptRoot + ';' + $Env:PYTHONPATH
$PyScript = Join-Path -Path $PSScriptRoot -ChildPath "benchmark.py"
Write-Host "executing: python -u $PyScript -config $ConfigFile -result $LogDir"
Invoke-Expression -Command "python -u $PyScript -config $ConfigFile -result $LogDir"
