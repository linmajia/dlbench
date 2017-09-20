$ErrorActionPreference = "Stop"

# Set up ENV
$BaseConfig = Join-Path -Path $PSScriptRoot -ChildPath "configs\benchmark.config"
$ConfigText = Get-Content -Path $BaseConfig
$Mode = Invoke-Expression -Command "python -m pip list" | Where { $_.StartsWith("tensorflow-gpu") }
if (($Mode -eq $Null) -or ($Mode.Length == 0)) {
    $ConfigFile = Join-Path -Path $PSScriptRoot -ChildPath "cpu.config.tmp"
    $ConfigText | where { -not $_.ToLower().StartsWith("gpu") } | Set-Content -Path $ConfigFile
} else {
    $ConfigFile = Join-Path -Path $PSScriptRoot -ChildPath "gpu.config.tmp"
    $ConfigText | where { -not $_.ToLower().StartsWith("cpu") } | Set-Content -Path $ConfigFile
}

# Run Python
$Env:PYTHONPATH = $PSScriptRoot + ';' + $Env:PYTHONPATH
$PyScript = Join-Path -Path $PSScriptRoot -ChildPath "tools\benchmark.py"
Write-Host "executing: python -u $PyScript -config $ConfigFile"
Invoke-Expression -Command "python -u $PyScript -config $ConfigFile"
Remove-Item -Path $ConfigFile -Force
