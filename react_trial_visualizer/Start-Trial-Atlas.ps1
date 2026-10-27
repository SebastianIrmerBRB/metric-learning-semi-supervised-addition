#requires -Version 5.1

<#
.SYNOPSIS
Builds and starts Trial Atlas with its local TensorBoard data service.

.DESCRIPTION
The React site is exported once, then served by the Python companion service on
the same origin as the TensorBoard JSON API. Only one local process and one port
are needed, regardless of how many trials or folds are explored.
#>

[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,

    [switch]$SkipBuild,

    [switch]$OpenBrowser,

    [switch]$PassThru
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectDirectory = $PSScriptRoot
$repositoryDirectory = Split-Path -Parent $projectDirectory
$python = Join-Path $repositoryDirectory '.venv\Scripts\python.exe'
$service = Join-Path $projectDirectory 'tensorboard_api.py'
$tensorboardRoot = Join-Path $repositoryDirectory 'tensorboard-files'
$staticDirectory = Join-Path $projectDirectory 'out'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The repository Python environment was not found: $python"
}
if (-not (Test-Path -LiteralPath $tensorboardRoot -PathType Container)) {
    throw "The TensorBoard root was not found: $tensorboardRoot"
}

if (-not $SkipBuild) {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $npm) {
        throw 'npm.cmd was not found on PATH.'
    }
    Push-Location $projectDirectory
    try {
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) {
            throw "The Trial Atlas production build failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $staticDirectory 'index.html') -PathType Leaf)) {
    throw "The static website is missing below $staticDirectory. Run without -SkipBuild first."
}

$occupied = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
    Where-Object Port -eq $Port |
    Select-Object -First 1
if ($null -ne $occupied) {
    throw "Port $Port is already in use. Choose another value with -Port."
}

$arguments = @(
    "`"$service`"",
    "--root=`"$tensorboardRoot`"",
    "--static-dir=`"$staticDirectory`"",
    '--host=127.0.0.1',
    "--port=$Port"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repositoryDirectory -WindowStyle Hidden -PassThru
$url = "http://127.0.0.1:$Port/"
$healthUrl = "${url}health"
$ready = $false

for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($process.HasExited) {
        throw "Trial Atlas exited during startup with code $($process.ExitCode)."
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 1
        if ($health.status -eq 'ok') {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 250
    }
}

if (-not $ready) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw 'Trial Atlas did not become ready in time.'
}

Write-Host "Trial Atlas is ready: $url"
Write-Host "Process ID: $($process.Id)"
Write-Host "Stop it with: Stop-Process -Id $($process.Id)"

if ($OpenBrowser) {
    Start-Process -FilePath $url | Out-Null
}

if ($PassThru) {
    [pscustomobject]@{
        URL = $url
        PID = $process.Id
        TensorboardRoot = $tensorboardRoot
        StaticDirectory = $staticDirectory
    }
}
