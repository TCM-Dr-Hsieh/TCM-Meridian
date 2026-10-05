$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host 'The Python environment .venv was not found. Run install.cmd (or .\setup.ps1) first; see README.md.'
    exit 1
}
# One-click start does not use NiceGUI auto-reload (a development feature). The environment variable and the current
# location are restored afterwards so they do not leak into an already-open PowerShell session.
$previousReload = $env:TCM_RELOAD
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:TCM_RELOAD = '0'
    & $python TCM_Meridian_main.py
    $exitCode = $LASTEXITCODE
} finally {
    $env:TCM_RELOAD = $previousReload
    Pop-Location
}
exit $exitCode
