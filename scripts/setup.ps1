# GridPulse one-shot setup for Windows PowerShell.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "`n=== Creating virtual environment ===" -ForegroundColor Cyan
if (-Not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

Write-Host "`n=== Installing dependencies ===" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-deps

if (-Not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "`nCreated .env - add your EIA_API_KEY before continuing." -ForegroundColor Yellow
    Write-Host "Free key: https://www.eia.gov/opendata/register.php" -ForegroundColor Yellow
}

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Next:  gridpulse probe   then   gridpulse all"
