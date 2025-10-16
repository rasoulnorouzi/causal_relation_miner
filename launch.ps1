<#
 Causal Relationship Extractor - PowerShell Launcher
 Uses the venv in ./myenv and runs Streamlit from this repo folder.
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Causal Relationship Extractor" -ForegroundColor Green
Write-Host "  Starting Streamlit App..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Move to the folder where this script resides
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot "myenv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
	Write-Host "⚠️  Could not find venv Python at: $venvPython" -ForegroundColor Yellow
	Write-Host "   Falling back to system 'python'..." -ForegroundColor Yellow
	$venvPython = "python"
}

# Launch Streamlit app
& $venvPython -m streamlit run app.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
