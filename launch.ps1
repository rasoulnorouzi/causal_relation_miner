# Causal Relationship Extractor - PowerShell Launcher
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Causal Relationship Extractor" -ForegroundColor Green
Write-Host "  Starting Streamlit App..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\Users\norouzin\Desktop\prototype"

& "C:\Users\norouzin\Desktop\prototype\myenv\Scripts\python.exe" -m streamlit run app.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
