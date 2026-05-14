# Serves the simulation and accepts POST /api/save-capture to write PNGs under Research/ACADIA-2026/data
Set-Location $PSScriptRoot
Write-Host "Starting capture server (see serve_capture.py for paths)."
python .\serve_capture.py
