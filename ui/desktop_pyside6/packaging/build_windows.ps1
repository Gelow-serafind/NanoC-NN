$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path "$PSScriptRoot\..\..\.."
Set-Location $RootDir

python -m PyInstaller `
  --clean `
  --noconfirm `
  ui\desktop_pyside6\packaging\NanoC-NN.spec

Write-Host ""
Write-Host "Built: $RootDir\dist\NanoC-NN"
