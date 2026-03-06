# Activate venv myenv011 and run lung pipeline: report -> NLP -> TotalSegmentator
# Run from project root: .\scripts\run_lung_pipeline_myenv011.ps1
# Or with CT path: .\scripts\run_lung_pipeline_myenv011.ps1 "D:\project\medical_3Dmodel\manifest-1585232716547\LIDC-IDRI\LIDC-IDRI-0004"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$venv = Join-Path $ProjectRoot "myenv011\Scripts\Activate.ps1"
if (-not (Test-Path $venv)) {
    $venv = Join-Path $ProjectRoot "myenv001\Scripts\Activate.ps1"
    if (Test-Path $venv) { Write-Host "myenv011 not found, using myenv001" }
}
if (Test-Path $venv) {
    & $venv
}

$script = Join-Path $ProjectRoot "scripts\run_lung_report_nlp_totalseg.py"
$args = $args
if ($args) {
    & python $script @args
} else {
    & python $script
}
exit $LASTEXITCODE
