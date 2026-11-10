# run_pipeline_v6_2_10.ps1
# AIC pipeline driver for v6.2.10
#
# Scope:
#   Regenerates predictions, external statistics, Figure 2, Figure S2,
#   recalibration, DCA, supplementary tables, main tables, and the
#   closure report. It does NOT re-run eICU validation, APACHE IVa
#   comparison, or the SOFA comparison when their inputs are unchanged.
#   If any upstream product is missing or stale, run eicu_build_wide.py /
#   eicu_validate.py / compare_eicu_apache.py and run_mimic_sofa_all.py
#   manually first.
#
# v6.2.10 changes vs v6.2.9:
#   - pipeline renamed to run_pipeline_v6_2_10.ps1
#   - requiredScripts includes shared_calib.py
#   - pre-check aborts if shared_calib.py is not v1.3
#   - step order: run_model -> make_external_stats -> fig2_be -> figS2
#
# Usage:
#   .\run_pipeline_v6_2_10.ps1
#   .\run_pipeline_v6_2_10.ps1 -SkipSofa
#   .\run_pipeline_v6_2_10.ps1 -SkipClosureReport

param(
    [string]$ProjectBase = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$MimicWideCsv = 'D:\mimic_wide_with_hco3.csv',
    [string]$MimicPath = 'D:\mimic-iv-3.1',
    [string]$SofaCsv = 'D:\mimic-code\results\mimic_sofa_official.csv',
    [string]$PythonExe = 'C:\Users\24339\AppData\Local\Programs\Python\Python310\python.exe',
    [switch]$SkipSofa,
    [switch]$SkipClosureReport
)

$ErrorActionPreference = 'Stop'

$ExpectedMimicSha256 = 'b151612f75c568fce52f5a472206608682daaecd1a8ae2302595c45a0b3a72eb'

$env:PROJECT_BASE  = $ProjectBase
$env:MIMIC_WIDE_CSV = $MimicWideCsv
$env:MIMIC_PATH    = $MimicPath
$env:SOFA_CSV      = $SofaCsv

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ManifestPath = Join-Path $ProjectBase "results\pipeline_run_manifest_$Stamp.txt"
$ManifestLines = New-Object System.Collections.Generic.List[string]

function Log([string]$msg) {
    Write-Host $msg
    $script:ManifestLines.Add($msg) | Out-Null
}

Log "AIC pipeline v6.2.10"
Log "Run started: $Stamp"
Log "ProjectBase  : $ProjectBase"
Log "MimicWideCsv : $MimicWideCsv"
Log "MimicPath    : $MimicPath"
Log "SofaCsv      : $SofaCsv"
Log "PythonExe    : $PythonExe"
Log "SkipSofa     : $SkipSofa"
Log "SkipClosure  : $SkipClosureReport"
Log ""

Log "========== Pre-checks =========="

if (-not (Test-Path $ProjectBase)) {
    throw "ProjectBase not found: $ProjectBase"
}
Push-Location $ProjectBase

try {
    if (-not (Test-Path $PythonExe)) {
        throw "Python not found: $PythonExe"
    }
    $pyVer = & $PythonExe -c "import sys; print(sys.version.split()[0])"
    Log "[OK] Python $pyVer"
    Log "[OK] Python path: $PythonExe"

    $requiredScripts = @(
        'ci_config.py',
        'shared_calib.py',
        'make_external_stats.py',
        'run_model.py',
        'fig2_be.py',
        'figS2_hco3.py',
        'recal_bootstrap.py',
        'run_mimic_sofa_all.py',
        'make_eicu_supp_tables.py',
        'make_tables.py',
        'check_v6_2_1.py',
        'make_closure_report.py'
    )
    foreach ($s in $requiredScripts) {
        if (-not (Test-Path (Join-Path $ProjectBase $s))) {
            throw "Required script not found: $s"
        }
        Log "[OK] script present: $s"
    }

    # shared_calib.py must be v1.3 (outcome-stratified bootstrap_ci_auc_brier)
    $sc = Get-Content (Join-Path $ProjectBase 'shared_calib.py') -Raw
    if ($sc -notmatch 'v1\.3') {
        throw "shared_calib.py is not v1.3. Replace it before running the pipeline."
    }
    Log "[OK] shared_calib.py is v1.3"

    if (-not (Test-Path $MimicWideCsv)) {
        throw "MIMIC wide CSV not found: $MimicWideCsv"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -Path $MimicWideCsv).Hash.ToLower()
    if ($actualHash -ne $ExpectedMimicSha256) {
        throw @"
MIMIC wide SHA256 mismatch.
  expected: $ExpectedMimicSha256
  actual  : $actualHash
  path    : $MimicWideCsv
Update EXPECTED_MIMIC_SHA256 in run_model.py, recal_bootstrap.py,
make_external_stats.py, and this script before rerunning.
"@
    }
    Log "[OK] MIMIC wide SHA256 verified: $actualHash"

    $ciInfo = & $PythonExe -c @"
from ci_config import CI_SEED, N_BOOT_CI, CI_METHOD, CI_LEVEL
print(f'{CI_SEED}|{N_BOOT_CI}|{CI_METHOD}|{CI_LEVEL}')
"@
    $ciParts = $ciInfo -split '\|'
    Log "[OK] CI config: seed=$($ciParts[0])  n_boot=$($ciParts[1])  method=$($ciParts[2])  level=$($ciParts[3])"

    Log ""

    $steps = New-Object System.Collections.Generic.List[object]

    $steps.Add(@{ Name = 'run_model.py';           Exe = $PythonExe; Args = @('run_model.py') })
    $steps.Add(@{ Name = 'make_external_stats.py'; Exe = $PythonExe; Args = @('make_external_stats.py') })
    $steps.Add(@{ Name = 'fig2_be.py';             Exe = $PythonExe; Args = @('fig2_be.py') })
    $steps.Add(@{ Name = 'figS2_hco3.py';          Exe = $PythonExe; Args = @('figS2_hco3.py') })
    $steps.Add(@{ Name = 'recal_bootstrap.py';     Exe = $PythonExe; Args = @('recal_bootstrap.py') })

    if (-not $SkipSofa) {
        $steps.Add(@{ Name = 'run_mimic_sofa_all.py'; Exe = $PythonExe; Args = @('run_mimic_sofa_all.py') })
    } else {
        Log "[skip] run_mimic_sofa_all.py (--SkipSofa)"
    }

    $steps.Add(@{ Name = 'make_eicu_supp_tables.py'; Exe = $PythonExe; Args = @('make_eicu_supp_tables.py') })
    $steps.Add(@{ Name = 'make_tables.py';           Exe = $PythonExe; Args = @('make_tables.py') })
    $steps.Add(@{ Name = 'check_v6_2_1.py';          Exe = $PythonExe; Args = @('check_v6_2_1.py') })

    if (-not $SkipClosureReport) {
        $steps.Add(@{ Name = 'make_closure_report.py'; Exe = $PythonExe; Args = @('make_closure_report.py') })
    } else {
        Log "[skip] make_closure_report.py (--SkipClosureReport)"
    }

    $stepNum = 0
    $total   = $steps.Count

    foreach ($step in $steps) {
        $stepNum++
        $t0 = Get-Date
        Log ""
        Log "========== [$stepNum/$total] $($step.Name) =========="
        Log "started: $($t0.ToString('yyyy-MM-dd HH:mm:ss'))"

        & $step.Exe @($step.Args)
        $exit = $LASTEXITCODE
        $t1 = Get-Date
        $elapsed = [math]::Round(($t1 - $t0).TotalSeconds, 1)

        if ($exit -ne 0) {
            Log "FAILED: $($step.Name) exit=$exit elapsed=${elapsed}s"
            Log "Downstream steps NOT run."
            $ManifestLines | Set-Content -Path $ManifestPath -Encoding UTF8
            throw "Step failed: $($step.Name) (exit code $exit). Manifest: $ManifestPath"
        }
        Log "OK: $($step.Name) elapsed=${elapsed}s"
    }

    Log ""
    Log "========== Artifact SHA256 =========="

    $artifacts = @(
        'results\external_auc_stats.csv',
        'results\external_predictions.csv',
        'results\external_predictions_hco3_v2.csv',
        'results\mimic_prediction_manifest.csv',
        'results\recal_summary_dual.csv',
        'results\dca_results_dual.csv',
        'results\mimic_sofa_vs_model.csv',
        'results\TableS6_this_study_row.csv',
        'results\TableS13_eicu_alb_measured_vs_not.csv',
        'tables\Table3.csv',
        'tables\Table4.csv',
        'tables\Table5.csv',
        'tables\TableS14.csv'
    )
    foreach ($rel in $artifacts) {
        $p = Join-Path $ProjectBase $rel
        if (Test-Path $p) {
            $h = (Get-FileHash -Algorithm SHA256 -Path $p).Hash.ToLower()
            Log "$rel  $h"
        } else {
            Log "$rel  MISSING"
        }
    }

    Log ""
    Log "========== Final summary =========="
    Log "Run finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Log "All $total steps completed successfully."

    $ManifestLines | Set-Content -Path $ManifestPath -Encoding UTF8
    Log ""
    Log "Manifest written: $ManifestPath"
    Write-Host "`nPipeline v6.2.10 completed. Manifest: $ManifestPath" -ForegroundColor Green
}
finally {
    Pop-Location
}