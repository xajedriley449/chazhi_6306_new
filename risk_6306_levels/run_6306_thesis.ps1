param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($period in @("A","B")) {
  foreach ($task in @("low","high","dual")) {
    foreach ($seed in @(42,43,44)) {
      $out = "output_6306_levels/train_${period}_${task}_all_seed${seed}"
      if (Test-Path $out) {
        if ((Test-Path "$out/completed.json") -and (Test-Path "$out/evaluation/summary.json")) {
          Write-Host "Already complete: $out"
          continue
        }
        throw "Incomplete directory: $out. Inspect it and rename before retrying."
      }
      & $Python risk_6306_levels/run_6306_levels.py train --period $period --task $task --seed $seed --epochs 15 --output $out
      if ($LASTEXITCODE -ne 0) { throw "Training failed: $out" }
    }
  }
}
foreach ($period in @("A","B")) {
  foreach ($seed in @(42,43,44)) {
    $out = "output_6306_levels/shuffled_${period}_dual_all_seed${seed}"
    if (Test-Path $out) {
      if (Test-Path "$out/completed.json") { continue }
      throw "Incomplete directory: $out"
    }
    & $Python risk_6306_levels/run_6306_shuffled.py train --period $period --task dual --seed $seed --epochs 15 --output $out
    if ($LASTEXITCODE -ne 0) { throw "Shuffled training failed" }
  }
}
& $Python risk_6306_levels/diagnose_6306_cases.py
if ($LASTEXITCODE -ne 0) { throw "Diagnostics failed" }
& $Python risk_6306_levels/compare_6306_thesis.py
if ($LASTEXITCODE -ne 0) { throw "Summary failed" }
