# Unattended GPU chain for the WWW run (no human labels needed).
# Recommended (judge chosen on dev):  .\scripts\run_www_gpu.ps1 -Judge nli -Scope all -ClaimScope all
# Resume after a failure:             .\scripts\run_www_gpu.ps1 -Judge nli -Scope all -ClaimScope all -From 12
# Every wgs step resumes from its own cache, so re-running a finished step is cheap.
param([int]$From = 1, [string]$Judge = 'nli', [string]$Scope = 'all', [string]$ClaimScope = 'all')
$judgeModel = @{nimble='nimble:9b'; llama='llama3.1:8b'; nli=''}[$Judge]
$C = @('--config', 'configs/www_short.yaml')
$core = @('--cohort', 'test_natural_core', 'test_diagnostic')
$ivb = @('--bundle-file', 'runs/www/data_manifests/bundles_interventions.jsonl')
$J = @('--judge', $Judge)
$steps = @(
  @{n=1;  a=@('build-candidates') + $core},
  @{n=2;  a=@('retrieve', '--kind', 'candidate')},
  @{n=3;  a=@('judge') + $J + @('--kind', 'candidate', '--scope', $Scope)},
  @{n=4;  stop=$judgeModel},
  @{n=5;  a=@('plan', '--planner', 'frequency') + $J + @('--scope', $Scope)},
  @{n=6;  a=@('plan', '--planner', 'frequency_unverified') + $J + @('--scope', $Scope)},
  @{n=7;  a=@('plan', '--planner', 'agent') + $J + @('--scope', $Scope)},
  @{n=8;  a=@('generate', '--model', 'qwen3:8b') + $core + @('--method', 'freqframe')},
  @{n=9;  a=@('generate', '--model', 'qwen3:8b') + $core + @('--method', 'plan', '--planner', 'frequency')},
  @{n=10; a=@('generate', '--model', 'qwen3:8b') + $core + @('--method', 'plan', '--planner', 'frequency_unverified')},
  @{n=11; a=@('generate', '--model', 'qwen3:8b') + $core + @('--method', 'plan', '--planner', 'agent')},
  @{n=12; a=@('agent', '--model', 'qwen3:8b', '--judge', 'nli') + $core},
  @{n=13; a=@('interventions') + $J + @('--scope', $Scope, '--cohort', 'test_diagnostic', 'test_disclosure', '--targets-per-bundle', '1')},
  @{n=14; a=@('generate', '--model', 'qwen3:8b') + $ivb + @('--method', 'direct')},
  @{n=15; a=@('generate', '--model', 'qwen3:8b') + $ivb + @('--method', 'coverage')},
  @{n=16; stop='qwen3:8b'},
  @{n=17; a=@('generate', '--model', 'gemma4:12b') + $core + @('--method', 'plan', '--planner', 'frequency')},
  @{n=18; a=@('generate', '--model', 'gemma4:12b') + $ivb + @('--method', 'direct')},
  @{n=19; stop='gemma4:12b'},
  @{n=20; a=@('annotate', 'export', '--kind', 'intervention', '--n', '150')},
  @{n=21; a=@('extract-claims')},
  @{n=22; stop='qwen3:8b'},
  @{n=23; a=@('retrieve', '--kind', 'claim')},
  @{n=24; a=@('judge') + $J + @('--scope', $ClaimScope)},
  @{n=25; a=@('metrics') + $J + @('--scope', $ClaimScope)},
  @{n=26; a=@('eval-corrected') + $J + @('--scope', $ClaimScope)},
  @{n=27; a=@('eval-gold-recall') + $J},
  @{n=28; a=@('eval-auto-recall') + $J + @('--scope', $Scope)},
  @{n=29; a=@('eval-interventions', '--auto-judge', $Judge)},
  @{n=30; a=@('annotate', 'export', '--kind', 'summary', '--n', '400')},
  @{n=31; a=@('report')},
  @{n=32; a=@('paper') + $J + @('--scope', $ClaimScope)},
  @{n=33; a=@('figures') + $J}
)
foreach ($s in $steps) {
  if ($s.n -lt $From) { continue }
  $t = Get-Date -Format 'MM-dd HH:mm'
  if ($s.ContainsKey('stop')) {
    if ($s.stop) { Write-Host "[$t] step $($s.n): ollama stop $($s.stop)"; ollama stop $s.stop }
    continue
  }
  Write-Host "[$t] step $($s.n): wgs $($s.a -join ' ')"
  python -m wgs @($s.a) @C
  if ($LASTEXITCODE -ne 0) {
    Write-Host "step $($s.n) FAILED. Fix the error, then resume with -From $($s.n) (same -Judge/-Scope options)" -ForegroundColor Red
    exit 1
  }
}
Write-Host "all steps done: $(Get-Date -Format 'MM-dd HH:mm')" -ForegroundColor Green
