# Evidence agent v2.  Phase 1 (dev): build dev candidates, run K in {0,2,4,6}, choose K WITHOUT recall.
#                     Phase 2 (test): run the chosen K once per generator, then re-evaluate.
#   .\scripts\run_agent2.ps1 -Phase dev
#   .\scripts\run_agent2.ps1 -Phase test -KQwen 4 -KGemma 4      (K values printed by phase dev)
param([string]$Phase = 'dev', [int]$KQwen = -1, [int]$KGemma = -1, [int]$From = 1)
$C = @('--config', 'configs/www_short.yaml')
$core = @('--cohort', 'test_natural_core', 'test_diagnostic')
if ($Phase -eq 'dev') {
  $steps = @(
    @{n=1;  a=@('extract-experiences', '--split', 'dev')},
    @{n=2;  a=@('build-candidates', '--split', 'dev')},
    @{n=3;  a=@('retrieve', '--kind', 'candidate')},
    @{n=4;  a=@('judge', '--judge', 'nli', '--kind', 'candidate', '--scope', 'all')},
    @{n=5;  a=@('plan', '--planner', 'agent', '--judge', 'nli', '--scope', 'all')},
    @{n=6;  a=@('agent2', '--model', 'qwen3:8b', '--k', '0', '--split', 'dev')},
    @{n=7;  a=@('agent2', '--model', 'qwen3:8b', '--k', '2', '--split', 'dev')},
    @{n=8;  a=@('agent2', '--model', 'qwen3:8b', '--k', '4', '--split', 'dev')},
    @{n=9;  a=@('agent2', '--model', 'qwen3:8b', '--k', '6', '--split', 'dev')},
    @{n=10; stop='qwen3:8b'},
    @{n=11; a=@('agent2', '--model', 'gemma4:12b', '--k', '0', '--split', 'dev')},
    @{n=12; a=@('agent2', '--model', 'gemma4:12b', '--k', '2', '--split', 'dev')},
    @{n=13; a=@('agent2', '--model', 'gemma4:12b', '--k', '4', '--split', 'dev')},
    @{n=14; a=@('agent2', '--model', 'gemma4:12b', '--k', '6', '--split', 'dev')},
    @{n=15; stop='gemma4:12b'},
    @{n=16; a=@('extract-claims')},
    @{n=17; stop='qwen3:8b'},
    @{n=18; a=@('retrieve', '--kind', 'claim')},
    @{n=19; a=@('judge', '--judge', 'nli', '--scope', 'all')},
    @{n=20; a=@('metrics', '--judge', 'nli', '--scope', 'all')},
    @{n=21; a=@('agent2-choose-k', '--model', 'qwen3:8b')},
    @{n=22; a=@('agent2-choose-k', '--model', 'gemma4:12b')}
  )
} else {
  if ($KQwen -lt 0 -or $KGemma -lt 0) { Write-Host 'give -KQwen and -KGemma from phase dev' -ForegroundColor Red; exit 1 }
  $steps = @(
    @{n=1;  a=@('agent2', '--model', 'qwen3:8b', '--k', "$KQwen", '--split', 'test') + $core},
    @{n=2;  stop='qwen3:8b'},
    @{n=3;  a=@('agent2', '--model', 'gemma4:12b', '--k', "$KGemma", '--split', 'test') + $core},
    @{n=4;  stop='gemma4:12b'},
    @{n=5;  a=@('extract-claims')},
    @{n=6;  stop='qwen3:8b'},
    @{n=7;  a=@('retrieve', '--kind', 'claim')},
    @{n=8;  a=@('judge', '--judge', 'nli', '--scope', 'all')},
    @{n=9;  a=@('metrics', '--judge', 'nli', '--scope', 'all')},
    @{n=10; a=@('eval-corrected', '--judge', 'nli', '--scope', 'all')},
    @{n=11; a=@('eval-gold-recall', '--judge', 'nli')},
    @{n=12; a=@('eval-auto-recall', '--judge', 'nli', '--scope', 'all')},
    @{n=13; a=@('paper', '--judge', 'nli', '--scope', 'all')},
    @{n=14; a=@('figures', '--judge', 'nli')}
  )
}
foreach ($s in $steps) {
  if ($s.n -lt $From) { continue }
  $t = Get-Date -Format 'MM-dd HH:mm'
  if ($s.ContainsKey('stop')) { Write-Host "[$t] step $($s.n): ollama stop $($s.stop)"; ollama stop $s.stop; continue }
  Write-Host "[$t] step $($s.n): wgs $($s.a -join ' ')"
  python -m wgs @($s.a) @C
  if ($LASTEXITCODE -ne 0) { Write-Host "step $($s.n) FAILED; resume with -From $($s.n)" -ForegroundColor Red; exit 1 }
}
Write-Host "phase $Phase done: $(Get-Date -Format 'MM-dd HH:mm')" -ForegroundColor Green
