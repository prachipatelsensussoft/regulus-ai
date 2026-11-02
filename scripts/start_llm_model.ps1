# Start the llm-model API on Windows (scripts/start_llm_model.sh's counterpart): scripts/serve_llm_model.py with
# Qwen3-4B-Instruct-2507 (the MLX 4-bit directory, run by PyTorch: scripts/torch_worker.py) and the production pipeline.
# Every machine-specific value comes from configs/llm_model.env: copy configs/llm_model.env.example and fill it in.
# Check it with:  curl http://127.0.0.1:8100/llm-model/v1/ready      Stop with Ctrl+C.
#   powershell -ExecutionPolicy Bypass -File scripts\start_llm_model.ps1 [extra serve_llm_model.py flags]
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$profilePath = if ($env:LLM_MODEL_PROFILE) { $env:LLM_MODEL_PROFILE } else { "configs\llm_model.env" }
if (-not (Test-Path $profilePath)) {
    Write-Error "missing ${profilePath}: copy configs\llm_model.env.example to configs\llm_model.env and fill it in"
    exit 1
}
. "$PSScriptRoot\load_profile.ps1" $profilePath
$hostName = if ($env:LLM_MODEL_HOST) { $env:LLM_MODEL_HOST } else { "127.0.0.1" }
$port = if ($env:LLM_MODEL_PORT) { $env:LLM_MODEL_PORT } else { "8100" }
$tls = @()
if ($env:LLM_MODEL_TLS_CERT) { $tls += @("--tls-cert", $env:LLM_MODEL_TLS_CERT) }
if ($env:LLM_MODEL_TLS_KEY) { $tls += @("--tls-key", $env:LLM_MODEL_TLS_KEY) }
& .\.venv\Scripts\python.exe scripts\serve_llm_model.py --host $hostName --port $port @tls --offline @args
exit $LASTEXITCODE
