# Start the Regulus UI on Windows (scripts/start_ui.sh's counterpart): Qwen3-4B-Instruct-2507 (the MLX 4-bit directory,
# run by PyTorch: scripts/torch_worker.py) and the production pipeline (scripts/serve_ui.py).
# The machine's paths come from configs/ui.env: copy configs/ui.env.example and fill in its Windows section.
# Open http://127.0.0.1:8000 when it prints "Serving". Stop with Ctrl+C.
#   powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1 [extra serve_ui.py flags]
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$profilePath = if ($env:REGULUS_UI_PROFILE) { $env:REGULUS_UI_PROFILE } else { "configs\ui.env" }
if (-not (Test-Path $profilePath)) {
    Write-Error "missing ${profilePath}: copy configs\ui.env.example to configs\ui.env and set the paths"
    exit 1
}
. "$PSScriptRoot\load_profile.ps1" $profilePath
$port = if ($env:REGULUS_UI_PORT) { $env:REGULUS_UI_PORT } else { "8000" }
& .\.venv\Scripts\python.exe scripts\serve_ui.py --port $port --offline @args
exit $LASTEXITCODE
