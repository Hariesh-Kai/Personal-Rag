param(
    [int]$BackendPort = 8012,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$TempDir = Join-Path $Root "backend\data\tmp"

New-Item -ItemType Directory -Force $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:TMPDIR = $TempDir
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HOME = "D:\models\hf-cache"
$env:RAG_TEMP_DIR = $TempDir
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $TempDir "torch-cache\inductor"

function Stop-PortProcess([int]$Port) {
    $lines = netstat -ano | Select-String -Pattern ":$Port"
    foreach ($line in $lines) {
        if ($line -match "LISTENING\s+(\d+)$") {
            Stop-Process -Id ([int]$Matches[1]) -Force -ErrorAction SilentlyContinue
        }
    }
}

Stop-PortProcess $BackendPort

$python = Join-Path $Root ".venv\Scripts\python.exe"
Start-Process -WindowStyle Hidden -FilePath $python -ArgumentList "-B manage.py runserver 127.0.0.1:$BackendPort --noreload" -WorkingDirectory $Root

Start-Sleep -Seconds 4
$health = Invoke-RestMethod "http://127.0.0.1:$BackendPort/api/health" -TimeoutSec 30

$frontendListening = netstat -ano | Select-String -Pattern ":$FrontendPort" | Select-String -Pattern "LISTENING"
if (-not $frontendListening) {
    Start-Process -WindowStyle Hidden -FilePath "npm.cmd" -ArgumentList "run dev -- --host 127.0.0.1 --port $FrontendPort" -WorkingDirectory (Join-Path $Root "frontend")
}

Write-Host "Backend:  http://127.0.0.1:$BackendPort"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "Embedding: $($health.embedding_model) / $($health.embedding_backend) / $($health.embedding_dimensions)d"
Write-Host "Index: $($health.index_session) ($($health.chunks) chunks)"
