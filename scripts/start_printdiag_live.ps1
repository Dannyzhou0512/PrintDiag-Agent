# PrintDiag P10-7C formal Live Qwen launcher.
# Token is read securely or inherited from the environment; it is never printed,
# written to disk, sent to the browser, or passed as a command-line argument.
param(
    [string]$Python = "",
    [int]$Port = 8010,
    [string]$CloudUrl = "http://127.0.0.1:18008",
    [int]$Budget = 10,
    [int]$MaxConcurrency = 1,
    [string]$StatePath = "",
    [switch]$ResetBudget,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$plainToken = $null
$bstr = [IntPtr]::Zero
$probePath = $null
$budgetProbePath = $null

function Fail($message) {
    Write-Host "BLOCKED: $message" -ForegroundColor Red
    exit 1
}

# 1. Python discovery: prefer the requested local venv, never a hard-coded dev path.
if (-not $Python) {
    $candidates = @(
        (Join-Path $root ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
    }
    if (-not $Python) { $Python = "python" }
}
if ($Python -ne "python" -and -not (Test-Path -LiteralPath $Python)) {
    Fail "Python not found: $Python"
}

# 2. Port availability; never kill another process automatically.
$webApp = Join-Path $root "src\printdiag\web\app.py"
if (-not (Test-Path -LiteralPath $webApp)) { Fail "latest web source not found: $webApp" }
Write-Host "Web source: $webApp (P10-7C)"
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    Fail "port $Port is in use by PID $($listener.OwningProcess); choose -Port 8011/8012 or stop that process yourself"
}
Write-Host "Port $Port is available."

# 3. Dependency check.
$env:PYTHONPATH = (Join-Path $root "src") + ";" + $root
$env:PYTHONUTF8 = "1"
& $Python -c "import fastapi, uvicorn, httpx, printdiag; print('dependencies: ok')"
if ($LASTEXITCODE -ne 0) { Fail "Python dependencies missing; install with pip install -e `".[agent]`"" }

# 4. Token: inherit a controlled env var, otherwise prompt without echo.
if ($env:PRINTDIAG_REMOTE_INFERENCE_TOKEN) {
    $plainToken = $env:PRINTDIAG_REMOTE_INFERENCE_TOKEN
} elseif ($env:PRINTDIAG_CLOUD_TOKEN) {
    $plainToken = $env:PRINTDIAG_CLOUD_TOKEN
} else {
    Write-Host "Enter the cloud inference token (input is hidden, not logged):"
    $secure = Read-Host -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
}
if (-not $plainToken) { Fail "no token configured" }

try {
    $env:PRINTDIAG_REMOTE_INFERENCE_TOKEN = $plainToken
    $env:PRINTDIAG_ALLOW_REMOTE_INFERENCE = "1"
    $env:PRINTDIAG_REMOTE_INFERENCE_URL = $CloudUrl

    # 5. Cloud health/model-info only. /v1/generate is NEVER called here.
    $probe = @'
import json, os, sys
import httpx
url = os.environ["PRINTDIAG_REMOTE_INFERENCE_URL"].rstrip("/")
token = os.environ["PRINTDIAG_REMOTE_INFERENCE_TOKEN"]
headers = {"X-PrintDiag-Token": token, "Accept": "application/json"}
try:
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=5.0) as client:
        health = client.get(url + "/healthz", headers=headers)
        info = client.get(url + "/v1/model-info", headers=headers)
    result = {
        "cloud_online": health.status_code == 200,
        "model_info_status": info.status_code,
        "model_ready": bool(
            health.status_code == 200 and info.status_code == 200
            and health.json().get("model_ready")
            and (info.json().get("available_modes") or [])),
        "service": health.json().get("service") if health.status_code == 200 else None,
        "modes": info.json().get("available_modes") if info.status_code == 200 else [],
        "http_status": health.status_code,
        "gpu_calls": 0,
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["cloud_online"] and result["model_ready"] else 1)
except Exception as exc:
    print(json.dumps({"cloud_online": False, "model_ready": False,
                      "reason": type(exc).__name__, "gpu_calls": 0}, ensure_ascii=False))
    sys.exit(1)
'@
    $probePath = Join-Path $env:TEMP "printdiag_probe_$PID.py"
    Set-Content -LiteralPath $probePath -Value $probe -Encoding UTF8
    $probeText = & $Python $probePath
    Write-Host $probeText
    if ($LASTEXITCODE -ne 0) { Fail "cloud health/model check failed; start the GPU service and SSH tunnel, then retry" }

    # 6. Persistent budget ledger (survives restarts unless -ResetBudget).
    if (-not $StatePath) {
        $StatePath = Join-Path $root "runs\p10_7c_live_budget\ledger.json"
    }
    $env:PRINTDIAG_LIVE_BUDGET_STATE = $StatePath
    $env:PRINTDIAG_LIVE_GPU_BUDGET = "$Budget"
    $env:PRINTDIAG_LIVE_MAX_CONCURRENCY = "$MaxConcurrency"
    $env:PRINTDIAG_LIVE_AUDIT_DIR = Join-Path $root "runs\p10_7c_live_audit"

    if ($ResetBudget) {
        if (Test-Path -LiteralPath $StatePath) { Remove-Item -LiteralPath $StatePath -Force }
        Write-Host "Budget ledger explicitly reset by -ResetBudget." -ForegroundColor Yellow
    }
    $budgetProbe = @'
import json, os
from printdiag.web.budget import BudgetStore
store = BudgetStore(os.environ["PRINTDIAG_LIVE_BUDGET_STATE"])
store.configure(int(os.environ["PRINTDIAG_LIVE_GPU_BUDGET"]))
print(json.dumps(store.status(), ensure_ascii=False))
'@
    $budgetProbePath = Join-Path $env:TEMP "printdiag_budget_probe_$PID.py"
    Set-Content -LiteralPath $budgetProbePath -Value $budgetProbe -Encoding UTF8
    $budgetText = & $Python $budgetProbePath
    Write-Host "Budget: $budgetText"

    if ($CheckOnly) {
        Write-Host "CheckOnly complete: configuration OK; server NOT started; no /v1/generate call."
        exit 0
    }

    Write-Host "Starting PrintDiag Live Web on http://127.0.0.1:$Port"
    Write-Host "Device: Marlin Simulator http://127.0.0.1:8100 (read-only)"
    Write-Host "Cloud: $CloudUrl; budget=$Budget; concurrency=$MaxConcurrency"
    & $Python -m uvicorn printdiag.web.app:app --host 127.0.0.1 --port $Port
}
finally {
    Remove-Item Env:PRINTDIAG_REMOTE_INFERENCE_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_ALLOW_REMOTE_INFERENCE -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_REMOTE_INFERENCE_URL -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_LIVE_GPU_BUDGET -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_LIVE_MAX_CONCURRENCY -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_LIVE_BUDGET_STATE -ErrorAction SilentlyContinue
    Remove-Item Env:PRINTDIAG_LIVE_AUDIT_DIR -ErrorAction SilentlyContinue
    $plainToken = $null
    if ($probePath -and (Test-Path -LiteralPath $probePath)) { Remove-Item -LiteralPath $probePath -Force }
    if ($budgetProbePath -and (Test-Path -LiteralPath $budgetProbePath)) { Remove-Item -LiteralPath $budgetProbePath -Force }
    if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    Write-Host "Temporary environment cleared."
}
