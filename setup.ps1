<#
.SYNOPSIS
  TCM-Meridian（杏林經緯）一鍵安裝：全新的 Windows 電腦也能用，不依賴 conda 等既有 Python 環境。

.DESCRIPTION
  依序完成：
    1. 預檢（作業系統、磁碟空間、網路；選擇安裝語音功能時另外檢查 NVIDIA 顯卡與驅動）
    2. 找到 Python 3.12（找不到時，經你同意用 winget 安裝）
    3. 建立專案內的 .venv 並安裝套件（依 constraints.txt 固定在已測試版本）
    4. 建立獨立的語音辨識環境 .venv-asr（含 CUDA 版 PyTorch，下載約 4 GB；不論是否安裝語音功能都會建立）
    5. 準備設定（沒有 config.json 時由 config.example.json 複製；已存在的絕不覆蓋）
    6. 下載語音模型（Qwen3-ASR-1.7B、ForcedAligner-0.6B 與說話者聲紋模型，共約 6.1 GB，固定版本、可續傳；
       只有選擇安裝語音功能時才下載）
    7. 健檢（tools\doctor.py）
  本專案的 LLM 與 embedding 都走 OpenAI 相容 API（不需要 GPU）；只有語音功能的語音辨識在本機用 NVIDIA GPU 執行。
  沒有安裝語音模型時，其他功能完全不受影響；「語音功能」按鈕會提示目前無下載語音模型。之後想補裝：重新執行 install.cmd 並選擇安裝語音功能。
  可重複執行：已完成的步驟會略過，下載可續傳。記錄寫在 setup.log。

.PARAMETER BasePython   指定 Python 3.12 的 python.exe（預設自動尋找）
.PARAMETER Voice        安裝語音功能（下載語音模型；需要 NVIDIA 顯卡）。沒指定 -Voice / -NoVoice 時會詢問；加 -Yes 則預設安裝
.PARAMETER NoVoice      不安裝語音功能（不下載語音模型；.venv-asr 仍會建立，之後補裝只需下載模型）
.PARAMETER ModelsDir    語音模型存放資料夾（預設 <專案>\models）
.PARAMETER HfEndpoint   Hugging Face 鏡像網址（網路受限時使用）
.PARAMETER SkipGpuCheck 不檢查 NVIDIA 顯卡與驅動（沒有 GPU 又要安裝語音功能時的逃生口；語音辨識在 CPU 上會非常慢）
.PARAMETER SkipDoctor   不做最後的健檢
.PARAMETER SkipSmoke    健檢時不實際載入語音模型辨識一次
.PARAMETER Dev          另外安裝開發用套件（pytest；只用於跑 tests\，執行程式不需要）
.PARAMETER Force        重建 .venv 與 .venv-asr
.PARAMETER Yes          不詢問，自動同意（用 winget 安裝 Python；語音功能預設安裝）
#>
[CmdletBinding()]
param(
    [string]$BasePython = '',
    [switch]$Voice,
    [switch]$NoVoice,
    [string]$ModelsDir = '',
    [string]$HfEndpoint = '',
    [switch]$SkipGpuCheck,
    [switch]$SkipDoctor,
    [switch]$SkipSmoke,
    [switch]$Dev,
    [switch]$Force,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$TotalSteps = 7
$MinDriverMajor = 570            # CUDA 12.8 (the pinned PyTorch build) needs a Windows driver of about 570 or newer
$NeedDiskGbEnv = 12              # .venv + .venv-asr (the PyTorch wheel alone is ~3.3 GB; unpacked it is ~8 GB)
$NeedDiskGbVoice = 20            # ... plus the speech models (~6.1 GB) and their download cache

function Write-Step([int]$n, [string]$text) {
    Write-Host ''
    Write-Host "==> [$n/$TotalSteps] $text" -ForegroundColor Cyan
}
function Write-Ok([string]$text)   { Write-Host "    [OK]   $text" -ForegroundColor Green }
function Write-Note([string]$text) { Write-Host "    $text" }
function Write-Warn([string]$text) { Write-Host "    [注意] $text" -ForegroundColor Yellow }

function Invoke-Checked([string]$what, [string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$what 失敗（結束碼 $LASTEXITCODE）。" }
}

function Test-Reachable([string]$url) {
    try {
        Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null
        return $true
    } catch {
        # An HTTP error status (403/404/405) still proves the host is reachable.
        if ($_.Exception.Response) { return $true }
        return $false
    }
}

function Get-FreeGb([string]$path) {
    $drive = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $path).Path)
    if ($drive.StartsWith('\\')) { return [double]::MaxValue }      # network share: cannot tell, do not block
    return (New-Object System.IO.DriveInfo $drive).AvailableFreeSpace / 1GB
}

function Get-NvidiaSmi {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    return $null
}

function Invoke-Quiet([string]$exe, [string[]]$arguments) {
    # Run a native probe without letting stderr output turn into a terminating error (Windows PowerShell 5.1).
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& $exe @arguments 2>&1 | ForEach-Object { "$_" })
        return [pscustomobject]@{ Code = $LASTEXITCODE; Lines = $lines }
    } catch {
        return [pscustomobject]@{ Code = -1; Lines = @() }
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Test-Python312([string]$exe) {
    if ([string]::IsNullOrWhiteSpace($exe) -or -not (Test-Path -LiteralPath $exe)) { return $false }
    $r = Invoke-Quiet $exe @('-c', "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.maxsize > 2**32)")
    return ($r.Code -eq 0 -and $r.Lines.Count -ge 2 -and $r.Lines[0] -eq '3.12' -and $r.Lines[1] -eq 'True')
}

function Find-Python312 {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $r = Invoke-Quiet 'py' @('-3.12', '-c', 'import sys; print(sys.executable)')
        if ($r.Code -eq 0 -and $r.Lines.Count -ge 1) { $candidates.Add($r.Lines[0]) }
    }
    foreach ($name in 'python', 'python3') {
        $c = Get-Command $name -ErrorAction SilentlyContinue
        if ($c -and $c.Source) { $candidates.Add($c.Source) }
    }
    $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'))
    $candidates.Add((Join-Path $env:ProgramFiles 'Python312\python.exe'))
    foreach ($c in $candidates) { if (Test-Python312 $c) { return $c } }
    return $null
}

function Install-Python312 {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw ("找不到 Python 3.12，這台電腦也沒有 winget。請先從 https://www.python.org/downloads/ 安裝 " +
               "Python 3.12（64 位元），再重新執行本腳本；或用 -BasePython 指定 python.exe。")
    }
    if (-not $Yes) {
        $answer = Read-Host '    找不到 Python 3.12。要用 winget 安裝（只安裝給目前使用者，不需系統管理員）嗎？ [Y/n]'
        if ($answer -match '^(n|no)$') { throw '已取消：需要 Python 3.12 才能繼續。' }
    }
    Write-Note '用 winget 安裝 Python 3.12 …'
    # A PowerShell function RETURNS everything it writes to the pipeline, not just what follows `return`. winget's own
    # output (the msstore agreement text, "Successfully installed ...") would be returned together with the path, so
    # $python would become an array. Out-Host still shows (and logs) the output but keeps it out of the return value.
    & winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements | Out-Host
    # winget returns a non-zero code when it is already installed; what matters is whether we can find it now.
    $found = Find-Python312
    if (-not $found) {
        throw 'Python 3.12 安裝後仍找不到。請關閉這個視窗、重新開啟後再執行 setup.ps1。'
    }
    return [string]$found
}

function Get-VenvHome([string]$venvDir) {
    $cfg = Join-Path $venvDir 'pyvenv.cfg'
    if (-not (Test-Path -LiteralPath $cfg)) { return $null }
    foreach ($line in Get-Content -LiteralPath $cfg) {
        if ($line -match '^\s*home\s*=\s*(.+?)\s*$') { return $Matches[1] }
    }
    return $null
}

function Ensure-Venv([string]$dir, [string]$python) {
    $venvPy = Join-Path $dir 'Scripts\python.exe'
    if (Test-Path -LiteralPath $dir) {
        $why = $null
        if ($Force) { $why = '指定了 -Force' }
        else {
            $works = $false
            if (Test-Path -LiteralPath $venvPy) { $works = ((Invoke-Quiet $venvPy @('-c', 'import sys')).Code -eq 0) }
            $baseHome = Get-VenvHome $dir
            $wantHome = Split-Path -Parent $python
            if (-not $works) { $why = '環境已損毀，或它的基底 Python 已不存在' }
            elseif ($baseHome -and ($baseHome.TrimEnd('\') -ne $wantHome.TrimEnd('\'))) {
                $why = "基底 Python 不同（$baseHome），改用 $wantHome"
            }
        }
        if ($why) {
            Write-Note "重建 ${dir}：$why"
            try { Remove-Item -LiteralPath $dir -Recurse -Force }
            catch { throw "無法刪除 $dir（可能有程式正在使用它，請先關閉 TCM-Meridian 與終端機）：$($_.Exception.Message)" }
        } else {
            Write-Ok "沿用既有環境 $dir"
            return $venvPy
        }
    }
    # Same rule as in Install-Python312: anything `python -m venv` prints must not become part of the returned path.
    Invoke-Checked "建立 $dir" $python @('-m', 'venv', $dir) | Out-Host
    return [string]$venvPy
}

function Install-Pip([string]$py, [string[]]$pipArguments) {
    # Deliberately NOT piped to Out-Host: it is only ever called as a statement (its output is never captured), and a
    # native command that is not piped keeps the console, so pip shows its live download progress bar.
    $all = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--retries', '10', '--timeout', '60') + $pipArguments
    & $py @all
    if ($LASTEXITCODE -ne 0) {
        throw "pip 安裝失敗（結束碼 $LASTEXITCODE）。請確認網路後重新執行 setup.ps1，已下載的部分會續用。"
    }
}

# Decide about the voice feature before anything is downloaded: it changes the disk space, the checks and the downloads.
if ($Voice -and $NoVoice) { Write-Host '[失敗] -Voice 與 -NoVoice 不能同時指定。' -ForegroundColor Red; exit 1 }
$InstallVoice = $true
if ($NoVoice) { $InstallVoice = $false }
elseif (-not $Voice -and -not $Yes) {
    Write-Host ''
    Write-Host '語音功能：錄下診間對話、即時辨識成逐字稿，交給 AI 主治醫師與各子 Agent 參考。' -ForegroundColor White
    Write-Host '  需要 NVIDIA 顯卡（驅動 570 以上）；要下載語音模型約 6.1 GB。不安裝也不影響其他功能，之後想補裝再執行一次 install.cmd 即可。'
    $answer = Read-Host '要安裝語音功能嗎？ [Y/n]'
    if ($answer -match '^(n|no)$') { $InstallVoice = $false }
}

$logPath = Join-Path $Root 'setup.log'
try { Start-Transcript -Path $logPath -Append | Out-Null } catch {}
$started = Get-Date
try {
    Write-Host 'TCM-Meridian（杏林經緯）安裝程式' -ForegroundColor White
    Write-Host "專案資料夾：$Root"
    Write-Host "記錄檔：$logPath"
    Write-Host ("語音功能：" + $(if ($InstallVoice) { '安裝（下載語音模型）' } else { '不安裝（不下載語音模型）' }))

    # ------------------------------------------------------------------ 1. preflight
    Write-Step 1 '預檢'
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw '需要 64 位元的 Windows。'
    }
    Write-Ok 'Windows 64 位元'
    $needGb = $(if ($InstallVoice) { $NeedDiskGbVoice } else { $NeedDiskGbEnv })
    $freeGb = Get-FreeGb $Root
    if ($freeGb -lt $needGb) { throw ("磁碟空間不足：可用 {0:N1} GB，至少需要 {1} GB。" -f $freeGb, $needGb) }
    Write-Ok ("磁碟可用空間 {0:N0} GB（需要約 {1} GB）" -f [Math]::Min($freeGb, 99999), $needGb)

    # chromadb's and PyTorch's package trees are deeply nested; a long project path can hit Windows' 260-character limit.
    $longPaths = $false
    try { $longPaths = ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction Stop).LongPathsEnabled -eq 1) } catch {}
    if ($Root.Length -gt 90 -and -not $longPaths) {
        Write-Warn ("專案路徑很長（{0} 個字元），而這台電腦沒有啟用 Windows 長路徑支援，安裝套件（尤其 PyTorch）時可能失敗。" -f $Root.Length)
        Write-Warn '建議把專案資料夾搬到較短的路徑（例如 C:\TCM-Meridian）後再安裝。'
    }

    if ($InstallVoice) {
        if ($SkipGpuCheck) { Write-Warn '已略過 NVIDIA 檢查（-SkipGpuCheck）。沒有 GPU 時語音辨識會非常慢。' }
        else {
            $smi = Get-NvidiaSmi
            if (-not $smi) {
                throw ('找不到 NVIDIA 驅動程式（nvidia-smi）。語音功能的語音辨識需要 NVIDIA 顯卡；請先安裝最新驅動 ' +
                       '（https://www.nvidia.com/Download/index.aspx）。不需要語音功能時請加 -NoVoice；若確定要在沒有 GPU 的電腦上安裝，可加 -SkipGpuCheck。')
            }
            $gpuLine = (& $smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
            $parts = $gpuLine -split ',\s*'
            $driver = [version]($parts[1].Trim())
            if ($driver.Major -lt $MinDriverMajor) {
                throw ("NVIDIA 驅動版本 {0} 太舊（需要 {1} 以上才支援 CUDA 12.8）。請到 https://www.nvidia.com/Download/index.aspx 更新驅動，或加 -NoVoice 不安裝語音功能。" -f $driver, $MinDriverMajor)
            }
            Write-Ok ("NVIDIA GPU：{0}，驅動 {1}" -f $parts[0].Trim(), $driver)
        }
    }

    $hosts = @(@('PyPI', 'https://pypi.org/simple/pip/'), @('PyPI 檔案', 'https://files.pythonhosted.org/'),
               @('PyTorch', 'https://download.pytorch.org/whl/cu128/'))
    if ($InstallVoice) {
        $hf = $(if ($HfEndpoint) { $HfEndpoint } else { 'https://huggingface.co' })
        $hosts += , @('Hugging Face', $hf)
    }
    foreach ($h in $hosts) {
        if (Test-Reachable $h[1]) { Write-Ok "連得上 $($h[0])" }
        else { throw "連不到 $($h[0])（$($h[1])）。請檢查網路、VPN 或代理（可設定環境變數 HTTPS_PROXY）。" }
    }

    # ------------------------------------------------------------------ 2. python
    Write-Step 2 '尋找 Python 3.12'
    if ($BasePython) {
        if (-not (Test-Python312 $BasePython)) { throw "-BasePython 指定的不是 64 位元 Python 3.12：$BasePython" }
        $python = (Resolve-Path -LiteralPath $BasePython).Path
    } else {
        $python = Find-Python312
        if (-not $python) { $python = Install-Python312 }
    }
    # Whatever got us here, $python must be exactly one path to a working Python 3.12 (a clear message beats a
    # confusing "無法辨識 ..." from the next step if a function ever leaks output into its return value again).
    if ($python -isnot [string] -or -not (Test-Python312 $python)) {
        throw ("找到的 Python 3.12 無法使用：{0}。請用 -BasePython 指定 python.exe，或重新執行 setup.ps1。" -f (@($python) -join ' | '))
    }
    Write-Ok "Python 3.12：$python"

    # ------------------------------------------------------------------ 3. venv + packages
    Write-Step 3 '建立 Python 環境（.venv）並安裝套件'
    $venvPy = Ensure-Venv (Join-Path $Root '.venv') $python
    Install-Pip $venvPy @('--upgrade', 'pip')
    $reqFile = $(if ($Dev) { 'requirements-dev.txt' } else { 'requirements.txt' })
    Install-Pip $venvPy @('-r', (Join-Path $Root $reqFile), '-c', (Join-Path $Root 'constraints.txt'))
    $probe = "import nicegui, openai, chromadb, numpy, requests, soundcard, av, opencc, onnxruntime; from langchain_community.vectorstores import Chroma; from langchain.docstore.document import Document; print('packages OK')"
    Invoke-Checked '確認套件' $venvPy @('-c', $probe)
    Write-Ok '.venv 就緒'

    # ------------------------------------------------------------------ 4. speech-recognition venv
    Write-Step 4 '建立獨立的語音辨識環境（.venv-asr，含 CUDA 版 PyTorch，下載約 4 GB，請耐心等候）'
    $asrPy = Ensure-Venv (Join-Path $Root '.venv-asr') $python
    Install-Pip $asrPy @('--upgrade', 'pip')
    Install-Pip $asrPy @('-r', (Join-Path $Root 'requirements-asr.txt'))
    $asrProbe = "import torch, qwen_asr; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
    Invoke-Checked '確認語音辨識套件' $asrPy @('-c', $asrProbe)
    if ($InstallVoice -and -not $SkipGpuCheck) {
        & $asrPy -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
        if ($LASTEXITCODE -ne 0) {
            throw 'PyTorch 看不到 GPU（CUDA 不可用）。請更新 NVIDIA 驅動並重新開機後再執行 setup.ps1；或加 -NoVoice 不安裝語音功能。'
        }
    }
    Write-Ok '.venv-asr 就緒'

    # ------------------------------------------------------------------ 5. settings
    Write-Step 5 '準備設定'
    $configPath = Join-Path $Root 'config.json'
    $examplePath = Join-Path $Root 'config.example.json'
    if (Test-Path -LiteralPath $configPath) {
        Write-Ok '沿用既有 config.json（不會覆蓋）'
    } elseif (Test-Path -LiteralPath $examplePath) {
        Copy-Item -LiteralPath $examplePath -Destination $configPath
        Write-Ok '已由 config.example.json 建立 config.json（API key 仍是佔位符）'
    } else {
        Write-Warn '找不到 config.example.json，已略過；程式會使用內建預設值（LM Studio localhost）。'
    }
    $dataRoot = Join-Path $Root 'patient_data'
    if (-not (Test-Path -LiteralPath $dataRoot)) { New-Item -ItemType Directory -Path $dataRoot | Out-Null }
    Write-Ok 'patient_data 資料夾就緒'

    # ------------------------------------------------------------------ 6. speech models
    if (-not $InstallVoice) {
        Write-Step 6 '下載語音模型（已略過：未選擇安裝語音功能）'
        Write-Note '沒有語音模型時，「語音功能」按鈕會提示目前無下載語音模型，其他功能不受影響。'
        Write-Note '之後想補裝：重新執行 install.cmd 並選擇安裝語音功能，或執行 .venv-asr\Scripts\python.exe tools\download_models.py --write-config'
    } else {
        Write-Step 6 '下載語音模型（Qwen3-ASR-1.7B、ForcedAligner-0.6B 與說話者聲紋模型，共約 6.1 GB，可續傳）'
        $dlArgs = @((Join-Path $Root 'tools\download_models.py'), '--write-config')
        # 明確指定 -ModelsDir 時以它為準；否則才沿用 config.json 裡已指到完整模型的資料夾。
        if ($ModelsDir) { $dlArgs += @('--dir', $ModelsDir) } else { $dlArgs += '--reuse-config' }
        if ($HfEndpoint) { $dlArgs += @('--endpoint', $HfEndpoint) }
        Invoke-Checked '下載語音模型' $asrPy $dlArgs
        Write-Ok '語音模型就緒'
    }

    # ------------------------------------------------------------------ 7. doctor
    $doctorFailed = $false
    if ($SkipDoctor) {
        Write-Step 7 '健檢（已略過）'
    } else {
        Write-Step 7 '健檢'
        $docArgs = @((Join-Path $Root 'tools\doctor.py'))
        if ($InstallVoice) {
            $docArgs += '--voice'
            if (-not $SkipSmoke) { $docArgs += '--smoke' }
        }
        & $venvPy @docArgs
        $doctorFailed = ($LASTEXITCODE -ne 0)
    }

    $minutes = [Math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    Write-Host ''
    if ($doctorFailed) {
        Write-Host "安裝已完成，但健檢有失敗項目（見上方 [FAIL]）。處理後可重新執行 setup.ps1。（耗時 $minutes 分鐘）" -ForegroundColor Yellow
        exit 2
    }
    Write-Host "安裝完成！（耗時 $minutes 分鐘）" -ForegroundColor Green
    Write-Host '  啟動：    雙擊 start.cmd（或 .venv\Scripts\python.exe TCM_Meridian_main.py）'
    Write-Host '  網址：    http://localhost:8080/'
    Write-Host '  模型：    本腳本不設定任何 LLM。啟動後請到「模型設定」分頁填入各 Agent 的 API 網址、金鑰與模型名稱'
    Write-Host '            （語音功能另有「逐字稿校稿」LLM，也在「模型設定」分頁）。'
    if ($InstallVoice) {
        Write-Host '  語音：    已安裝。左欄「語音功能」按鈕開啟錄音視窗；啟動程式時會把語音模型預載到顯示卡。'
    } else {
        Write-Host '  語音：    未安裝語音模型（左欄「語音功能」按鈕會提示目前無下載語音模型）；之後重新執行 install.cmd 並選擇安裝即可。'
    }
    Write-Host '  教授：    示範教授 professor_01、professor_02 尚未建立向量索引；設好 embedding 模型後，'
    Write-Host '            到「教授設定」分頁按「建立資料庫」。'
    exit 0
} catch {
    Write-Host ''
    Write-Host "[失敗] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "詳細記錄：$logPath" -ForegroundColor Red
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch {}
}
