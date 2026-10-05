<#
.SYNOPSIS
  TCM-Meridian（杏林經緯）一鍵安裝：全新的 Windows 電腦也能用，不依賴 conda 等既有 Python 環境。

.DESCRIPTION
  依序完成：
    1. 預檢（作業系統、磁碟空間、網路）
    2. 找到 Python 3.12（找不到時，經你同意用 winget 安裝）
    3. 建立專案內的 .venv 並安裝套件（依 constraints.txt 固定在已測試版本）
    4. 準備設定（沒有 config.json 時由 config.example.json 複製；已存在的絕不覆蓋）
    5. 健檢（tools\doctor.py）
  本專案的 LLM 與 embedding 都走 OpenAI 相容 API：不需要 GPU，也不下載模型。
  可重複執行：已完成的步驟會略過。記錄寫在 setup.log。

.PARAMETER BasePython   指定 Python 3.12 的 python.exe（預設自動尋找）
.PARAMETER SkipDoctor   不做最後的健檢
.PARAMETER Force        重建 .venv
.PARAMETER Yes          不詢問，自動同意（例如用 winget 安裝 Python）
#>
[CmdletBinding()]
param(
    [string]$BasePython = '',
    [switch]$SkipDoctor,
    [switch]$Force,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$TotalSteps = 5
$NeedDiskGb = 3

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
            Write-Note "重建 $dir：$why"
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

$logPath = Join-Path $Root 'setup.log'
try { Start-Transcript -Path $logPath -Append | Out-Null } catch {}
$started = Get-Date
try {
    Write-Host 'TCM-Meridian（杏林經緯）安裝程式' -ForegroundColor White
    Write-Host "專案資料夾：$Root"
    Write-Host "記錄檔：$logPath"

    # ------------------------------------------------------------------ 1. preflight
    Write-Step 1 '預檢'
    if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw '需要 64 位元的 Windows。'
    }
    Write-Ok 'Windows 64 位元'
    $freeGb = Get-FreeGb $Root
    if ($freeGb -lt $NeedDiskGb) { throw ("磁碟空間不足：可用 {0:N1} GB，至少需要 {1} GB。" -f $freeGb, $NeedDiskGb) }
    Write-Ok ("磁碟可用空間 {0:N0} GB（需要約 {1} GB）" -f [Math]::Min($freeGb, 99999), $NeedDiskGb)

    # chromadb's package tree is deeply nested; a long project path can hit Windows' 260-character limit.
    $longPaths = $false
    try { $longPaths = ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction Stop).LongPathsEnabled -eq 1) } catch {}
    if ($Root.Length -gt 90 -and -not $longPaths) {
        Write-Warn ("專案路徑很長（{0} 個字元），而這台電腦沒有啟用 Windows 長路徑支援，安裝套件時可能失敗。" -f $Root.Length)
        Write-Warn '建議把專案資料夾搬到較短的路徑（例如 C:\TCM-Meridian）後再安裝。'
    }

    foreach ($h in @(@('PyPI', 'https://pypi.org/simple/pip/'), @('PyPI 檔案', 'https://files.pythonhosted.org/'))) {
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
    Install-Pip $venvPy @('-r', (Join-Path $Root 'requirements.txt'), '-c', (Join-Path $Root 'constraints.txt'))
    $probe = "import nicegui, openai, chromadb, numpy, requests; from langchain_community.vectorstores import Chroma; from langchain.docstore.document import Document; print('packages OK')"
    Invoke-Checked '確認套件' $venvPy @('-c', $probe)
    Write-Ok '.venv 就緒'

    # ------------------------------------------------------------------ 4. settings
    Write-Step 4 '準備設定'
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

    # ------------------------------------------------------------------ 5. doctor
    $doctorFailed = $false
    if ($SkipDoctor) {
        Write-Step 5 '健檢（已略過）'
    } else {
        Write-Step 5 '健檢'
        & $venvPy (Join-Path $Root 'tools\doctor.py')
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
    Write-Host '  模型：    本腳本不設定任何 LLM。啟動後請到「模型設定」分頁填入各 Agent 的 API 網址、金鑰與模型名稱。'
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
