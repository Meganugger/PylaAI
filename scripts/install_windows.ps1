param(
    [ValidateSet("directml", "cpu", "cuda")]
    [string]$Backend,

    [string]$PythonExecutable = "",

    [switch]$Editable
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$startBatPath = Join-Path $repoRoot "start.bat"
$generalConfigPath = Join-Path $repoRoot "cfg\general_config.toml"

Set-Location $repoRoot

function Invoke-PythonCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & $Executable @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    return [PSCustomObject]@{
        Output = ($output | Out-String).Trim()
        ExitCode = $exitCode
    }
}

function Get-BackendChoice {
    param([string]$ConfiguredBackend)

    if ($ConfiguredBackend) {
        return $ConfiguredBackend
    }

    Write-Host ""
    Write-Host "Choose your GPU backend:" -ForegroundColor Cyan
    Write-Host "  1. CUDA     - Best performance, but intended for NVIDIA RTX-series systems." -ForegroundColor Yellow
    Write-Host "  2. DirectML - Broader GPU support on Windows and the safest default for most users." -ForegroundColor Yellow
    Write-Host "  3. CPU      - Fallback if you do not want GPU inference." -ForegroundColor Yellow
    Write-Host ""

    while ($true) {
        $choice = Read-Host "Enter 1, 2, or 3"
        switch ($choice.Trim()) {
            "1" { return "cuda" }
            "2" { return "directml" }
            "3" { return "cpu" }
            default { Write-Host "Please enter 1, 2, or 3." -ForegroundColor Red }
        }
    }
}

function Get-PythonVersion {
    param(
        [string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    $result = Invoke-PythonCapture -Executable $Executable -Arguments ($PrefixArguments + @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"))
    if ($result.ExitCode -ne 0) {
        return $null
    }

    return $result.Output
}

function Ensure-BasePython310 {
    param([string]$ConfiguredPython)

    if ($ConfiguredPython) {
        $version = Get-PythonVersion -Executable $ConfiguredPython
        if (-not $version) {
            throw "Could not run the configured Python executable '$ConfiguredPython'."
        }
        if ($version -ne "3.10.0") {
            throw "Detected Python $version. PylaAI setup requires Python 3.10.0."
        }
        return [PSCustomObject]@{
            Executable = $ConfiguredPython
            PrefixArgs = @()
        }
    }

    $launcherVersion = Get-PythonVersion -Executable "py" -PrefixArguments @("-3.10")
    if (-not $launcherVersion) {
        throw "Python 3.10.0 was not found through the Windows py launcher. Install Python 3.10.0 x64 first."
    }
    if ($launcherVersion -ne "3.10.0") {
        throw "Detected Python $launcherVersion for py -3.10. PylaAI setup requires Python 3.10.0 exactly."
    }

    return [PSCustomObject]@{
        Executable = "py"
        PrefixArgs = @("-3.10")
    }
}

function Ensure-Venv {
    param([pscustomobject]$BasePython)

    if (Test-Path $venvPython) {
        $venvVersion = Get-PythonVersion -Executable $venvPython
        if ($venvVersion -ne "3.10.0") {
            throw "The existing .venv uses Python $venvVersion. Delete .venv and rerun setup with Python 3.10.0."
        }
        Write-Host "Using existing Python 3.10.0 virtual environment." -ForegroundColor Green
        return
    }

    Write-Host "Creating .venv with Python 3.10.0..." -ForegroundColor Cyan
    & $BasePython.Executable @($BasePython.PrefixArgs + @("-m", "venv", ".venv"))
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "Failed to create the virtual environment."
    }
}

function Write-StartBat {
    $content = @'
@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe main.py
pause
'@
    Set-Content -Path $startBatPath -Value $content -Encoding ASCII
}

function Set-GeneralConfigValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Key,
        [Parameter(Mandatory = $true)]
        [string]$ValueLiteral
    )

    if (-not (Test-Path $Path)) {
        throw "Missing configuration file at '$Path'."
    }

    $content = Get-Content $Path -Raw
    $escapedKey = [regex]::Escape($Key)
    $pattern = "(?m)^$escapedKey\s*=.*$"
    $replacement = "$Key = $ValueLiteral"

    if ([regex]::IsMatch($content, $pattern)) {
        $updated = [regex]::Replace($content, $pattern, $replacement)
    }
    else {
        $trimmed = $content.TrimEnd("`r", "`n")
        $updated = "$trimmed`r`n$replacement`r`n"
    }

    Set-Content -Path $Path -Value $updated -Encoding ASCII
}

function Apply-BackendThreadPreset {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedBackend
    )

    $preset = switch ($SelectedBackend) {
        "cuda" {
            @{
                preferred_backend = '"cuda"'
                cpu_or_gpu = '"gpu"'
                process_threads = "3"
                opencv_threads = "1"
                onnx_intra_threads = "2"
                onnx_inter_threads = "1"
                torch_threads = "1"
                torch_interop_threads = "1"
            }
        }
        "directml" {
            @{
                preferred_backend = '"directml"'
                cpu_or_gpu = '"gpu"'
                process_threads = "3"
                opencv_threads = "1"
                onnx_intra_threads = "2"
                onnx_inter_threads = "1"
                torch_threads = "1"
                torch_interop_threads = "1"
            }
        }
        default {
            @{
                preferred_backend = '"cpu"'
                cpu_or_gpu = '"cpu"'
                process_threads = "4"
                opencv_threads = "1"
                onnx_intra_threads = "2"
                onnx_inter_threads = "1"
                torch_threads = "1"
                torch_interop_threads = "1"
            }
        }
    }

    foreach ($entry in $preset.GetEnumerator()) {
        Set-GeneralConfigValue -Path $generalConfigPath -Key $entry.Key -ValueLiteral $entry.Value
    }

    Write-Host "Applied the recommended $SelectedBackend performance preset to cfg/general_config.toml." -ForegroundColor Green
}

function Test-InstalledRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedBackend
    )

    $testCode = "import pkg_resources; import scrcpy; print('runtime-import-check-ok')"
    if ($SelectedBackend -eq "cuda") {
        $testCode = "import pkg_resources; import scrcpy; from detect import preload_onnxruntime_gpu_dlls; preload_onnxruntime_gpu_dlls(); print('runtime-import-check-ok')"
    }

    & $venvPython -c $testCode
    if ($LASTEXITCODE -ne 0) {
        throw "Installed runtime smoke test failed. Please check the Python environment and dependency pins."
    }
}

try {
    Write-Host "PylaAI setup expects a Python 3.10.0 virtual environment." -ForegroundColor Cyan
    $selectedBackend = Get-BackendChoice -ConfiguredBackend $Backend
    $basePython = Ensure-BasePython310 -ConfiguredPython $PythonExecutable

    if ($selectedBackend -eq "cuda") {
        Write-Host "CUDA mode selected." -ForegroundColor Green
        Write-Host "CUDA offers the best performance, but this setup is intended for NVIDIA RTX-series systems." -ForegroundColor Yellow
        Write-Host "It pins the runtime libraries to CUDA 12.4, cuDNN 9.20, and the supporting CUDA/NVRTC DLL set inside the virtual environment." -ForegroundColor Cyan
    }
    elseif ($selectedBackend -eq "directml") {
        Write-Host "DirectML mode selected." -ForegroundColor Green
        Write-Host "DirectML works on a broader range of Windows GPUs and is the safest default for most users." -ForegroundColor Yellow
        Write-Host "Setup will also apply an IPS-friendly DirectML thread preset." -ForegroundColor Cyan
    }
    else {
        Write-Host "CPU mode selected." -ForegroundColor Green
        Write-Host "Setup will also apply an IPS-friendly CPU thread preset for lower oversubscription." -ForegroundColor Cyan
    }

    if ($selectedBackend -eq "cuda") {
        Write-Host "Setup will also apply an IPS-friendly CUDA thread preset." -ForegroundColor Cyan
    }

    Ensure-Venv -BasePython $basePython

    Write-Host "Upgrading pip, wheel, and a setuptools version compatible with adbutils inside .venv..." -ForegroundColor Cyan
    & $venvPython -m pip install --upgrade pip wheel "setuptools<81"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip tooling inside .venv."
    }

    $editableArgs = @()
    if ($Editable) {
        $editableArgs = @("-e")
    }

    $packageSpec = ".[{0}]" -f $selectedBackend
    Write-Host "Installing PylaAI with backend '$selectedBackend'..." -ForegroundColor Cyan
    Write-Host "Git must be available because the scrcpy client is installed from the intended upstream source revision." -ForegroundColor Yellow
    & $venvPython -m pip install @editableArgs $packageSpec
    if ($LASTEXITCODE -ne 0) {
        throw "PylaAI installation failed."
    }

    Apply-BackendThreadPreset -SelectedBackend $selectedBackend

    Write-Host "Running a quick runtime smoke test..." -ForegroundColor Cyan
    Test-InstalledRuntime -SelectedBackend $selectedBackend

    Write-StartBat
    Write-Host ""
    Write-Host "Setup completed successfully." -ForegroundColor Green
    Write-Host "You can now launch the bot by double-clicking start.bat." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
