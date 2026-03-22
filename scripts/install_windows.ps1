param(
    [ValidateSet("directml", "cpu", "cuda")]
    [string]$Backend = "directml",

    [string]$PythonExecutable = "python",

    [switch]$Editable
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$editableFlag = @()
if ($Editable) {
    $editableFlag = @("-e")
}

$packageSpec = ".[{0}]" -f $Backend

Write-Host "Upgrading pip in the selected interpreter..."
& $PythonExecutable -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Installing PylaAI with backend '$Backend'..."
& $PythonExecutable -m pip install @editableFlag $packageSpec
exit $LASTEXITCODE
