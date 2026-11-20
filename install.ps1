$ErrorActionPreference = "Stop"

# NALA-airllm-rs Windows Installer (100% Native Safe Rust, Zero-Python)
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $env:USERPROFILE "bin"

Write-Host "[NALA-airllm-rs] Building 100% Pure Safe Rust release binary..."
cargo build --release --manifest-path (Join-Path $RepoDir "Cargo.toml")

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

$BinSrc = Join-Path $RepoDir "target\release\nala-airllm-rs.exe"
$BinDst = Join-Path $InstallDir "nala-airllm-rs.exe"
Copy-Item -Path $BinSrc -Destination $BinDst -Force

Write-Host "[NALA-airllm-rs] Installed to $BinDst"
& $BinDst version
