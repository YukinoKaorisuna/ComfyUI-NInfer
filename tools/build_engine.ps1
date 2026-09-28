<#
.SYNOPSIS
  Build ninfer_capi.dll from source (the fallback path when you do not want the
  prebuilt engine, or want to patch the engine yourself).

.DESCRIPTION
  Most users should just run:  python tools\fetch_engine.py

  This script does the long way round, and it is also the reproducible record of how the
  published DLL was produced:

    1. clone the NInfer fork that carries the Windows/MSVC port
    2. check out the pinned base commit (so the patch below applies cleanly)
    3. apply tools\patch\0001-embeddable-c-abi.patch — the C ABI layer this pack calls
    4. configure + build just the `ninfer_capi` target with CMake/Ninja

  Requirements: Visual Studio 2022 (C++ workload), CUDA Toolkit 12.8+ (sm_120a),
  CMake, Ninja, git, and a vcpkg installation providing FFmpeg + curl
  (`installed\x64-windows`).

.EXAMPLE
  pwsh -File tools\build_engine.ps1 -VcpkgPrefix E:\deps\vcpkg\installed\x64-windows
#>
[CmdletBinding()]
param(
    # The fork carrying the Windows port. Upstream (Neroued/ninfer) has no MSVC build yet, and
    # neither does toddballinger's 5080 fork, so this defaults to the 5070 Ti fork.
    #
    # For a DIFFERENT architecture, base on a community Windows build for that card instead —
    # they already did the hard part (the MSVC port), and only need the C ABI patch adding:
    #   sm_89  https://github.com/Ambolio/ninfer-4090-windows     (v3, preferred for Ada)
    #   sm_89  https://github.com/UDPSendToFailed/ninfer-4090     (v1 + v2)
    #   sm_86  https://github.com/Don-Chad/ninfer-3090            (v2)
    #   sm_120a https://github.com/natpate/ninfer-windows         (v3)
    # Example:
    #   -RepoUrl https://github.com/Ambolio/ninfer-4090-windows -Branch main -Commit '' -CudaArch 89
    [string] $RepoUrl = 'https://github.com/YukinoKaorisuna/ninfer-5070ti.git',
    [string] $Branch  = 'rtx5070ti-windows',
    # Pinned so the bundled patch applies. Pass -Commit '' to use the branch tip (then the
    # patch may need a 3-way apply).
    [string] $Commit  = '3be58a8d',
    [string] $WorkDir = '',
    [string] $BuildDir = '',
    # vcpkg prefix that contains include/, lib/ and bin/ for FFmpeg and curl.
    [string] $VcpkgPrefix = '',

    # Compute capability to compile for. This is the compatibility knob:
    #   120a  RTX 50-series (GB20x)            <- default, and what the published DLL is
    #   121a  GB10 / DGX Spark
    #    90a  H100 / H200
    #    89   RTX 40-series, L4, L40S
    #    86   RTX 30-series, A40, A6000
    #    80   A100, A30
    #    75   RTX 20-series, T4
    # A CUDA binary only runs on the capabilities it was compiled for, and an 'a' suffix
    # only runs on exactly that one. To cover several cards in a single DLL, pass a
    # semicolon list — it costs build time and binary size:
    #     -CudaArch '89;120a'
    # Note: the CUDA Toolkit must still support the architectures you ask for (CUDA 13
    # removed sm_50..sm_70).
    [string] $CudaArch = '120a',
    [int]    $Jobs = 16
)

$ErrorActionPreference = 'Continue'
$packRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir)  { $WorkDir  = Join-Path $packRoot 'build\ninfer-src' }
if (-not $BuildDir) { $BuildDir = Join-Path $packRoot 'build\ninfer-build' }

function Require-Command([string] $name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "'$name' not found on PATH. Install it or open a Developer PowerShell."
    }
}

function Get-VsInstall {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found. Install Visual Studio 2022.' }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $path) { throw 'No Visual Studio installation with the C++ workload found.' }
    return $path.Trim()
}

function Enter-MsvcEnvironment([string] $vsPath) {
    Write-Host "==> importing MSVC environment from $vsPath"
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path ("env:" + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
}

# --- 1. toolchain ---------------------------------------------------------------------
foreach ($tool in @('git', 'cmake', 'ninja')) { Require-Command $tool }
$vsPath = Get-VsInstall
Enter-MsvcEnvironment $vsPath

if (-not $env:CUDA_PATH) {
    $cuda = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
    if ($cuda) { $env:CUDA_PATH = $cuda.FullName }
}
if (-not $env:CUDA_PATH) { throw 'CUDA Toolkit not found. Install it and set CUDA_PATH.' }
Write-Host "==> CUDA: $env:CUDA_PATH"

if (-not $VcpkgPrefix) { throw 'Pass -VcpkgPrefix (the vcpkg installed\x64-windows folder).' }
if (-not (Test-Path (Join-Path $VcpkgPrefix 'include\libavformat\avformat.h'))) {
    throw "FFmpeg headers not found under $VcpkgPrefix. Install ffmpeg+curl with vcpkg first."
}

# --- 2. source ------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $WorkDir '.git'))) {
    Write-Host "==> cloning $RepoUrl ($Branch)"
    New-Item -ItemType Directory -Force -Path (Split-Path $WorkDir) | Out-Null
    git clone --branch $Branch $RepoUrl $WorkDir
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
} else {
    Write-Host "==> reusing $WorkDir"
}

if ($Commit) {
    Write-Host "==> checking out $Commit"
    git -C $WorkDir fetch --all --quiet
    git -C $WorkDir checkout --quiet $Commit
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "commit $Commit not available; staying on the branch tip. The patch may need a 3-way apply."
    }
}

# --- 3. patch -------------------------------------------------------------------------
$patch = Join-Path $PSScriptRoot 'patch\0001-embeddable-c-abi.patch'
Write-Host "==> applying $patch"
git -C $WorkDir apply --check --3way $patch 2>$null
if ($LASTEXITCODE -eq 0) {
    git -C $WorkDir apply --3way $patch
    if ($LASTEXITCODE -ne 0) { throw 'git apply failed.' }
} else {
    # Already applied (re-running the script) is the common, harmless case.
    Write-Host '    already applied or not applicable — continuing'
}

# --- 3b. relax the upstream architecture guard ----------------------------------------
# Upstream pins the whole build to 120a with a FATAL_ERROR before compiler detection, so
# -CudaArch anything-else would die in configure. Relax it here so the real compiler errors
# surface instead of a configure abort.
# NOTE (measured): relaxing this alone is NOT sufficient for non-sm_120 targets — the NVFP4
# kernels (ops/*/nvfp4/*.cu) emit PTX only sm_100+ can assemble, so ptxas aborts later in the
# build. See docs/COMPATIBILITY.md, "The architecture guard", for what a non-50-series build
# still needs.
$cmakeLists = Join-Path $WorkDir 'CMakeLists.txt'
$guardText = [System.IO.File]::ReadAllText($cmakeLists)
$relaxed = $guardText.Replace('if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")', 'if(FALSE)')
if ($relaxed -ne $guardText) {
    [System.IO.File]::WriteAllText($cmakeLists, $relaxed)
    Write-Host '    relaxed the upstream 120a-only guard (required for any other -CudaArch)'
}

# --- 4. build -------------------------------------------------------------------------
$cfgArgs = @(
    '-S', $WorkDir,
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DNINFER_FFMPEG_ROOT=$VcpkgPrefix",
    "-DNINFER_CURL_ROOT=$VcpkgPrefix"
)
Write-Host '==> configuring'
& cmake @cfgArgs
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed.' }

Write-Host "==> building ninfer_capi (-j $Jobs)"
& cmake --build $BuildDir --target ninfer_capi -j $Jobs
if ($LASTEXITCODE -ne 0) { throw 'build failed.' }

$dll = Join-Path $BuildDir 'apps\ninfer_capi.dll'
if (-not (Test-Path $dll)) { throw "build finished but $dll is missing." }

$size = (Get-Item $dll).Length / 1MB
Write-Host ''
Write-Host ("built : {0} ({1:N1} MiB)" -f $dll, $size)
Write-Host 'Copy it plus the sibling DLLs into the pack:'
Write-Host ("  Copy-Item '{0}' -Destination '{1}'" -f (Split-Path $dll), (Join-Path $packRoot 'bin'))
