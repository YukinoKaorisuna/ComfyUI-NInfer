<#
.SYNOPSIS
  Maintainer helper: package the engine into the archive published as a GitHub Release
  asset, then print the exact command to upload it.

.DESCRIPTION
  Users get the engine through `python tools/fetch_engine.py`, which pulls this archive.
  The archive contains:

    ninfer_capi.dll      the C ABI engine this node pack loads (the big one, ~228 MB)
    av*.dll, sw*.dll     its FFmpeg runtime imports
    libcurl.dll, z.dll   and the rest of the resolved dependencies
    LICENSE-ninfer.txt   Apache-2.0 text for the engine (required when redistributing)
    NOTICE.txt           attribution + what was changed

  Debug symbols (*.pdb) and the CLI executables are deliberately excluded: the first
  triples the size for no benefit, the second is not needed by a ctypes host.

.EXAMPLE
  pwsh -File tools\make_release.ps1 -Version 0.1.0
#>
[CmdletBinding()]
param(
    # Where the Windows build put ninfer_capi.dll and the runtime DLLs next to it.
    # Relative to the pack root (matches a ninfer checkout sitting beside this repo);
    # pass an absolute path to override.
    [string] $BuildDir = '..\ninfer-5080\build-windows\apps',

    # Where to write the staged folder and the .zip (defaults to build\release).
    [string] $OutDir = '',

    # Version stamped into the archive name; defaults to the pack version.
    [string] $Version = '0.1.0',

    # Compute capability the engine was built for, e.g. '120a', '89', '86'.
    # It goes into the asset name so tools/fetch_engine.py can pick the right build for a
    # user's GPU automatically:  ninfer-engine-win-x64-sm120a.zip
    # Leave empty to publish a generic asset name (only do this when there is exactly one
    # build, otherwise users cannot tell them apart).
    [string] $CudaArch = '120a'
)

$ErrorActionPreference = 'Stop'

$packRoot = Split-Path -Parent $PSScriptRoot          # ...\ComfyUI-NInfer
if (-not $OutDir) { $OutDir = Join-Path $packRoot 'build\release' }

# A relative BuildDir is resolved against the pack root, so the default above works
# regardless of where the repo was cloned.
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $packRoot $BuildDir
}
$engine = Join-Path $BuildDir 'ninfer_capi.dll'
if (-not (Test-Path $engine)) {
    throw "ninfer_capi.dll not found in '$BuildDir'. Build it first: see tools\build_engine.ps1"
}

$archTag = if ($CudaArch) { "-sm$($CudaArch -replace '[^0-9a-z]', '')" } else { '' }
$assetName = "ninfer-engine-win-x64$archTag-$Version"
$stage = Join-Path $OutDir $assetName
$zip = Join-Path $OutDir "$assetName.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# --- collect the runtime set ---------------------------------------------------------
$runtime = Get-ChildItem $BuildDir -Filter *.dll -File | Sort-Object Name
foreach ($file in $runtime) {
    Copy-Item $file.FullName $stage -Force
}
Write-Host "$($runtime.Count) DLL(s) copied (engine included)"
$total = ($runtime | Measure-Object Length -Sum).Sum
Write-Host ("total size: {0:N1} MiB" -f ($total / 1MB))

# --- attribution (required: Apache-2.0 engine + LGPL FFmpeg) --------------------------
$candidates = @(
    (Join-Path $packRoot 'THIRD_PARTY\LICENSE-ninfer-Apache-2.0.txt'),
    (Join-Path $BuildDir '..\..\..\LICENSE')
)
$licenseSource = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($licenseSource) {
    Copy-Item $licenseSource (Join-Path $stage 'LICENSE-ninfer.txt') -Force
    Write-Host "included LICENSE-ninfer.txt from $licenseSource"
} else {
    Write-Warning "Could not find ninfer's Apache-2.0 LICENSE to bundle. Add it manually."
}

foreach ($extra in @('NOTICE.md', 'THIRD_PARTY')) {
    $path = Join-Path $packRoot $extra
    if (Test-Path $path) {
        if ((Get-Item $path).PSIsContainer) {
            Copy-Item $path (Join-Path $stage $extra) -Recurse -Force
        } else {
            Copy-Item $path (Join-Path $stage 'NOTICE.txt') -Force
        }
    }
}

# --- zip ------------------------------------------------------------------------------
if (Test-Path $zip) { Remove-Item $zip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip, 'Optimal', $false)

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
$size = (Get-Item $zip).Length / 1MB

Write-Host ''
Write-Host ('archive : {0}' -f $zip)
Write-Host ('size    : {0:N1} MiB' -f $size)
Write-Host ('sha256  : {0}' -f $hash)
Write-Host ''
Write-Host 'Upload it (needs the GitHub CLI, logged in):'
Write-Host ''
$archLabel = if ($CudaArch) { 'sm' + ($CudaArch -replace '[^0-9a-z]', '') } else { 'generic' }
$notes = "Prebuilt NInfer engine for Windows x64 (CUDA $archLabel)."
$uploadLines = @(
    "gh release create v$Version `"$zip`""
    "    --repo YukinoKaorisuna/ComfyUI-NInfer"
    "    --title `"v$Version`""
    "    --notes `"$notes`""
)
foreach ($line in $uploadLines) { Write-Host ('  ' + $line) }
Write-Host ''
Write-Host 'Then verify the user-facing path actually works:'
Write-Host ('  python tools\fetch_engine.py --tag v{0}' -f $Version)
Write-Host ''
Write-Host 'Remember to bump `version` in pyproject.toml too.'
