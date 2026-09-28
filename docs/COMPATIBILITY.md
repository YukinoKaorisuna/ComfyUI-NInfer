# Compatibility

## The engine binary is architecture-specific

A CUDA binary runs only on the compute capabilities it was compiled for. Measured on the
published Windows engine:

```
> cuobjdump --list-elf bin\ninfer_capi.dll
ELF file    1: ninfer_capi.1.sm_120a.cubin
ELF file    2: ninfer_capi.2.sm_120a.cubin
ELF file    3: ninfer_capi.3.sm_120a.cubin

> cuobjdump --list-ptx bin\ninfer_capi.dll
No PTX file found to extract
```

Three `sm_120a` cubins and **no PTX**. An `a`-suffixed cubin runs only on exactly that
architecture, and without PTX there is no JIT fallback in either direction — so this build runs
**exclusively on RTX 50-series (GB20x)**.

On any other card the driver reports:

```
CUDA error: no kernel image is available for execution on the device
```

The node recognises that message and answers with the build command for your GPU, and
`tools/doctor.py` prints both your compute capability and the architectures actually present in
your engine file.

Check any downloaded engine yourself:

```
python tools/doctor.py
```

## GPU matrix

| GPU | Compute | Prebuilt | Base repo to build from |
|---|---|---|---|
| RTX 5090 / 5080 / 5070 Ti / 5070 | `sm_120` | ✅ **tested** | this repo's default |
| RTX 4090 / 4080 / L4 / L40S | `sm_89` | ❌ | [Ambolio/ninfer-4090-windows](https://github.com/Ambolio/ninfer-4090-windows) (v3) or [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) (v1+v2) |
| RTX 3090 / 3080 / A40 / A6000 | `sm_86` | ❌ | [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) (v2) |
| A100 / A30 | `sm_80` | ❌ | upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) |
| H100 / H200 | `sm_90` | ❌ | upstream |
| RTX 2000-series / T4 | `sm_75` | ❌ | upstream |
| AMD / Intel / Apple | — | ❌ | not possible, the engine is CUDA-only |

Only `sm_120a` on an RTX 5070 Ti has been tested end to end. The kernel launch-threshold tables
were swept on 70–84 SMs (GB20x), so on other architectures expect it to *work*, not to be
optimal. Community builds for other architectures are welcome as releases.

## Other GPUs

The hard part of a Windows build is the MSVC port, and the community builds above already carry
it for exactly those cards. Base your `ninfer_capi` build on one of them rather than starting
from upstream:

```powershell
# RTX 4090 (sm_89)
pwsh -File tools\build_engine.ps1 `
     -RepoUrl https://github.com/Ambolio/ninfer-4090-windows `
     -Branch main -Commit '' -CudaArch 89 `
     -VcpkgPrefix D:\deps\vcpkg\installed\x64-windows

# RTX 3090 (sm_86)
pwsh -File tools\build_engine.ps1 `
     -RepoUrl https://github.com/Don-Chad/ninfer-3090 `
     -Branch main -Commit '' -CudaArch 86 `
     -VcpkgPrefix D:\deps\vcpkg\installed\x64-windows
```

Requirements: VS 2022 with the C++ workload, a CUDA Toolkit that still supports your architecture
(12.8+ for `sm_120a`; CUDA 13 dropped `sm_50`..`sm_70`), CMake, Ninja, git, and a vcpkg prefix
providing FFmpeg + curl. The script clones the repo, applies `tools/patch/0001-embeddable-c-abi.patch`
(the C ABI layer this pack calls) and builds the `ninfer_capi` target.

Copy the result plus its sibling DLLs into `bin/`, then verify:

```powershell
Copy-Item '<build>\apps\*.dll' .\bin\
python tools\doctor.py
```

The bundled patch was cut against the RTX 5070 Ti fork. `git apply --3way` usually copes, but an
older lineage may need the wrapper adapted — it mirrors the field names in
`include/ninfer/types.h`.

**The architecture guard.** Upstream's `CMakeLists.txt` hard-pins the build to `120a` — a
`FATAL_ERROR` fires before compiler detection for anything else. The build script relaxes that
guard automatically after applying the patch, so `-CudaArch 89` reaches the compiler instead of
dying in configure.

**But relaxing the guard is not enough (measured on this machine, CUDA 13.4).** A full
`ninfer_capi` build for `sm_89` fails in `ptxas`:

```
nvfp4_attn_input_w4a4.ptx: error: Feature '.scale_vec::4X' not supported on .target 'sm_89'
ptxas fatal: Ptx assembly aborted due to errors
```

The NVFP4 kernels under `ops/attn_input_proj/nvfp4/` and `ops/gdn_input_proj/nvfp4/` emit PTX
only `sm_100`+ can assemble. On Windows only the two *TMA* translation units are stubbed — these
are not. Supporting `sm_89` / `sm_86` therefore needs one more engine change: guard those
kernels with `#if __CUDA_ARCH__ >= 1200` and fall back to a throws-at-runtime stub (the same
pattern the MSVC build already uses for the TMA units), then every non-Blackwell architecture
compiles. Until that patch exists, non-50-series cards cannot build from current sources.

Note that the community 4090 / 3090 engine builds predate those kernels — which is why their
engines compile. Basing a build on their source lines (with the C ABI patch ported) is the
alternative route.

To cover several cards with one DLL, pass a semicolon list: `-CudaArch '89;120a'`. It costs build
time and binary size.

## Linux

Upstream is Linux-first and the C ABI layer is plain `extern "C"`, so building `ninfer_capi.so`
works. The pack is OS-agnostic: it looks for `ninfer_capi.dll`, `ninfer_capi.so` or
`ninfer_capi.dylib` in `bin/`. Nothing in this repository has been tested on Linux yet.

## macOS

Not possible. The engine is CUDA-only.

## Publishing a build for another architecture

```powershell
pwsh -File tools\make_release.ps1 -CudaArch 89 -Version 0.1.0
```

The architecture goes into the asset name (`ninfer-engine-win-x64-sm89-0.1.0.zip`), which is how
`fetch_engine.py` hands the right build to the right GPU. Attach it to the same release as the
other architectures; assets are told apart by name, not by release.
