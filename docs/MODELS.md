# Models

`.ninfer` artifacts only. GGUF and safetensors are not supported — they belong to a different
engine, and the engine cannot read them at all.

## Official artifacts

Published by NInfer's author, [Neroued](https://huggingface.co/neroued). All Apache-2.0 (derived
from Qwen), all including vision and MTP.

| File | Size | Repository | Also includes |
|---|---|---|---|
| `qwen3_8_27b.ninfer` | 19.03 GiB | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | DFlash2 |
| `qwen3_6_35b_a3b.ninfer` | 21.23 GiB | [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | DFlash — MoE, fastest decode |
| `qwen3_6_27b.ninfer` | 16.29 GiB | [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | smallest official artifact |
| `qwen3_8_27b_nvfp4.ninfer` | 22.09 GiB | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | ⚠️ `sm_120a` only, unusable on Windows |
| `qwen3_6_27b_nvfp4.ninfer` | 17.07 GiB | [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | ⚠️ same |

Each repository carries an `artifact-manifest.json` with the authoritative size and sha256.
Verify against that, **not** against the download's ETag — an ETag is not a file hash.

## Community artifacts

| File | Repository | Note |
|---|---|---|
| `ornith_1_5_35b_a3b.ninfer` | [huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer) | container v2 — required for RTX 3090 / 3080 |
| `qwen3_8_27b_uncensored.ninfer` | [YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer](https://huggingface.co/YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer) | abliterated, self-converted |

## Mirrors

The files are 16–23 GiB, so the origin matters:

| Mirror | Notes |
|---|---|
| `aifasthub.com` | fastest in practice — ~11 MB/s on a single connection, far more with `aria2c -x16` |
| `hf-mirror.com` | reliable fallback, roughly 10× slower |
| `huggingface.co` | authoritative; may need a proxy |
| `modelscope.cn` | fast, but only mirrors what someone happened to upload — search before relying on it |

Replace the host in a repository URL, e.g.

```
https://aifasthub.com/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer
```

Resumable, multi-connection download:

```
aria2c -x16 -s16 -k1M -c --file-allocation=none -d D:\models -o qwen3_8_27b.ninfer <url>
curl.exe -L -C - -o qwen3_8_27b.ninfer <url>          # if aria2c is unavailable
```

## Where to put them

`ComfyUI/models/LLM/` — restart ComfyUI and they appear in the node's `model` dropdown. Extra
folders can be added:

```bat
set NINFER_MODEL_DIRS=D:\models;E:\another\folder
```

## Two gotchas

### The container version has to match the engine

Artifacts are packaged in a versioned container (v1 / v2 / v3, v3 being current). Older engine
builds accept only v1/v2 and will refuse a v3 file.

The trap: **an artifact's filename does not change between container versions while its contents
do.** A same-named file you download today may not load in a build from last month. Upstream ships
a local v2 → v3 upgrade path that avoids re-downloading the weights — see its
`docs/weight-conversion.md`.

If an artifact fails to load, suspect the container version before suspecting VRAM.

### NVFP4 artifacts do not work on Windows

Two TMA translation units cannot be compiled by MSVC, so the Windows build ships a stub and rejects
NVFP4 artifacts at run time rather than failing the build. Use a `groupwise-int` artifact instead.

## Sizes and VRAM

Every VRAM and performance figure in the node pack's README was measured on the 15.33 GiB 16 GB
artifact. The larger ones do not fit a 16 GB card:

| Artifact | On disk | Fits a 16 GB card? |
|---|---|---|
| `qwen3_6_27b.ninfer` | 16.29 GiB | no — upstream lists these as needing ≥24 GB |
| `qwen3_8_27b.ninfer` | 19.03 GiB | no |
| `qwen3_6_35b_a3b.ninfer` | 21.23 GiB | no |

On 16 GB, use the 15.33 GiB profile above.

As a rule of thumb the VRAM a file occupies is **about 0.93× its size on disk** (measured: 15.33 GiB
file → ~14.27 GiB resident), plus ~0.4–1.4 GiB of engine runtime and ~1.3 GiB for ComfyUI itself.
`tools/doctor.py` prints this estimate for whatever artifacts it finds.
