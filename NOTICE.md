# Notices and attribution

This repository is a **wrapper**. It contains no inference engine of its own — it loads a
native binary built from a third-party project and calls it through a C ABI. The Python
code here is MIT (see `LICENSE`); everything below is about the binary that gets loaded.

---

## 1. The NInfer engine — Apache License 2.0

| | |
|---|---|
| Project | **NInfer** — https://github.com/Neroued/ninfer, by **Neroued**, the author of the engine |
| Lineage of this binary | https://github.com/toddballinger/ninfer-5080 (an RTX 5080 optimisation fork) → https://github.com/YukinoKaorisuna/ninfer-5070ti (adds the Windows/MSVC port) |
| License | Apache License 2.0 — full text in `THIRD_PARTY/LICENSE-ninfer-Apache-2.0.txt` |
| Copyright | the NInfer authors |

The prebuilt `ninfer_capi.dll` published in this repository's Releases is a build of that
source tree. Neither the author of NInfer nor the maintainers of this pack endorse each other;
this is an independent integration.

### Modifications (Apache-2.0 §4(b) requires this notice)

The engine source was modified to add an embeddable C ABI. The exact change is
`tools/patch/0001-embeddable-c-abi.patch`:

- **added `apps/capi/ninfer_capi.cpp`** — `extern "C"` exports that wrap `ninfer::Engine`:
  `ninfer_create` / `ninfer_generate` / `ninfer_destroy` / `ninfer_has_vision` /
  `ninfer_last_error`, plus POD option structs mirroring the C++ request types.
- **added `apps/capi/test_ninfer_dll.py`** — a ctypes smoke test that proves the engine and
  PyTorch can share one process and one CUDA context.
- **modified `apps/CMakeLists.txt`** — a `SHARED` target named `ninfer_capi`, plus a
  post-build step that copies its runtime DLLs next to it.

No operator, kernel, scheduler, or artifact-format source was touched, so inference output
is identical to the unmodified engine.

---

## 2. Runtime libraries bundled next to the engine

`ninfer_capi.dll` links **dynamically** against the libraries below. They ship unmodified
as separate DLL files in the same directory, which is what satisfies the LGPL's
"allow the user to relink" requirement — you may replace any of them with your own build.

| Library | DLLs | License |
|---|---|---|
| FFmpeg | `avcodec-63`, `avformat-63`, `avutil-61`, `avfilter-12`, `avdevice-63`, `swresample-7`, `swscale-10` | LGPL-2.1-or-later — built via vcpkg **without** the `gpl` / `nonfree` features |
| libcurl | `libcurl` | curl license (MIT-like) |
| zlib | `z` | zlib license |
| pkgconf | `pkgconf-8` | ISC (a build-time dependency that got copied into the runtime folder; safe to delete) |

Full texts: `THIRD_PARTY/`.

---

## 3. Components statically linked inside the engine

All permissive; included here for completeness.

| Component | License |
|---|---|
| cpp-httplib | MIT |
| nlohmann/json | MIT |
| utf8proc | MIT |

---

## 4. What is **not** distributed here

- **Model artifacts (`.ninfer`).** Separate downloads, Apache-2.0 (derived from Qwen), not
  distributed here. Sizes differ and the files are **not** interchangeable — two of them even
  share a filename:

  | Artifact | Repository | Published by |
  |---|---|---|
  | Qwen3.8-27B, 15.33 GiB — the 16 GB release profile this pack targets | https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080 | [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080) |
  | Qwen3.8-27B, 19.03 GiB | https://huggingface.co/neroued/Qwen3.8-27B-NInfer | NInfer's author |
  | Qwen3.6-35B-A3B | https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer | NInfer's author |
  | Qwen3.6-27B | https://huggingface.co/neroued/Qwen3.6-27B-NInfer | NInfer's author |
  | Qwen3.8-27B NVFP4 | https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer | NInfer's author |
  | Qwen3.6-27B NVFP4 | https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer | NInfer's author |
  | Ornith-1.5-35B-A3B | https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer | community |

  Every benchmark figure and the validated VRAM configuration in this pack come from the first
  one.

- **The chat web UI and any uncensored/abliterated derivatives** that exist in the
  development fork. They are not part of this node pack.

---

## 5. Trademarks

"Qwen" and model names belong to their respective owners. "ComfyUI" belongs to its
authors. No affiliation or endorsement is implied by this integration.
