# How it works

Only worth reading if you want to modify the pack or integrate a different engine.

## Shape

```
┌─ ComfyUI process ──────────────────────────────────────────────┐
│  Python  ──ctypes──▶  ninfer_capi.dll  ──▶  ninfer::Engine     │
│                        (C ABI, ~228 MB)      C++ / CUDA        │
└────────────────────────────────────────────────────────────────┘
```

The engine is a layered static library with a thin CLI on top, so making it embeddable was small:
a `SHARED` CMake target plus ~170 lines of `extern "C"` wrappers. The patch is
`tools/patch/0001-embeddable-c-abi.patch`:

- **adds** `apps/capi/ninfer_capi.cpp` — `ninfer_create` / `ninfer_generate` / `ninfer_destroy` /
  `ninfer_has_vision` / `ninfer_last_error`, plus POD option structs mirroring the C++ request types
- **adds** `apps/capi/test_ninfer_dll.py` — a ctypes smoke test
- **modifies** `apps/CMakeLists.txt` — the `SHARED` target and a post-build step copying its runtime DLLs

ctypes then loads that DLL into ComfyUI's process exactly like a `.pyd`. There is no IPC and no
serialisation, and because both share one CUDA context, destroying the engine returns memory
immediately.

## Why in-process

| | This pack | llama.cpp server nodes | Remote API |
|---|---|---|---|
| Load cost after the first call | 0 s (resident) | spawn / connect | — |
| Per-token overhead | a function call | HTTP | HTTP |
| VRAM released on demand | one node | usually manual | — |
| Model format | `.ninfer` only | GGUF | — |

Trade-off: you need an artifact in `.ninfer` format, and the binary is architecture-specific.

## Two memory worlds

They share one GPU and neither can see the other's accounting:

| Memory | Owner | Who can free it |
|---|---|---|
| the engine's ~14 GiB | native `cudaMalloc` inside the DLL | only `ninfer_destroy` |
| ComfyUI's models and caches | torch's allocator | `unload_all_models()` + `soft_empty_cache()` |

That is why a third-party "clean VRAM" node cannot help with the first row: it reports success
while the engine still fails to start. `NInfer Free VRAM` handles both, and the LLM node's
`free_comfy_vram` switch handles the second row automatically at the moment it matters — right
before the engine is created.

## What the engine's errors actually mean

The engine measures the card with `cudaMemGetInfo` **after** the weights are resident, so
"only 0 bytes are available for runtime capacity" means the card is genuinely full at that point,
not that something failed to be released. Two consequences for the node:

- a failed creation costs a full weight load first, which is why failures take seconds
- `consumed` in the CUDA-graph error is a delta of *process-wide* free memory, so unrelated
  allocations during capture get attributed to the graph — that is the source of the intermittent
  graph-budget failures

`is_arch_mismatch_error` / `is_graph_budget_error` in `_capi.py` route these to different advice.
An architecture mismatch is routed away from the VRAM suggestions entirely, since telling someone
to lower `max_context` when their GPU cannot execute the binary wastes their time.

`auto_recover` implements the retry chain: as configured → with `use_cuda_graph` disabled → the
last configuration that worked. A second engine cannot be built before the first is destroyed
(they would not both fit), so a failed creation would otherwise leave the user with nothing.

## Two details that cost real debugging time

**`__declspec(dllexport)` must be on every function.** `extern "C"` only controls name mangling; it
does not export anything. Written wrongly, the DLL loads successfully while every symbol lookup
fails — which looks exactly like a path problem. Check with
`hasattr(ctypes.CDLL(path), "ninfer_create")` right after building.

**ComfyUI maps saved workflow values onto widgets by position.** Inserting a widget in the middle
of `INPUT_TYPES` silently shifts every later value into the wrong control — a path string landing
in an INT field. New widgets are therefore always appended to the end of `required`.
