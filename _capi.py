"""ctypes binding for ``ninfer_capi.dll`` — run a NInfer engine inside ComfyUI's process.

Loading the DLL here (rather than talking to a server or a child process) means the engine
shares ComfyUI's address space and CUDA context: no IPC per token, and destroying the
engine returns its VRAM to the diffusion pipeline immediately.

Creating an engine reads a ~15 GB artifact, so engines are cached and reused across node
executions until the caller releases them.

VRAM reality check
------------------
The engine's footprint is measured by the engine itself with ``cudaMemGetInfo`` *after*
the weights are resident, so a message like "only 0 bytes are available for runtime
capacity" means the card is genuinely full at that moment — not that something failed to
be released. Two memory worlds share one GPU:

* the engine's ~14 GiB, allocated by native ``cudaMalloc`` inside the DLL;
* ComfyUI's own residence, owned by torch's allocator.

A generic "clean VRAM" node only ever touches the second, which is why it cannot make room
for the first. :func:`free_comfy_vram` exists so this node can do that at the exact moment
it matters.
"""

from __future__ import annotations

import ctypes
import io
import os
import re
import threading

from ._paths import build_arch_hint, device_arch, dll_setup_hint, resolve_dll

#: widget value -> NinferCreateOptions::kv_dtype
KV_DTYPES = {"bf16": 0, "int8": 1, "int4": 2}

#: The engine returns UTF-8 text into this buffer. 256 KiB is far more than any
#: prompt-length answer needs.
_OUT_CAP = 1 << 18


# --------------------------------------------------------------------------------------
# C structures — field order and types must mirror apps/capi/ninfer_capi.cpp exactly.
# --------------------------------------------------------------------------------------

class CreateOptions(ctypes.Structure):
    _fields_ = [
        ("artifact_path", ctypes.c_char_p),
        ("device", ctypes.c_int32),
        ("max_context", ctypes.c_int32),
        ("speculative_mtp", ctypes.c_int32),
        ("kv_tokens", ctypes.c_int32),
        ("enable_vision", ctypes.c_int32),
        ("vision_max_tokens", ctypes.c_int32),
        ("embedding_host", ctypes.c_int32),
        ("kv_dtype", ctypes.c_int32),
        ("use_cuda_graph", ctypes.c_int32),
    ]


class RequestOptions(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int32),
        ("enable_thinking", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("seed", ctypes.c_int64),
    ]


class NinferError(RuntimeError):
    """Raised when the native engine reports a failure."""


# --------------------------------------------------------------------------------------
# Library handles + engine cache
# --------------------------------------------------------------------------------------

_lock = threading.RLock()
_libs: dict[str, ctypes.CDLL] = {}
_engines: dict[tuple, "Engine"] = {}
#: The last configuration that actually produced a live engine. Used as the fallback when
#: a newly requested configuration cannot fit: without it, changing one knob on a card
#: that is already at capacity would destroy the working engine and leave nothing behind.
_last_good_cfg: dict | None = None
#: ``os.add_dll_directory`` returns a handle that must stay alive for the directory to
#: remain on the DLL search path.
_dll_dir_handles: list = []


def _bind(lib: ctypes.CDLL) -> None:
    lib.ninfer_last_error.restype = ctypes.c_char_p
    lib.ninfer_last_error.argtypes = []

    lib.ninfer_create.restype = ctypes.c_void_p
    lib.ninfer_create.argtypes = [ctypes.POINTER(CreateOptions)]

    lib.ninfer_generate.restype = ctypes.c_int32
    lib.ninfer_generate.argtypes = [
        ctypes.c_void_p,             # handle
        ctypes.c_char_p,             # system text (UTF-8, may be None)
        ctypes.c_char_p,             # user text (UTF-8)
        ctypes.c_char_p,             # image bytes (may be None)
        ctypes.c_int32,              # image length
        ctypes.c_char_p,             # image mime (may be None)
        ctypes.POINTER(RequestOptions),
        ctypes.c_char_p,             # out buffer
        ctypes.c_int32,              # out capacity
    ]

    lib.ninfer_has_vision.restype = ctypes.c_int32
    lib.ninfer_has_vision.argtypes = [ctypes.c_void_p]

    lib.ninfer_destroy.restype = None
    lib.ninfer_destroy.argtypes = [ctypes.c_void_p]


def _load_library(dll_path: str) -> ctypes.CDLL:
    with _lock:
        lib = _libs.get(dll_path)
        if lib is not None:
            return lib

        # The engine's runtime imports (avformat-*.dll, libcurl.dll, ...) live beside it.
        # add_dll_directory scopes that search to this process without touching PATH.
        add_dir = getattr(os, "add_dll_directory", None)
        if add_dir is not None:
            _dll_dir_handles.append(add_dir(os.path.dirname(dll_path)))

        try:
            lib = ctypes.CDLL(dll_path)
        except OSError as exc:
            # "DLL load failed" is the least informative error in this whole stack, so
            # translate the common causes instead of letting ctypes' one-liner through.
            raise NinferError(_library_load_hint(dll_path, exc)) from None
        _bind(lib)
        _libs[dll_path] = lib
        return lib


def _library_load_hint(dll_path: str, exc: OSError) -> str:
    siblings = []
    directory = os.path.dirname(dll_path)
    try:
        if os.path.isdir(directory):
            siblings = sorted(
                name for name in os.listdir(directory)
                if name.lower().endswith((".dll", ".so", ".dylib"))
            )
    except OSError:
        pass

    missing_deps = len(siblings) <= 1          # only the engine itself is present
    return (
        f"Could not load the engine binary / 无法加载引擎二进制:\n  {dll_path}\n"
        f"  {type(exc).__name__}: {exc}\n"
        "\n"
        "Most common causes / 最常见原因:\n"
        "  1. Its runtime libraries are missing from the same folder.\n"
        "     引擎的运行时库不在同一个目录里。\n"
        f"     Files found there / 该目录下的库文件: "
        f"{', '.join(siblings) if siblings else '(none)'}\n"
        + ("     -> re-run: python tools/fetch_engine.py\n"
           if missing_deps else "")
        + "  2. Visual C++ redistributable missing -> install the latest VC++ x64 runtime.\n"
          "     缺少 VC++ 运行库。\n"
        "  3. Architecture mismatch (32/64-bit, or an engine built for another platform).\n"
          "     架构不匹配。\n"
        "  4. The file is blocked: right-click the .dll -> Properties -> Unblock.\n"
          "     文件被系统标记为「已阻止」。\n"
        "\n"
        "Full check: python tools/doctor.py"
    )


class Engine:
    """An owned engine. Cheap to hold, expensive to create."""

    def __init__(self, lib: ctypes.CDLL, handle: int, key: tuple):
        self._lib = lib
        self._handle = handle
        self.key = key

    def has_vision(self) -> bool:
        if not self._handle:
            return False
        return bool(self._lib.ninfer_has_vision(self._handle))

    def generate(self, system: str, user: str, image: bytes | None = None,
                 image_mime: str = "image/jpeg", req: RequestOptions | None = None) -> str:
        if not self._handle:
            raise NinferError("engine has been released")
        out = ctypes.create_string_buffer(_OUT_CAP)
        written = self._lib.ninfer_generate(
            self._handle,
            system.encode("utf-8") if system else None,
            (user or "").encode("utf-8"),
            image,
            len(image) if image else 0,
            image_mime.encode("utf-8") if image else None,
            ctypes.byref(req) if req is not None else None,
            out,
            _OUT_CAP,
        )
        if written < 0:
            raise NinferError(self._lib.ninfer_last_error().decode("utf-8", "replace"))
        return out.value.decode("utf-8", "replace")

    def close(self) -> None:
        if self._handle:
            self._lib.ninfer_destroy(self._handle)
            self._handle = None


# --------------------------------------------------------------------------------------
# VRAM helpers
# --------------------------------------------------------------------------------------

def _mib(n: float) -> str:
    return f"{n / 2 ** 20:.0f} MiB"


def vram_snapshot() -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` for device 0, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return None


def _empty_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - purely a best-effort hint to the allocator
        pass


def free_comfy_vram(*, unload_models: bool = True, empty_cache: bool = True) -> str:
    """Ask ComfyUI to give up its own models/cache before the engine takes the card.

    This is the half a generic "clean VRAM" node cannot help with *for us*: the engine's
    memory is native and invisible to torch, but ComfyUI's residence is exactly what
    decides whether the weights still fit. Doing it here — immediately before
    ``ninfer_create`` — is what makes it actually count.

    Returns a short human-readable note about what was freed.
    """
    notes: list[str] = []

    if unload_models:
        try:
            import comfy.model_management as mm

            mm.unload_all_models()
            notes.append("unloaded ComfyUI models / 卸载 ComfyUI 模型")
        except Exception as exc:  # noqa: BLE001 - best effort only
            notes.append(f"unload skipped / 卸载跳过 ({type(exc).__name__})")

    if empty_cache:
        freed = False
        try:
            import comfy.model_management as mm

            soft_empty = getattr(mm, "soft_empty_cache", None)
            if soft_empty is not None:
                soft_empty(force=True)
                freed = True
        except Exception:  # noqa: BLE001 - fall through to the plain torch call
            pass
        if not freed:
            _empty_torch_cache()
        notes.append("cache emptied / 清空缓存")

    return "; ".join(notes)


# --------------------------------------------------------------------------------------
# Error translation — the native messages carry exact byte counts; surface them usefully.
# --------------------------------------------------------------------------------------

_SHORTFALL_RE = re.compile(r"requires (\d+) bytes, but only (\d+) bytes")
_WEIGHTS_RE = re.compile(
    r"model weights require (\d+) bytes of device memory, but only (\d+) bytes are free")
_GRAPH_RE = re.compile(r"CUDA Graph preparation consumed (\d+) bytes, exceeding the planned "
                       r"allowance of (\d+) bytes")
_ARCH_RE = re.compile(r"no kernel image is available for execution on the device", re.I)


def is_graph_budget_error(message: str) -> bool:
    """True when the failure is the engine's CUDA-graph allowance being too small.

    Worth retrying without the graph: that only ever lowers the required budget, so the
    retry succeeds in exactly the situation where the original attempt cannot.
    """
    return _GRAPH_RE.search(message) is not None


def is_arch_mismatch_error(message: str) -> bool:
    """True when the engine has no cubin for this GPU's compute capability.

    Measured fact: the published Windows engine contains only sm_120a cubins and no PTX
    (``cuobjdump --list-elf`` shows three sm_120a entries, ``--list-ptx`` finds none), so it
    runs exclusively on RTX 50-series. On any other card the CUDA driver reports this error.
    Retrying or freeing VRAM cannot help — the binary has to be rebuilt for that arch.
    """
    return _ARCH_RE.search(message) is not None


def _format_failure(message: str, cfg: dict) -> str:
    """Turn a native error into a bilingual message with the deficit and next steps."""
    if is_arch_mismatch_error(message):
        # Not a memory problem at all — the VRAM advice below would be actively misleading.
        return "\n".join([
            message, "",
            "This is an architecture mismatch, not a VRAM problem.",
            "这是架构不匹配，不是显存问题。", "",
            build_arch_hint(),
        ])

    facts_en: list[str] = []
    facts_zh: list[str] = []

    match = _SHORTFALL_RE.search(message)
    if match:
        need, have = int(match.group(1)), int(match.group(2))
        facts_en.append(f"short by {_mib(need - have)} "
                        f"(runtime needs {_mib(need)}, only {_mib(have)} left after weights)")
        facts_zh.append(f"还差 {_mib(need - have)}"
                        f"（运行期需要 {_mib(need)}，权重加载完后只剩 {_mib(have)}）")

    match = _WEIGHTS_RE.search(message)
    if match:
        need, have = int(match.group(1)), int(match.group(2))
        facts_en.append(f"the weights alone do not fit: short by {_mib(need - have)}")
        facts_zh.append(f"权重本身就放不下，还差 {_mib(need - have)}")

    match = _GRAPH_RE.search(message)
    if match:
        used, allow = int(match.group(1)), int(match.group(2))
        facts_en.append(f"CUDA Graph used {_mib(used)} against a {_mib(allow)} allowance")
        facts_zh.append(f"CUDA Graph 实际用了 {_mib(used)}，只预留了 {_mib(allow)}")

    snap = vram_snapshot()
    if snap is not None:
        free, total = snap
        facts_en.append(f"GPU currently free: {_mib(free)} / {_mib(total)}")
        facts_zh.append(f"此刻 GPU 空闲 {_mib(free)} / 共 {_mib(total)}")

    return "\n".join(
        [message, ""]
        + facts_en
        + ["", "Engine creation failed. Try, in order:",
           "  1. free_comfy_vram = true — unloads ComfyUI's models first (most effective)",
           "  2. use_cuda_graph = false — relaxes the graph budget; also helps when tight",
           f"  3. lower max_context (now {cfg.get('max_context')}; try 3072 or 2048)",
           f"  4. kv_dtype = int8 (now {cfg.get('kv_dtype')}; halves the KV footprint)",
           "  5. vision = false (saves ~2 GiB when you are not using an image)",
           "  6. a smaller .ninfer artifact (~3.2 BPW, ~11 GiB) fits alongside diffusion",
           ]
        + ["", "引擎创建失败，按顺序试：",
           "  1. 勾上 free_comfy_vram —— 创建前先卸载 ComfyUI 模型，最有效",
           "  2. 关掉 use_cuda_graph —— 放宽 graph 预算，显存紧时也常管用",
           f"  3. 降 max_context（当前 {cfg.get('max_context')}，试 3072 或 2048）",
           f"  4. kv_dtype 改 int8（当前 {cfg.get('kv_dtype')}，KV 显存减半）",
           "  5. 关掉 vision（不看图时省约 2 GiB）",
           "  6. 换更小量化的 .ninfer（约 3.2 BPW ≈ 11 GiB），才能和扩散模型同卡",
           ]
        + ["", "See README.md → Troubleshooting for the full VRAM budget table.",
           "完整显存预算表见 README.zh-CN.md"]
    )


def guidance_for(message: str, cfg: dict | None = None) -> str:
    """Public wrapper: native error text -> bilingual, actionable guidance.

    Used both when the engine fails to *create* and when it fails *during* generation
    (an arch mismatch, for instance, can surface on the first kernel launch rather than
    at load time).
    """
    return _format_failure(message, cfg or {})


# --------------------------------------------------------------------------------------
# Engine acquisition
# --------------------------------------------------------------------------------------

def _key_for(cfg: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()))


def _create(cfg: dict, key: tuple) -> Engine:
    dll_path = resolve_dll(cfg.get("dll_path"))
    if dll_path is None:
        raise NinferError(dll_setup_hint())

    lib = _load_library(dll_path)
    options = CreateOptions(
        artifact_path=cfg["model_path"].encode("utf-8"),
        device=int(cfg.get("device", 0)),
        max_context=int(cfg.get("max_context", 4096)),
        speculative_mtp=int(cfg.get("speculative_mtp", 0)),
        kv_tokens=int(cfg.get("kv_tokens", 0)),
        enable_vision=1 if cfg.get("enable_vision") else 0,
        vision_max_tokens=int(cfg.get("vision_max_tokens", 0)),
        embedding_host=1 if cfg.get("embedding_host") else 0,
        kv_dtype=KV_DTYPES.get(cfg.get("kv_dtype", "bf16"), 0),
        use_cuda_graph=1 if cfg.get("use_cuda_graph", True) else 0,
    )
    handle = lib.ninfer_create(ctypes.byref(options))
    if not handle:
        raise NinferError(lib.ninfer_last_error().decode("utf-8", "replace"))
    return Engine(lib, handle, key)


def _replace_with(candidate: dict) -> Engine:
    """Drop whatever is resident, then build ``candidate`` and cache it."""
    global _last_good_cfg

    with _lock:
        for other in list(_engines.values()):
            other.close()
        _engines.clear()
    _empty_torch_cache()

    engine = _create(candidate, _key_for(candidate))
    with _lock:
        _engines[_key_for(candidate)] = engine
    _last_good_cfg = dict(candidate)
    return engine


def acquire(cfg: dict, *, free_comfy_first: bool = False,
            auto_recover: bool = True) -> tuple[Engine, str]:
    """Return ``(engine, notes)`` for ``cfg``, creating or replacing as needed.

    ``notes`` is a human-readable log of anything non-obvious that happened (ComfyUI models
    unloaded, a fallback configuration used, ...) so the node can surface it to the user.

    A *different* configuration replaces the resident engine instead of adding a second
    one: two engines would need roughly twice the VRAM and neither would fit. Changing
    max_context / vision / kv_dtype therefore costs one reload, which is expected — those
    are fixed at engine construction time.

    On a 16 GB card the fit margin is a few hundred MiB, so a new configuration can fail
    even though the previous one worked. When that happens we retry with the CUDA graph
    disabled (which only lowers the required budget) and finally fall back to the last
    configuration that did fit, rather than leaving the user with no engine at all.
    """
    key = _key_for(cfg)
    with _lock:
        engine = _engines.get(key)
        if engine is not None:
            return engine, ""

    notes: list[str] = []
    if free_comfy_first:
        notes.append(free_comfy_vram())

    try:
        return _replace_with(cfg), " | ".join(notes)
    except NinferError as exc:
        first_error = str(exc)

    if is_arch_mismatch_error(first_error):
        # No retry can help: the binary simply has no code for this GPU.
        raise NinferError(_format_failure(first_error, cfg)) from None

    if not auto_recover:
        raise NinferError(_format_failure(first_error, cfg)) from None

    last_error = first_error
    if is_graph_budget_error(first_error):
        without_graph = dict(cfg, use_cuda_graph=False)
        notes.append("graph budget exceeded → retried with use_cuda_graph=false / "
                     "CUDA Graph 超预算 → 已关闭 use_cuda_graph 重建")
        try:
            return _replace_with(without_graph), " | ".join(notes)
        except NinferError as exc:
            last_error = str(exc)

    if _last_good_cfg is not None and _key_for(_last_good_cfg) != key:
        try:
            engine = _replace_with(_last_good_cfg)
            notes.append("new config did not fit → fell back to the last working one / "
                         "新配置装不下 → 已回退到上次可用配置")
            return engine, " | ".join(notes)
        except NinferError:  # noqa: PERF203 - fall through to the original diagnosis
            pass

    raise NinferError(_format_failure(last_error, cfg)) from None


def release_all() -> int:
    """Destroy every cached engine and hand its VRAM back to ComfyUI."""
    with _lock:
        count = len(_engines)
        for engine in _engines.values():
            engine.close()
        _engines.clear()
    _empty_torch_cache()
    return count


def loaded_count() -> int:
    return len(_engines)


def tensor_to_jpeg(image, max_side: int = 1024, quality: int = 95) -> bytes:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float in 0..1) into JPEG bytes.

    JPEG rather than PNG on purpose: the bundled FFmpeg build has no PNG decoder, so a PNG
    payload would be rejected by the engine's media path.
    """
    import numpy as np
    from PIL import Image

    array = image[0].detach().cpu().float().numpy()
    array = (array.clip(0.0, 1.0) * 255.0).round().astype("uint8")
    pil = Image.fromarray(array)
    if max_side and max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    pil.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
