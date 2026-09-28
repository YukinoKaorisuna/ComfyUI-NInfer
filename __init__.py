"""NInfer nodes for ComfyUI — run a NInfer engine inside ComfyUI's own process.

The engine is a C++/CUDA library shipped as ``ninfer_capi.dll``; this pack loads it
through ctypes. No HTTP server, no child process: the engine shares ComfyUI's address
space and CUDA context, so a model teardown returns VRAM straight to the diffusion
pipeline, and a resident engine costs 0 s to reuse.

Nodes:

  NInfer Local LLM      generate text from a prompt, optionally with an image (vision)
  NInfer Free VRAM      release the engine(s); pass-through, so it can sit on any link

See README.md for install steps and the VRAM budget on a 16 GB card.
"""

from __future__ import annotations

import time

from ._capi import (
    KV_DTYPES,
    NinferError,
    RequestOptions,
    acquire,
    guidance_for,
    loaded_count,
    release_all,
    tensor_to_jpeg,
    vram_snapshot,
)
# Aliased on purpose: NinferLocalLLM exposes a *widget* named free_comfy_vram, which would
# shadow a plain import of the same name inside that method's scope.
from ._capi import free_comfy_vram as _free_comfy_vram
from ._paths import (
    ARCH_BUILD_FLAG,
    ENV_DLL,
    ENV_MODEL_DIRS,
    MODEL_SOURCES,
    build_arch_hint,
    device_arch,
    device_name,
    dll_setup_hint,
    model_choices,
    model_setup_hint,
    resolve_dll,
    resolve_model,
)

#: Compute capabilities the published engine binary actually contains. Verified with
#: ``cuobjdump --list-elf bin/ninfer_capi.dll`` -> three sm_120a cubins, no PTX.
PREBUILT_ARCHES = ("sm_120",)

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior prompt engineer for image and video generation models. "
    "When asked to write a prompt, return only the prompt text itself."
)

DEFAULT_USER_PROMPT = "Rewrite this as a rich, detailed prompt: a cat on a roof."

#: Shown in the dropdown when nothing was found, so the node still loads and the user gets
#: an actionable message at run time instead of a red box at import time.
_NO_MODEL = "<no .ninfer model found — see README>"


def _mib(n: float) -> str:
    return f"{n / 2 ** 20:.0f} MiB"


class NinferLocalLLM:
    """Single-turn local generation, with an optional image for vision models."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = model_choices() or [_NO_MODEL]
        return {
            "required": {
                # Populated by scanning ComfyUI/models/LLM and friends. Adding a model
                # needs a ComfyUI restart for the dropdown to refresh.
                "model": (choices, {"default": choices[0]}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "user_prompt": ("STRING", {"multiline": True, "default": DEFAULT_USER_PROMPT}),
                "max_context": ("INT", {"default": 4096, "min": 2048, "max": 131072, "step": 1024}),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "vision": ("BOOLEAN", {"default": True}),
                "mtp_draft_tokens": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "kv_dtype": (list(KV_DTYPES.keys()), {"default": "bf16"}),
                "embedding_host": ("BOOLEAN", {"default": True}),
                "image_max_side": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                # --- engine-level switches ---------------------------------------------
                # Everything from here down is fixed at engine construction time, so
                # changing one costs a reload. New widgets MUST be appended at the end:
                # ComfyUI maps a saved workflow's widgets_values onto widgets BY POSITION,
                # so inserting one in the middle silently shifts every later value into
                # the wrong control (e.g. a path landing in an INT field).
                "use_cuda_graph": ("BOOLEAN", {"default": True}),
                "free_comfy_vram": ("BOOLEAN", {"default": True}),
                "auto_recover": ("BOOLEAN", {"default": True}),
                # --- overrides for unusual layouts -------------------------------------
                "model_path_override": ("STRING", {"default": "", "multiline": False}),
                "dll_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "info")
    FUNCTION = "generate"
    CATEGORY = "NInfer"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # A saved workflow may hold a model label that no longer exists (the file was moved
        # or renamed). Combo validation would then red-box the whole graph; we resolve the
        # value at run time instead and raise a message that says how to fix it.
        return True

    def generate(self, model, system_prompt, user_prompt, max_context, max_tokens,
                 temperature, top_k, top_p, min_p, presence_penalty, frequency_penalty,
                 seed, enable_thinking, vision, mtp_draft_tokens, kv_dtype, embedding_host,
                 image_max_side, keep_loaded, use_cuda_graph=True, free_comfy_vram=True,
                 auto_recover=True, model_path_override="", dll_path="", image=None):

        requested = model_path_override.strip() or model
        model_path = resolve_model(requested)
        if model_path is None:
            raise NinferError(model_setup_hint(requested))

        cfg = {
            "dll_path": dll_path.strip(),
            "model_path": model_path,
            "device": 0,
            "max_context": int(max_context),
            "speculative_mtp": int(mtp_draft_tokens),
            "kv_tokens": 0,                       # 0 -> follow max_context
            "enable_vision": bool(vision),
            "vision_max_tokens": 0,
            "embedding_host": bool(embedding_host),
            "kv_dtype": kv_dtype,
            "use_cuda_graph": bool(use_cuda_graph),
        }

        before = vram_snapshot()
        load_started = time.perf_counter()
        engine, notes = acquire(
            cfg,
            free_comfy_first=bool(free_comfy_vram),
            auto_recover=bool(auto_recover),
        )
        load_seconds = time.perf_counter() - load_started
        after = vram_snapshot()
        has_vision = engine.has_vision()

        jpeg = None
        if image is not None:
            if not has_vision:
                raise NinferError(
                    "An image is connected but the resident engine has no vision encoder.\n"
                    "Set vision = true, then restart ComfyUI (or run the Free VRAM node) so a "
                    "new engine gets created.\n"
                    "连了图像，但当前引擎没有 vision。把 vision 打开，然后重启 ComfyUI"
                    "（或跑一次「释放显存」节点）让它重建。"
                )
            jpeg = tensor_to_jpeg(image, max_side=int(image_max_side))

        request = RequestOptions(
            max_new_tokens=int(max_tokens),
            enable_thinking=1 if enable_thinking else 0,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            min_p=float(min_p),
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty),
            seed=int(seed),
        )

        gen_started = time.perf_counter()
        try:
            text = engine.generate(system_prompt, user_prompt, image=jpeg, req=request)
        except NinferError as error:
            # Some failures (an architecture mismatch, for instance) only surface on the
            # first kernel launch rather than at engine construction, so they need the same
            # translation as a creation failure would get.
            raise NinferError(guidance_for(str(error), cfg)) from None
        gen_seconds = time.perf_counter() - gen_started

        released = False
        if not keep_loaded:
            release_all()
            released = True

        vram = ""
        if before is not None and after is not None:
            vram = (f"vram {_mib(before[0])} → {_mib(after[0])} free "
                    f"(engine ≈ {_mib(max(before[0] - after[0], 0))})")
        elif after is not None:
            vram = f"vram {_mib(after[0])} free"
        if notes:
            vram = f"{vram} | {notes}" if vram else notes

        info = (
            f"load {load_seconds:.2f}s | gen {gen_seconds:.2f}s | {len(text)} chars\n"
            f"model={model_path}\n"
            f"ctx={max_context} | vision={has_vision} | MTP={mtp_draft_tokens} | kv={kv_dtype} | "
            f"graph={'on' if use_cuda_graph else 'off'} | emb_host={'on' if embedding_host else 'off'}\n"
            f"thinking={'on' if enable_thinking else 'off'} | "
            f"image={'jpeg %d B' % len(jpeg) if jpeg else 'none'} | "
            f"{'released' if released else 'engine kept loaded'}\n"
            f"{vram}"
        )
        return (text, info)


class NinferUnload:
    """Release the cached engine(s) so a diffusion pass can take the whole card.

    The wildcard in/out pair is a pass-through, so this node can be spliced into the middle
    of any existing link (e.g. right before a sampler) without changing the types around
    it — and it doubles as a standalone cleanup node when nothing is connected.

    Releasing the engine is something only this node can do: the engine's ~14 GiB is native
    (plain ``cudaMalloc`` inside the DLL), so it is invisible to torch and no ComfyUI-side
    "clean VRAM" node can touch it. ComfyUI's own residence is what the ``offload_*``
    switches handle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offload_model": ("BOOLEAN", {"default": True}),
                "offload_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Wildcard: accept anything so it can sit on any link.
                "anything": ("*",),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "status")
    FUNCTION = "unload"
    CATEGORY = "NInfer"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A side-effect node must run on *every* queue. NaN never compares equal, so
        # ComfyUI treats it as changed instead of serving a cached result — which would
        # silently skip the release and leave the card occupied.
        return float("nan")

    def unload(self, offload_model=True, offload_cache=True, anything=None):
        before = loaded_count()
        snap_before = vram_snapshot()
        release_all()
        note = _free_comfy_vram(unload_models=bool(offload_model),
                                empty_cache=bool(offload_cache))
        snap_after = vram_snapshot()

        if snap_before is not None and snap_after is not None:
            status = (f"released {before} engine(s); "
                      f"vram {_mib(snap_before[0])} → {_mib(snap_after[0])} free")
        else:
            status = f"released {before} engine(s)"
        if note:
            status = f"{status}; {note}"
        return (anything, status)


NODE_CLASS_MAPPINGS = {
    "NinferLocalLLM": NinferLocalLLM,
    "NinferUnload": NinferUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NinferLocalLLM": "NInfer Local LLM (.ninfer / Qwen3.8)",
    "NinferUnload": "NInfer Free VRAM (pass-through)",
}

# --------------------------------------------------------------------------------------
# Import-time sanity check — this is the onboarding path most people will see first.
# --------------------------------------------------------------------------------------
# Warn loudly but do NOT raise: raising here would make ComfyUI drop the whole pack, and
# the user would see nothing in the menu to click on. The node raises a detailed error at
# run time instead.

_engine = resolve_dll()
_arch = device_arch()
_models = model_choices()

_CONTEXT = (f"{device_name() or 'no CUDA device'} ({_arch or 'unknown'}), "
            f"{len(_models)} model(s)")


def _banner(title: str, body: str) -> None:
    print("\n[NInfer] " + "=" * 74)
    print(f"[NInfer] {title}")
    for line in body.splitlines():
        print(f"[NInfer] {line}")
    print("[NInfer] " + "=" * 74 + "\n")


if _engine is None and _arch is not None and _arch not in PREBUILT_ARCHES:
    # Stop the user before they download 250 MB that cannot possibly run.
    _banner(f"Your GPU is not supported by the prebuilt engine. Detected: {_CONTEXT}",
            build_arch_hint())
elif _engine is None:
    _banner(f"The engine binary was not found, so the nodes will not run yet. ({_CONTEXT})",
            "Fix it with:  python tools/fetch_engine.py\n\n" + dll_setup_hint())
elif not _models:
    _banner(f"Engine is ready but no .ninfer model was found. ({_CONTEXT})",
            f"Download a model and drop it into ComfyUI/models/LLM, then restart ComfyUI.\n"
            f"Pick by file size - the sources are not interchangeable:\n"
            + "".join(f"  {url}   [{note}]\n" for url, note in MODEL_SOURCES)
            + f"Or point at your own folder:  set {ENV_MODEL_DIRS}=D:\\models")
else:
    print(f"[NInfer] Ready — {_CONTEXT}, engine at {_engine}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
