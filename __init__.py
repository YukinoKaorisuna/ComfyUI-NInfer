"""NInfer nodes for ComfyUI — run a NInfer engine inside ComfyUI's own process.

The engine is a C++/CUDA library shipped as ``ninfer_capi.dll``; this pack loads it
through ctypes. No HTTP server, no child process: the engine shares ComfyUI's address
space and CUDA context, so a model teardown returns VRAM straight to the diffusion
pipeline, and a resident engine costs 0 s to reuse.

Nodes:

  NInfer Local LLM      generate text from a prompt, with any number of images
                        (vision) or a whole VIDEO (the engine's FFmpeg decodes it,
                        samples frames at the model's video fps, and applies
                        temporal position encoding) — connect one IMAGE, a batch,
                        and/or a video
  NInfer Free VRAM      release the engine(s); pass-through, so it can sit on any link

See README.md for install steps and the VRAM budget on a 16 GB card.
"""

from __future__ import annotations

import os
import time

from . import _state
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

#: Compute capabilities the engine binary can actually contain. Verified with
#: ``cuobjdump --list-elf bin/ninfer_capi.dll``. The published binary ships sm_120a only;
#: locally built engines (e.g. the sm_89 build for RTX 40-series) drop the same name into
#: bin/, so the arch check must accept it too.
PREBUILT_ARCHES = ("sm_120", "sm_89")

#: Persona used ONLY for the i2i / prompt-rewriter path (images connected and the
#: system_prompt widget left empty). Never applied to plain-text or video-only
#: requests — those would otherwise inherit the "prompt engineer" tone, which is
#: exactly the contamination this default used to cause.
I2I_SYSTEM_PROMPT = (
    "You are a senior prompt engineer for image and video generation models. "
    "When asked to write a prompt, return only the prompt text itself."
)

DEFAULT_USER_PROMPT = "Rewrite this as a rich, detailed prompt: a cat on a roof."

#: Shown in the dropdown when nothing was found, so the node still loads and the user gets
#: an actionable message at run time instead of a red box at import time.
_NO_MODEL = "<no .ninfer model found — see README>"


def _mib(n: float) -> str:
    return f"{n / 2 ** 20:.0f} MiB"


def _frames_to_jpeg(batch, max_side: int) -> list[bytes]:
    """Convert an IMAGE tensor (B,H,W,C float in 0..1) into one JPEG per frame."""
    count = int(batch.shape[0])
    return [tensor_to_jpeg(batch[i:i + 1], max_side=max_side) for i in range(count)]


def _effective_system_prompt(system_prompt: str, model_label: str) -> str:
    """Resolve the system prompt for a request.

    Explicit widget text always wins. Otherwise AUTO: only models whose name
    contains "i2i" (any case — covers pe_i2i_heretic, Qwen-Image-2.1-PE-I2I and
    any future i2i-named prompt enhancer) get I2I_SYSTEM_PROMPT; every other
    model runs with NO system prompt, so plain-text / video chats keep their own
    tone instead of inheriting the prompt-engineer persona.
    """
    explicit = (system_prompt or "").strip()
    if explicit:
        return explicit
    if "i2i" in (model_label or "").lower():
        return I2I_SYSTEM_PROMPT
    return ""


#: File-extension → MIME for the video payloads handed to the engine. Anything
#: unrecognized falls back to video/mp4 (the engine sniffs the container anyway).
_VIDEO_MIMES = {
    "mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime",
    "webm": "video/webm", "mkv": "video/x-matroska", "avi": "video/x-msvideo",
}


def _video_to_medias(video, max_side: int):
    """Turn a ComfyUI VIDEO input into engine media entries plus an info label.

    The engine decodes "video/*" payloads itself (its bundled FFmpeg: frame
    sampling at the model's configured fps, temporal patching, timestamps), so a
    file-backed VIDEO is handed over as raw bytes with its mime — the highest
    quality path. Only when the VIDEO object has no file behind it (e.g. it was
    built from tensors) do we fall back to exporting its frames as JPEGs, which
    the engine then treats as separate pictures.
    """
    try:
        source = video.get_stream_source()
    except Exception:
        source = None
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(source, timeout=60) as resp:
            data = resp.read()
        return [(data, "video/mp4")], "remote video (%d KiB)" % (len(data) // 1024)
    if isinstance(source, str) and os.path.exists(source):
        with open(source, "rb") as fh:
            data = fh.read()
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        mime = _VIDEO_MIMES.get(ext, "video/mp4")
        name = source.replace("\\", "/").rsplit("/", 1)[-1]
        return [(data, mime)], "%s (%d KiB, %s)" % (name, len(data) // 1024, mime)
    if hasattr(source, "read"):
        # io.BytesIO (or any readable buffer): in-memory video, mime unknowable.
        data = source.read()
        return [(data, "video/mp4")], "in-memory video (%d KiB)" % (len(data) // 1024)
    frames = None
    try:
        frames = getattr(video.get_components(), "images", None)
    except Exception:
        frames = None
    if frames is None:
        raise NinferError(
            "The connected VIDEO carries neither a readable file nor frame data.\n"
            "连接的视频既没有可读的文件路径，也没有帧数据。"
        )
    entries = [(b, "image/jpeg") for b in _frames_to_jpeg(frames, max_side)]
    return entries, "%d frame(s) exported from video object" % len(entries)


class NinferLocalLLM:
    """Single-turn local generation, with up to eight image batches for vision models.

    Image sockets appear progressively: wire one and the next shows up (frontend
    extension in web/; on frontends without socket hiding all eight simply show at
    once — the node works either way). Every frame is sent, in order — the engine
    numbers them Picture 1..N ahead of the text.

    System prompt: leave the widget empty for AUTO — only models whose name contains
    "i2i" get the prompt-engineer persona injected; all other models run with no
    system prompt. Text typed into the widget always overrides AUTO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        choices = model_choices() or [_NO_MODEL]
        # Remember what loaded successfully last time: a fresh node should default to the
        # model the user actually works with, not whichever artifact sorts first. A saved
        # workflow keeps its own stored value; this only steers newly created nodes.
        remembered = _state.last_model()
        default_model = remembered if remembered in choices else choices[0]
        return {
            "required": {
                # Populated by scanning ComfyUI/models/LLM and friends. Adding a model
                # needs a ComfyUI restart for the dropdown to refresh.
                "model": (choices, {"default": default_model}),
                # Leave empty for AUTO: models whose name contains "i2i" (any case)
                # get I2I_SYSTEM_PROMPT injected, every other model gets NO system
                # prompt at all. Anything typed here always wins over AUTO.
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
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
                # Progressive image sockets (8 total). "image"/"images" keep their
                # historic names so saved workflows keep working; image3..image8 are
                # new. All are batches; frames are sent in this listed order. The web/
                # extension hides every image socket after the first unconnected one
                # and reveals them one by one as sockets get wired.
                "image": ("IMAGE",),
                "images": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                # A video (from Load Video). The engine decodes it with its bundled
                # FFmpeg and reads it as a temporal frame sequence. Sent after images.
                "video": ("VIDEO",),
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
                 auto_recover=True, model_path_override="", dll_path="", image=None,
                 images=None, image3=None, image4=None, image5=None, image6=None,
                 image7=None, image8=None, video=None):

        requested = model_path_override.strip() or model
        model_path = resolve_model(requested)
        if model_path is None:
            raise NinferError(model_setup_hint(requested))
        system = _effective_system_prompt(system_prompt, requested)

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

        jpegs: list[bytes] = []
        for img in (image, images, image3, image4, image5, image6, image7, image8):
            if img is not None:
                jpegs.extend(_frames_to_jpeg(img, int(image_max_side)))
        medias: list[tuple] = [(b, "image/jpeg") for b in jpegs]
        video_info = "none"
        if video is not None:
            v_entries, video_info = _video_to_medias(video, int(image_max_side))
            medias.extend(v_entries)
        if medias and not has_vision:
            raise NinferError(
                "Image(s)/video are connected but the resident engine has no vision "
                "encoder.\n"
                "Set vision = true, then restart ComfyUI (or run the Free VRAM node) so a "
                "new engine gets created.\n"
                "连了图像/视频，但当前引擎没有 vision。把 vision 打开，然后重启 ComfyUI"
                "（或跑一次「释放显存」节点）让它重建。"
            )

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
        multi_note = ""
        try:
            if len(medias) >= 2 and not engine.supports_multi_image():
                # Old engine binaries only take one media payload per request. Downgrade
                # instead of failing — but say it loudly in the info output. A video
                # cannot survive this downgrade (the old binary would decode it as a
                # still image and fail), so it only ever applies to plain images.
                multi_note = (f"engine binary predates multi-image support; sent media 1 "
                              f"of {len(medias)} only")
                text = engine.generate(system, user_prompt,
                                       image=medias[0][0], image_mime=medias[0][1],
                                       req=request)
            elif len(medias) >= 2:
                text = engine.generate_media(system, user_prompt,
                                             medias, req=request)
            elif medias:
                # Single payload — images and videos alike (mime decides the pipeline).
                text = engine.generate(system, user_prompt,
                                       image=medias[0][0], image_mime=medias[0][1],
                                       req=request)
            else:
                text = engine.generate(system, user_prompt,
                                       image=None, req=request)
        except NinferError as error:
            # Some failures (an architecture mismatch, for instance) only surface on the
            # first kernel launch rather than at engine construction, so they need the same
            # translation as a creation failure would get.
            raise NinferError(guidance_for(str(error), cfg)) from None
        gen_seconds = time.perf_counter() - gen_started

        # Persist for next time: this model label and the binary that is actually running.
        # Both surface again as the new-node default and the engine= line in the info text.
        _state.save(model=model, engine=engine.dll_path)

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

        if medias:
            sizes = ", ".join("%d KiB" % (len(b) // 1024) for b, _ in medias)
            media_info = "%d media (%s)" % (len(medias), sizes)
        else:
            media_info = "none"

        if (system_prompt or "").strip():
            sys_mode = "explicit"
        elif "i2i" in requested.lower():
            sys_mode = "auto: i2i persona"
        else:
            sys_mode = "auto: none (not an i2i model)"

        info = (
            f"load {load_seconds:.2f}s | gen {gen_seconds:.2f}s | {len(text)} chars\n"
            f"model={model_path}\n"
            f"engine={engine.dll_path}  (dll_path widget was {'auto-resolved' if not dll_path.strip() else 'set manually'})\n"
            f"ctx={max_context} | vision={has_vision} | MTP={mtp_draft_tokens} | kv={kv_dtype} | "
            f"graph={'on' if use_cuda_graph else 'off'} | emb_host={'on' if embedding_host else 'off'}\n"
            f"thinking={'on' if enable_thinking else 'off'} | system={sys_mode} | "
            f"media={media_info} | video={video_info} | "
            f"{'released' if released else 'engine kept loaded'}\n"
            f"{vram}"
        )
        if multi_note:
            info += f"note: {multi_note}\n"
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

# Frontend extension (web/ninfer_dynamic_images.js): progressive disclosure of the
# image sockets — they appear one by one as the ones before them get wired.
WEB_DIRECTORY = "./web"

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
