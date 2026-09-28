"""Portable path resolution: where the engine DLL lives, where the models live.

Deliberately free of ComfyUI imports at module scope (they are attempted lazily) so this
module can be unit-tested, and so importing the node pack never fails on a machine that
has neither the engine nor a model installed.
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Where the engine binary is expected. Populated by ``tools/fetch_engine.py`` or by a
#: local build; ``bin/*.dll`` is git-ignored and excluded from the registry archive.
BIN_DIR = os.path.join(PACKAGE_DIR, "bin")

#: Accepted engine file names. Only Windows has a prebuilt binary today, but the C ABI is
#: portable (plain ``extern "C"``), so a Linux/macOS build drops in the same way.
ENGINE_NAMES = ("ninfer_capi.dll", "ninfer_capi.so", "ninfer_capi.dylib")
ENGINE_STEMS = ("ninfer_capi",)

#: Optional overrides, useful when the engine lives outside the node folder.
ENV_DLL = "NINFER_CAPI_DLL"
ENV_MODEL_DIRS = "NINFER_MODEL_DIRS"

#: Sub-folders of ComfyUI's ``models/`` that are scanned for ``.ninfer`` artifacts.
MODEL_SUBDIRS = ("LLM", "llm", "NInfer", "ninfer")

#: Where to send someone who has no model yet. Two families, and the file size decides which is
#: usable — this pack targets 16 GB cards, so the 15.33 GiB release profile comes first. A
#: same-named 19.03 GiB file exists in the other one; they are not interchangeable.
MODEL_SOURCES = (
    ("https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080", "15.33 GiB - fits 16 GB"),
    ("https://huggingface.co/neroued", "16-23 GiB - 24 GB and up"),
)

#: Compute capability -> the value to pass to ``-DCMAKE_CUDA_ARCHITECTURES``.
#: Confirmed empirically: the published Windows binary contains sm_120a cubins only, so it
#: cannot run anywhere else (and there is no PTX to JIT from).
ARCH_BUILD_FLAG = {
    "12.0": "120a",
    "12.1": "121a",
    "9.0": "90a",
    "8.9": "89",
    "8.6": "86",
    "8.0": "80",
    "7.5": "75",
}


def device_arch() -> str | None:
    """Return the local GPU's compute capability as ``sm_XXX`` (e.g. ``sm_120``)."""
    capability = device_capability()
    if capability is None:
        return None
    major, _, minor = capability.partition(".")
    return f"sm_{major}{minor}"


def device_capability() -> str | None:
    """Return ``'12.0'`` for an RTX 50-series card, or None without CUDA.

    Kept separate from :func:`device_arch` because the dotted form is the one ARCH_BUILD_FLAG
    is keyed by — deriving it by splitting the ``sm_120`` string back apart is ambiguous
    (is that 12.0 or 1.20?).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return None


def device_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def build_arch_hint() -> str:
    """What to tell someone whose GPU does not match the prebuilt engine."""
    arch = device_arch()
    if arch is None:
        return (
            "No CUDA device visible from this Python.\n"
            "Check that an NVIDIA GPU is present and the driver is installed."
        )

    capability = device_capability() or ""
    flag = ARCH_BUILD_FLAG.get(capability) or arch.removeprefix("sm_")
    return (
        f"Your GPU: {device_name() or '?'} ({arch}).\n"
        f"本机显卡：{device_name() or '?'}（{arch}）。\n"
        "\n"
        "The published engine is compiled for sm_120a only, which runs **exclusively** on\n"
        "RTX 50-series (GB20x) cards. It also contains no PTX, so there is no JIT fallback.\n"
        "已发布的引擎只编译了 sm_120a，只能在 RTX 50 系上执行；且没有 PTX，没有 JIT 兜底。\n"
        "\n"
        "Two options:\n"
        "  1. Build it for your architecture (recommended):\n"
        f"       pwsh -File tools\\build_engine.ps1 -CudaArch {flag} -VcpkgPrefix <vcpkg>\\installed\\x64-windows\n"
        "     On Linux, build the upstream project directly and copy ninfer_capi.so into bin/.\n"
        "  2. Wait for a prebuilt binary for your architecture to be published.\n"
        "\n"
        "Check what a downloaded engine actually contains with:\n"
        "  python tools/doctor.py"
    )



# --------------------------------------------------------------------------------------
# Engine binary
# --------------------------------------------------------------------------------------

def candidate_dll_paths(explicit: str | None = None):
    """Yield engine paths to try, most specific first."""
    if explicit:
        yield explicit

    from_env = os.environ.get(ENV_DLL)
    if from_env:
        yield from_env

    for name in ENGINE_NAMES:
        yield os.path.join(BIN_DIR, name)

    # Tolerate a renamed or architecture-suffixed build (ninfer_capi-120a.dll) in bin/.
    if os.path.isdir(BIN_DIR):
        for name in sorted(os.listdir(BIN_DIR)):
            lowered = name.lower()
            if (lowered.startswith(ENGINE_STEMS)
                    and lowered.endswith((".dll", ".so", ".dylib"))):
                yield os.path.join(BIN_DIR, name)


def resolve_dll(explicit: str | None = None) -> str | None:
    """Return the first existing engine path, or None when nothing was found."""
    for path in candidate_dll_paths(explicit):
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def dll_setup_hint() -> str:
    """What to tell the user when the engine binary is missing."""
    looked = "\n".join(
        [f"  - dll_path widget (empty = auto)",
         f"  - ${ENV_DLL}"]
        + [f"  - {os.path.join(BIN_DIR, name)}" for name in ENGINE_NAMES]
    )
    return (
        "Engine not found / 未找到引擎。\n"
        f"Looked in / 查找位置:\n{looked}\n"
        "\n"
        "Fix it with one of:\n"
        "  1. Download the prebuilt engine (easiest):\n"
        "       python tools/fetch_engine.py\n"
        "  2. Check whether your GPU is supported at all:\n"
        "       python tools/doctor.py\n"
        "  3. Build it yourself for your architecture:\n"
        "       pwsh -File tools\\build_engine.ps1 -CudaArch <your arch>\n"
        f"  4. Point at an existing build:\n"
        f"       set {ENV_DLL}=D:\\path\\to\\ninfer_capi.dll\n"
        "See README.md for details."
    )


# --------------------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------------------

_COMFY_ROOT_PROBED = False


def ensure_comfyui_on_path() -> str | None:
    """Make the ComfyUI root importable when the pack sits in ``custom_nodes/``.

    Inside a running ComfyUI this is already the case. It matters for the standalone tools
    (``doctor.py``, ``fetch_engine.py``): without it they run with ComfyUI's interpreter but
    still cannot import ``folder_paths``, so they would miss ComfyUI's model folders
    entirely and report "no models found" for a perfectly normal installation.

    Returns the detected ComfyUI root, or None.
    """
    global _COMFY_ROOT_PROBED
    if not _COMFY_ROOT_PROBED:
        _COMFY_ROOT_PROBED = True
        # .../ComfyUI/custom_nodes/<pack>  ->  .../ComfyUI
        root = os.path.dirname(os.path.dirname(PACKAGE_DIR))
        if os.path.isfile(os.path.join(root, "folder_paths.py")) and root not in sys.path:
            sys.path.insert(0, root)

    root = os.path.dirname(os.path.dirname(PACKAGE_DIR))
    return root if os.path.isfile(os.path.join(root, "folder_paths.py")) else None


def model_search_dirs() -> list[str]:
    """Directories scanned for ``.ninfer`` artifacts, in priority order, existing only."""
    dirs: list[str] = []

    ensure_comfyui_on_path()

    # ComfyUI's model tree. Imported lazily: this module must work outside ComfyUI too.
    try:
        import folder_paths  # noqa: PLC0415

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            dirs.extend(os.path.join(models_dir, sub) for sub in MODEL_SUBDIRS)
        try:
            dirs.extend(folder_paths.get_folder_paths("llm"))
        except Exception:  # noqa: BLE001 - the "llm" folder type is not always registered
            pass
    except Exception:  # noqa: BLE001 - running standalone
        pass

    # A model dropped next to the node itself.
    dirs.append(os.path.join(PACKAGE_DIR, "models"))

    for extra in (os.environ.get(ENV_MODEL_DIRS) or "").split(os.pathsep):
        if extra.strip():
            dirs.append(extra.strip())

    seen: set[str] = set()
    out: list[str] = []
    for directory in dirs:
        key = os.path.normcase(os.path.abspath(directory))
        if key in seen or not os.path.isdir(directory):
            continue
        seen.add(key)
        out.append(os.path.abspath(directory))
    return out


def replace_separators(path: str) -> str:
    """Show forward slashes in the UI regardless of platform."""
    return path.replace("\\", "/")


def discover_models() -> list[str]:
    """Absolute paths of every ``.ninfer`` artifact we can find, sorted and de-duped."""
    found: dict[str, str] = {}
    for root in model_search_dirs():
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".ninfer"):
                continue
            absolute = os.path.abspath(entry.path)
            found[os.path.normcase(absolute)] = absolute
    return sorted(found.values())


def label_models(models: list[str]) -> list[str]:
    """Dropdown labels for ``models``: bare file names, de-duplicated when needed.

    A bare name is what a user wants to see, but two models can share a name in different
    folders, so collisions fall back to a path relative to the nearest scanned root.
    """
    by_name: dict[str, list[str]] = {}
    for path in models:
        by_name.setdefault(os.path.basename(path).lower(), []).append(path)

    roots = model_search_dirs()
    labels: list[str] = []
    for path in models:
        if len(by_name[os.path.basename(path).lower()]) == 1:
            labels.append(os.path.basename(path))
            continue
        fallback: str | None = None
        for root in roots:
            try:
                relative = os.path.relpath(path, root)
            except ValueError:
                continue
            if not relative.startswith(".."):
                fallback = replace_separators(relative)
                break
        labels.append(fallback or path)
    return labels


def model_choices() -> list[str]:
    """Labels for the model dropdown widget."""
    return label_models(discover_models())


#: Discovery touches the filesystem, and ComfyUI asks for INPUT_TYPES on every
#: /object_info refresh and every prompt validation. A couple of seconds of caching keeps
#: that cheap while still picking up a freshly downloaded model without a restart.
_CACHE_TTL_SECONDS = 2.0
_cache: tuple[float, list[str], dict[str, str]] | None = None


def _discovery() -> tuple[list[str], dict[str, str]]:
    """Return ``(labels, value -> absolute path)``, memoised for a short while."""
    global _cache

    import time

    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1], _cache[2]

    models = discover_models()
    labels = label_models(models)

    mapping: dict[str, str] = {}
    for label, path in zip(labels, models):
        mapping[label] = path
        mapping[replace_separators(label)] = path
    for path in models:
        # Bare name and raw path both resolve, so a workflow saved by hand (or a value
        # typed manually) keeps working.
        mapping.setdefault(os.path.basename(path), path)
        mapping.setdefault(path, path)
        mapping.setdefault(replace_separators(path), path)

    _cache = (now, labels, mapping)
    return labels, mapping


def resolve_model(value: str | None) -> str | None:
    """Resolve a widget value to an absolute model path, or None when unresolvable."""
    if not value:
        return None
    if os.path.isfile(value):
        return os.path.abspath(value)
    return _discovery()[1].get(value) or _discovery()[1].get(replace_separators(value))


def model_setup_hint(value: str | None) -> str:
    scanned = model_search_dirs()
    listing = "\n".join(f"  - {directory}" for directory in scanned) or "  (none)"
    return (
        f"Model not found / 未找到模型: {value!r}\n"
        f"Scanned / 已扫描:\n{listing}\n"
        "\n"
        "Download a .ninfer artifact and drop it into one of the folders above\n"
        "(ComfyUI/models/LLM is the usual place), then restart ComfyUI so the\n"
        "dropdown picks it up.\n"
        "下载 .ninfer 模型放到上面的目录（通常 ComfyUI/models/LLM），重启 ComfyUI 后\n"
        "下拉列表就会出现。\n"
        "Model sources (pick by size - the files are not interchangeable):\n"
        + "".join(f"  {url}   [{note}]\n" for url, note in MODEL_SOURCES)
        + "See README -> Models.\n"
        f"Or set {ENV_MODEL_DIRS}=D:\\models to add your own folder."
    )
