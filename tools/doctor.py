#!/usr/bin/env python3
"""Check whether this machine can run the NInfer engine, and say what to do next.

    python tools/doctor.py

Reports the environment, the GPU and its compute capability, the engine binary (including
the CUDA architectures it was actually compiled for), and the models it can see — then
prints a verdict with concrete next steps. Nothing is loaded and no VRAM is allocated, so
it is safe to run while ComfyUI is busy.

The architecture check matters because of a measured fact: the published Windows engine
contains **only sm_120a cubins and no PTX** (``cuobjdump --list-elf`` / ``--list-ptx``), so
it runs exclusively on RTX 50-series cards. Everywhere else the driver reports
"no kernel image is available for execution on the device" and the engine has to be rebuilt.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Writing progress to a redirected console on Windows otherwise mangles non-ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - older interpreters
    pass

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(PACK))

import _deps  # noqa: E402  (needs the path above)
import _paths  # noqa: E402  (needs the path above)

GIB = 1 << 30
MIB = 1 << 20

#: What the published Windows binary contains. Update when a new arch is published.
PUBLISHED_ARCHES = {"sm_120a": "RTX 50-series (GB20x)"}

#: VRAM budget model, from measurements on an RTX 5070 Ti (15.92 GiB) with a 15.33 GiB
#: artifact. An artifact's *file* size is not its VRAM footprint, and ignoring that is how
#: you end up telling someone their working setup cannot possibly work.
#:   * weights land at ~0.93x the file size (measured: 15.33 GiB file -> ~14.27 GiB VRAM)
#:   * embedding_host (default on) moves the token-embedding table to pinned host RAM
#:   * the engine reserves ~0.6 GiB of runtime at max_context 4096
#:   * ComfyUI itself occupies ~1.3 GiB before you generate anything
WEIGHTS_TO_VRAM_RATIO = 0.93
EMBEDDING_HOST_SAVING_GIB = 0.78
RUNTIME_AT_4096_GIB = 0.6
COMFYUI_RESIDENCE_GIB = 1.3

ARCH_NAMES = {
    "sm_120": "RTX 50-series (Blackwell client)",
    "sm_121": "GB10 / DGX Spark class",
    "sm_100": "B100 / B200 (Blackwell datacenter)",
    "sm_90": "H100 / H200 (Hopper)",
    "sm_89": "RTX 40-series, L4, L40S (Ada)",
    "sm_86": "RTX 30-series, A40, A6000 (Ampere)",
    "sm_80": "A100, A30 (Ampere datacenter)",
    "sm_75": "RTX 20-series, T4 (Turing)",
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def head(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def run(args: list[str], timeout: int = 20) -> str:
    """Run a command, returning stdout+stderr, or '' when the tool is unavailable."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (done.stdout or "") + (done.stderr or "")
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return ""


def sec_to_arch(cc: str) -> str:
    """'12.0' -> 'sm_120'; '8.9' -> 'sm_89'."""
    major, _, minor = cc.partition(".")
    return f"sm_{major}{minor}"


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------

def report_environment() -> dict:
    head("1. Environment")
    info = {
        "python": sys.version.split()[0],
        "os": f"{sys.platform}",
        "frozen": getattr(sys, "frozen", False),
    }
    print(f"  Python      : {info['python']}  ({'embedded/frozen' if info['frozen'] else 'standard'})")
    print(f"  Platform    : {info['os']}")

    # A standalone run with ComfyUI's interpreter does not have the ComfyUI root on
    # sys.path, so this has to be arranged before folder_paths can be imported.
    comfy_root = _paths.ensure_comfyui_on_path()
    try:
        import folder_paths  # noqa: PLC0415

        models_dir = getattr(folder_paths, "models_dir", "?")
        print(f"  ComfyUI     : detected (root {comfy_root}, models {models_dir})")
        info["comfyui"] = str(models_dir)
    except Exception:  # noqa: BLE001
        print("  ComfyUI     : not importable from this interpreter")
        print("                (fine for a standalone check; run it with ComfyUI's python")
        print("                 to also see ComfyUI's model folders)")
        info["comfyui"] = None

    return info


def report_gpu() -> dict:
    head("2. GPU")
    gpu: dict = {"name": None, "cc": None, "arch": None, "vram_total": None,
                 "vram_free": None, "driver": None, "cuda": None}

    # torch first: it is the thing that actually has to work.
    try:
        import torch  # noqa: PLC0415

        print(f"  torch       : {torch.__version__}")
        print(f"  CUDA built  : {torch.version.cuda}")
        if torch.cuda.is_available():
            gpu["name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            gpu["cc"] = f"{major}.{minor}"
            gpu["arch"] = sec_to_arch(gpu["cc"])
            free, total = torch.cuda.mem_get_info()
            gpu["vram_total"], gpu["vram_free"] = total, free
        else:
            print("  CUDA        : NOT AVAILABLE from torch")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch       : not importable ({type(exc).__name__})")

    # nvidia-smi fills in the driver/VRAM when torch is missing.
    smi = shutil.which("nvidia-smi")
    if smi:
        query = run([smi, "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                     "--format=csv,noheader,nounits"])
        first = (query.strip().splitlines() or [""])[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 5:
            gpu["name"] = gpu["name"] or parts[0]
            gpu["driver"] = parts[1]
            gpu["vram_total"] = gpu["vram_total"] or int(parts[2]) * MIB
            gpu["vram_free"] = gpu["vram_free"] or int(parts[3]) * MIB
            if not gpu["cc"] and parts[4] not in ("", "N/A"):
                gpu["cc"] = parts[4]
                gpu["arch"] = sec_to_arch(parts[4])
    else:
        print("  nvidia-smi  : not on PATH")

    if gpu["name"]:
        print(f"  GPU         : {gpu['name']}")
    if gpu["arch"]:
        print(f"  Arch        : {gpu['arch']}  (compute capability {gpu['cc']}) "
              f"— {ARCH_NAMES.get(gpu['arch'], 'unrecognised')}")
    else:
        print("  Arch        : unknown (no GPU visible)")
    if gpu["driver"]:
        print(f"  Driver      : {gpu['driver']}")
    if gpu["vram_total"]:
        print(f"  VRAM        : {gpu['vram_free'] / GIB:.2f} GiB free / "
              f"{gpu['vram_total'] / GIB:.2f} GiB total")
    return gpu


def engine_arches(path: Path) -> tuple[set[str], str]:
    """Architectures present in the binary, as ``({'sm_120a'}, 'cuobjdump')``."""
    cuobjdump = shutil.which("cuobjdump")
    if not cuobjdump:
        for root in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",):
            base = Path(root)
            if base.is_dir():
                candidates = sorted(base.glob("v*/bin/cuobjdump.exe"), reverse=True)
                if candidates:
                    cuobjdump = str(candidates[0])
                    break

    if cuobjdump:
        listing = run([cuobjdump, "--list-elf", str(path)], timeout=180)
        found = {f"sm_{m}" for m in re.findall(r"sm_(\d+[a-z]?)\.cubin", listing)}
        if found:
            return found, "cuobjdump --list-elf"

    # Best effort without the CUDA toolkit: the fatbin keeps the cubin names as strings.
    found: set[str] = set()
    try:
        with open(path, "rb") as handle:
            overlap = b""
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                for match in re.finditer(rb"sm_(\d{2,3}[a-z]?)\.cubin", overlap + chunk):
                    found.add("sm_" + match.group(1).decode("ascii"))
                overlap = chunk[-64:]
    except OSError:
        pass
    return found, "raw scan (best effort)"


def report_engine() -> tuple[Path | None, set[str]]:
    head("3. Engine binary")
    path = _paths.resolve_dll()
    if path is None:
        print("  NOT FOUND")
        for line in _paths.dll_setup_hint().splitlines():
            print(f"    {line}")
        return None, set()

    engine = Path(path)
    size = engine.stat().st_size
    print(f"  Path        : {engine}")
    print(f"  Size        : {size / MIB:.1f} MiB")

    neighbours = sorted(p.name for p in engine.parent.iterdir()
                        if p.suffix.lower() in (".dll", ".so", ".dylib") and p != engine)
    note = "" if len(neighbours) >= 5 else "   <- expected ~10 for FFmpeg/curl"
    print(f"  Runtime libs: {len(neighbours)} beside it{note}")
    if len(neighbours) < 5:
        print("                -> re-run: python tools/fetch_engine.py")

    # Counting files is not enough: a real bundle shipped 14 of them and still could not
    # load, because avcodec-63.dll imports swresample-7.dll and that one was missing. Walk
    # the import tables instead, so the answer names the file to go and get.
    gaps = []
    if sys.platform == "win32":
        try:
            gaps = _deps.missing_dependencies(str(engine))
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            gaps = []
        if gaps:
            print(f"  MISSING     : {', '.join(sorted({name for _, name in gaps}))}")
            for owner, name in gaps:
                print(f"                  {name}  <- imported by {owner}")
            print("                -> copy the missing file(s) next to the engine, or")
            print("                   re-copy the whole ComfyUI-NInfer folder from the bundle")
        else:
            print("  Dependencies: all resolved (transitive import walk)")

    arches, method = engine_arches(engine)
    if arches:
        print(f"  Architectures: {', '.join(sorted(arches))}   (via {method})")
    else:
        print("  Architectures: could not determine")
        print("                 install the CUDA Toolkit for an exact answer, or just try it")
    return engine, arches


def vram_estimate_gib(artifact_gib: float) -> float:
    """Rough total VRAM a given artifact needs, ComfyUI included.

    Deliberately conservative but grounded in measurements — see WEIGHTS_TO_VRAM_RATIO.
    """
    return (artifact_gib * WEIGHTS_TO_VRAM_RATIO
            - EMBEDDING_HOST_SAVING_GIB
            + RUNTIME_AT_4096_GIB
            + COMFYUI_RESIDENCE_GIB)


def report_models(gpu: dict) -> list[Path]:
    head("4. Models")
    scanned = _paths.model_search_dirs()
    print("  Searched    :")
    for directory in scanned or ["(nothing — no model folders found)"]:
        print(f"    - {directory}")

    models = [Path(p) for p in _paths.discover_models()]
    if not models:
        print()
        print("  No .ninfer artifact found.")
        print("  Download one and drop it into ComfyUI/models/LLM, then restart ComfyUI.")
        print("  Pick by file size - the sources are not interchangeable:")
        for url, note in _paths.MODEL_SOURCES:
            print(f"    {url}   [{note}]")
        print(f"  Or point at your own folder: set {_paths.ENV_MODEL_DIRS}=D:\\models")
        return []

    total_gib = (gpu.get("vram_total") or 0) / GIB
    print()
    for model in models:
        size = model.stat().st_size / GIB
        print(f"  {model.name}")
        print(f"      {size:.2f} GiB on disk   {model}")
        if not total_gib:
            continue

        need = vram_estimate_gib(size)
        spare = total_gib - need
        if spare >= 1.5:
            verdict = f"comfortable (needs ~{need:.1f} of {total_gib:.1f} GiB)"
        elif spare >= 0:
            verdict = (f"tight — needs ~{need:.1f} of {total_gib:.1f} GiB, "
                       f"~{spare:.2f} GiB spare; see the VRAM budget table")
        else:
            verdict = (f"WILL NOT FIT — needs ~{need:.1f} GiB, this card has {total_gib:.1f} GiB; "
                       f"use a smaller quantization")
        print(f"      -> {verdict}")
    return models


def report_verdict(gpu: dict, engine: Path | None, arches: set[str], models: list[Path]) -> None:
    head("5. Verdict")
    problems: list[str] = []
    steps: list[str] = []

    arch = gpu.get("arch")
    if arch is None:
        problems.append("No CUDA device is visible, so nothing can run yet.")
        steps.append("Install/repair the NVIDIA driver, then re-run this script.")
    else:
        # An engine compiled for sm_120a only runs on an sm_120 device.
        compat = any(a.rstrip("a") == arch for a in arches) if arches else None
        if arches and not compat:
            problems.append(
                f"The engine is built for {', '.join(sorted(arches))}, "
                f"but this GPU is {arch}.")
            flag = _paths.ARCH_BUILD_FLAG.get(gpu.get("cc") or "", "?")
            steps.append("Rebuild the engine for your card:")
            steps.append(f"    pwsh -File tools\\build_engine.ps1 -CudaArch {flag} "
                         f"-VcpkgPrefix <vcpkg>\\installed\\x64-windows")
            steps.append("Or use a prebuilt engine published for your architecture, if one exists.")
        elif compat is None:
            steps.append("Architecture could not be verified — just try it, or install the "
                         "CUDA Toolkit so this check can be exact.")

    if engine is None:
        if not (arches and arch and not any(a.rstrip("a") == arch for a in arches)):
            problems.append("The engine binary is missing.")
            steps.append("Download it:  python tools/fetch_engine.py")
    else:
        neighbours = [p for p in engine.parent.iterdir()
                      if p.suffix.lower() in (".dll", ".so", ".dylib")]
        if len(neighbours) < 5:
            problems.append("The engine's runtime libraries are missing next to it.")
            steps.append("Re-download:  python tools/fetch_engine.py")
        elif sys.platform == "win32":
            # The count looked fine but a transitive import may still be absent. This is
            # the case that produces "Could not find module ... (or one of its
            # dependencies)" with no further clue at run time.
            try:
                gaps = _deps.missing_dependencies(str(engine))
            except Exception:  # noqa: BLE001
                gaps = []
            if gaps:
                names = ", ".join(sorted({name for _, name in gaps}))
                problems.append(f"Runtime libraries missing next to the engine: {names}")
                steps.append(f"Copy {names} into {engine.parent} (same folder as the "
                             f"engine), then restart ComfyUI.")
                steps.append("Or re-copy the whole ComfyUI-NInfer folder from the "
                             "bundle/release archive - it ships every dependency.")

    if not models:
        problems.append("No .ninfer model is visible.")
        steps.append("Put a .ninfer artifact in ComfyUI/models/LLM and restart ComfyUI.")

    if not problems:
        print("  Everything looks usable.")
        print()
        print("  Next: restart ComfyUI, then add")
        print("    NInfer Local LLM (.ninfer / Qwen3.8)")
        print("  from the NInfer category, and wire `info` into a Show Text node the first")
        print("  time — it reports load time and the engine's VRAM footprint.")
        print()
        print("  Recommended starting settings (16 GB card):")
        print("    max_context 4096 | embedding_host on | use_cuda_graph on")
        print("    enable_thinking OFF  <- this one is worth 5-10x on prompt work")
        print("    vision on if you attach an image, off otherwise")
        return

    for problem in problems:
        print(f"  [!] {problem}")
    print()
    print("  Do this:")
    for index, step in enumerate(steps, 1):
        print(f"    {index}. {step}")
    print()
    print("  More: README.md -> Troubleshooting, or README.zh-CN.md -> 常见报错")


def main() -> int:
    print("ComfyUI-NInfer — compatibility check")
    print(f"pack: {PACK}")

    environment = report_environment()
    gpu = report_gpu()
    engine, arches = report_engine()
    models = report_models(gpu)
    report_verdict(gpu, engine, arches, models)

    head("Summary")
    print(f"  python {environment['python']} | "
          f"{gpu.get('name') or 'no GPU'} ({gpu.get('arch') or '?'}) | "
          f"engine {'ok' if engine else 'MISSING'} | "
          f"{len(models)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
