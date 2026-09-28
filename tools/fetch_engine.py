#!/usr/bin/env python3
"""Download the prebuilt NInfer engine into ``bin/`` so the nodes can run.

The engine (``ninfer_capi.dll`` plus its FFmpeg/curl runtime imports, ~250 MB) is published
as a GitHub Release asset. This script detects your GPU, picks the asset built for its
architecture, downloads it with a progress bar, and extracts the DLLs next to this pack.

    python tools/fetch_engine.py

Asset naming convention:

    ninfer-engine-<platform>-x64-<arch>.zip     e.g. ninfer-engine-win-x64-sm120a.zip
    ninfer-engine-<platform>-x64.zip            generic fallback when only one build exists

The architecture matters: a CUDA binary only runs on the compute capabilities it was
compiled for. The currently published Windows engine is sm_120a, which runs exclusively on
RTX 50-series cards.

Behind a slow or blocked GitHub connection, point it at a mirror instead:

    python tools/fetch_engine.py --url https://<mirror>/ninfer-engine-win-x64-sm120a.zip

or set ``NINFER_ENGINE_URL``. Only ``*.dll`` files (plus bundled license texts) are
extracted; anything else in the archive is ignored.

No third-party dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(PACK))

import _paths  # noqa: E402  (needs the path above)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

DEFAULT_REPO = "YukinoKaorisuna/ComfyUI-NInfer"
ASSET_PATTERN = "ninfer-engine"
ENV_URL = "NINFER_ENGINE_URL"
BIN_DIR = PACK / "bin"
USER_AGENT = "ComfyUI-NInfer-fetch_engine"

ARCH_IN_NAME = re.compile(r"sm(\d{2,3})a?", re.I)


def log(message: str) -> None:
    print(f"[fetch_engine] {message}", flush=True)


# --------------------------------------------------------------------------------------
# architecture selection
# --------------------------------------------------------------------------------------

def want_arch() -> str | None:
    """Local compute capability as ``sm_120``, or None when undetectable."""
    return _paths.device_arch()


def arch_base(token: str | None) -> str | None:
    """'sm_120a' / 'sm120' -> '120' so equivalent spellings compare equal."""
    if not token:
        return None
    match = re.match(r"sm_?(\d{2,3})", token, re.I)
    return match.group(1) if match else None


def platform_token() -> str:
    if sys.platform.startswith("win"):
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("darwin"):
        return "macos"
    return sys.platform


def select_asset(assets: list[dict], arch: str | None) -> tuple[dict | None, str]:
    """Pick the best engine archive for this machine. Returns ``(asset, why)``."""
    engines = [a for a in assets
               if ASSET_PATTERN in a.get("name", "") and a["name"].lower().endswith(".zip")]
    if not engines:
        return None, "the release has no engine archive"

    plat = platform_token()
    same_platform = [a for a in engines if plat in a["name"].lower()] or engines

    wanted = arch_base(arch)
    if wanted:
        for asset in same_platform:
            for token in ARCH_IN_NAME.findall(asset["name"]):
                if token == wanted:
                    return asset, f"built for {arch}"
        for asset in engines:
            for token in ARCH_IN_NAME.findall(asset["name"]):
                if token == wanted:
                    return asset, f"built for {arch} (different platform tag)"

    # A generic asset with no arch in its name is the last resort.
    generic = next((a for a in same_platform if not ARCH_IN_NAME.search(a["name"])), None)
    if generic:
        note = ("the asset does not state an architecture"
                + (f" — it may not run on {arch}" if arch else ""))
        return generic, note

    available = ", ".join(sorted(a["name"] for a in engines))
    return None, (f"no build for {arch or 'this machine'} — available: {available}")


# --------------------------------------------------------------------------------------
# download / extract
# --------------------------------------------------------------------------------------

def list_release(repo: str, tag: str | None) -> dict:
    api = (f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
           if tag else f"https://api.github.com/repos/{repo}/releases/latest")
    request = urllib.request.Request(api, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    log(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if total:
                    print(f"\r[fetch_engine]   {done / 2**20:7.1f} / {total / 2**20:.1f} MiB "
                          f"({done * 100 // total:3d}%)", end="", flush=True)
                else:
                    print(f"\r[fetch_engine]   {done / 2**20:7.1f} MiB", end="", flush=True)
    print()


def extract(archive: Path) -> list[str]:
    """Extract DLLs (and bundled notices) into ``bin/``, flattened."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if not base:
                continue
            lowered = base.lower()
            rel = info.filename.replace("\\", "/").lower()
            # Keep code plus notices; drop sources, symbols and documentation.
            keep = (lowered.endswith((".dll", ".so", ".dylib"))
                    or lowered.startswith("license")
                    or lowered.startswith("notice")
                    or "third_party/" in rel)
            if not keep:
                continue
            target = BIN_DIR / (rel if "third_party/" in rel else base)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, open(target, "wb") as sink:
                sink.write(source.read())
            written.append(base)
    return sorted(written)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"GitHub repo holding the release (default: {DEFAULT_REPO})")
    parser.add_argument("--tag", default=None, help="release tag (default: latest)")
    parser.add_argument("--url", default=None,
                        help=f"direct download URL, bypassing the GitHub API "
                             f"(or set {ENV_URL})")
    parser.add_argument("--arch", default=None,
                        help="override the detected architecture, e.g. sm_120a or sm_89")
    parser.add_argument("--list", action="store_true",
                        help="list the engine archives in the release and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if an engine is already installed")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the downloaded .zip next to the pack")
    args = parser.parse_args(argv)

    arch = args.arch or want_arch()
    gpu = _paths.device_name()
    if gpu:
        log(f"GPU: {gpu} ({arch or 'compute capability unknown'})")
    else:
        log("GPU: not visible from this interpreter; relying on --arch or the generic asset")

    if not args.list and not args.force:
        installed = _paths.resolve_dll()
        if installed is not None:
            log(f"engine already installed: {installed}")
            log("nothing to do. Use --force to re-download.")
            return 0

    url = args.url or os.environ.get(ENV_URL)

    if url is None:
        try:
            release = list_release(args.repo, args.tag)
        except urllib.error.HTTPError as exc:
            log(f"GitHub API returned HTTP {exc.code} ({exc.reason}).")
            if exc.code == 404:
                log(f"{args.repo} has no published release, or the repo name is wrong.")
            log(f"Pass --url <mirror link>, or set {ENV_URL}.")
            return 1
        except urllib.error.URLError as exc:
            log(f"cannot reach GitHub: {exc.reason}")
            log(f"Pass --url <mirror link>, or set {ENV_URL}.")
            return 1

        assets = release.get("assets") or []
        if args.list:
            log(f"release {release.get('tag_name')} — engine archives:")
            for asset in assets:
                mark = "*" if ASSET_PATTERN in asset.get("name", "") else " "
                log(f"  {mark} {asset.get('name')}  ({asset.get('size', 0) / 2**20:.1f} MiB)")
            return 0

        asset, why = select_asset(assets, arch)
        if asset is None:
            log(f"no usable engine archive: {why}")
            log(f"--list shows what is available; --url bypasses this check.")
            return 1
        log(f"chose {asset['name']} ({why})")
        if "may not run" in why or "does not state" in why:
            log("WARNING: this may be built for a different GPU. If it fails with")
            log("         'no kernel image is available', rebuild for your card:")
            log(f"           pwsh -File tools/build_engine.ps1 -CudaArch "
                f"{_paths.ARCH_BUILD_FLAG.get(_paths_cc(), '???')}")
        url = asset["browser_download_url"]

    archive = PACK / "_engine_download.zip"
    try:
        download(url, archive)
        written = extract(archive)
    finally:
        if not args.keep_archive and archive.is_file():
            archive.unlink()

    libs = [name for name in written if name.lower().endswith((".dll", ".so", ".dylib"))]
    print()
    log(f"extracted {len(libs)} library file(s) into {BIN_DIR}")
    for name in sorted(libs, key=lambda n: -(BIN_DIR / n).stat().st_size):
        log(f"  {name:28} {(BIN_DIR / name).stat().st_size / 2**20:8.1f} MiB")

    engine = _paths.resolve_dll()
    if engine is None:
        log("WARNING: no engine file was found after extracting — the archive did not")
        log("         contain ninfer_capi.dll/.so. Nothing has been installed.")
        return 2

    log(f"installed: {engine}")
    log("Next: restart ComfyUI. Verify with:  python tools/doctor.py")
    return 0


def _paths_cc() -> str:
    """Compute capability as 'x.y', for printing a build hint."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:  # noqa: BLE001
        pass
    return ""


if __name__ == "__main__":
    sys.exit(main())
