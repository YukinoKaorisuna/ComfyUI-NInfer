"""Read a PE file's import table and walk it transitively.

Windows reports a missing DLL as ``ERROR_MOD_NOT_FOUND`` (126), which Python surfaces as
``FileNotFoundError: Could not find module 'X' (or one of its dependencies)``. That message
never says *which* dependency, and the one that is missing is almost never the file named
in it — a real case:

    ninfer_capi.dll exists, all 14 of its usual neighbours exist, and the load still failed
    because avcodec-63.dll imports swresample-7.dll, which nobody shipped.

So on a load failure we parse the import tables ourselves and name the gap. Only the PE
headers are read (a few KiB), never the whole 300 MB binary.

No third-party dependencies: this has to run inside someone else's ComfyUI.
"""

from __future__ import annotations

import os
import struct

#: Where Windows looks, in order, after the loading DLL's own directory.
_SYSTEM_DIRS = None
_DELAY_LOAD_MARKER = 0


def _system_dirs() -> list[str]:
    global _SYSTEM_DIRS
    if _SYSTEM_DIRS is None:
        root = os.environ.get("SystemRoot") or r"C:\Windows"
        _SYSTEM_DIRS = [os.path.join(root, "System32"), os.path.join(root, "SysWOW64")]
    return _SYSTEM_DIRS


def pe_imports(path: str) -> list[str]:
    """Return the DLL names ``path`` imports, or [] when it is not a PE file.

    Delay-loaded imports are included: they are declared in the same table, and while they
    are not resolved at load time, listing one as missing is still the useful answer when
    the failure happens later.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(8192)
            if header[:2] != b"MZ":
                return []
            e_lfanew = struct.unpack_from("<I", header, 0x3C)[0]
            if header[e_lfanew:e_lfanew + 4] != b"PE\0\0":
                return []

            coff = e_lfanew + 4
            _machine, nsec, _ts, _psym, _nsym, opt_size, _chars = struct.unpack_from(
                "<HHIIIHH", header, coff)
            opt = coff + 20
            pe32plus = struct.unpack_from("<H", header, opt)[0] == 0x20B
            # Data directory entry 1 is the import table; its offset from the optional
            # header start differs between PE32 (0x60) and PE32+ (0x70).
            import_rva, import_size = struct.unpack_from(
                "<II", header, opt + (0x70 if pe32plus else 0x60) + 8)
            if not import_rva:
                return []

            sections = []
            for index in range(nsec):
                base = opt + opt_size + index * 40
                if base + 40 > len(header):
                    break
                vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", header, base + 8)
                sections.append((vaddr, max(vsize, rsize), raddr))

            def rva_to_offset(rva: int) -> int | None:
                for vaddr, span, raddr in sections:
                    if vaddr <= rva < vaddr + span:
                        return raddr + (rva - vaddr)
                return None

            offset = rva_to_offset(import_rva)
            if offset is None:
                return []

            handle.seek(offset)
            table = handle.read(import_size if import_size else 20 * 64)

            names: list[str] = []
            for index in range(len(table) // 20):
                _lookup, _stamp, _chain, name_rva, thunk = struct.unpack_from(
                    "<IIIII", table, index * 20)
                if not _lookup and not name_rva and not thunk:
                    break
                name_offset = rva_to_offset(name_rva)
                if name_offset is None:
                    continue
                handle.seek(name_offset)
                raw = handle.read(256)
                end = raw.find(b"\0")
                if end <= 0:
                    continue
                names.append(raw[:end].decode("ascii", "replace"))
            return names
    except (OSError, struct.error, ValueError):
        # Diagnostics must never raise: a truncated or packed binary just means "unknown".
        return []


def _resolve(name: str, own_dir: str, extra_dirs) -> str | None:
    for directory in [own_dir, *extra_dirs, *_system_dirs(),
                      *[d for d in os.environ.get("PATH", "").split(os.pathsep) if d]]:
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def missing_dependencies(root: str, extra_dirs=()) -> list[tuple[str, str]]:
    """Walk ``root``'s imports transitively; return ``[(importer, missing_name), ...]``.

    Mirrors how Windows resolves: the loading DLL's own directory first, then the usual
    system locations. Anything found outside ``own_dir`` is treated as satisfied by the OS
    and not descended into — crawling System32 would report thousands of names and none of
    them are actionable.
    """
    root = os.path.abspath(root)
    own_dir = os.path.dirname(root)
    if not os.path.isfile(root):
        return [(os.path.basename(root), "<the file itself is missing>")]

    gaps: list[tuple[str, str]] = []
    seen: set[str] = set()
    queue: list[str] = [root]

    while queue:
        current = queue.pop(0)
        key = os.path.normcase(current)
        if key in seen:
            continue
        seen.add(key)

        for name in pe_imports(current):
            resolved = _resolve(name, own_dir, extra_dirs)
            if resolved is None:
                gaps.append((os.path.basename(current), name))
                continue
            # Only follow siblings: a system DLL's own imports are the OS's problem.
            if (os.path.normcase(os.path.dirname(resolved)) == os.path.normcase(own_dir)
                    and os.path.normcase(resolved) not in seen):
                queue.append(resolved)

    return gaps
