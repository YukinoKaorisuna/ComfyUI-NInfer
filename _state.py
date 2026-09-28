"""Tiny persistence for the node pack: remember what worked last time.

Two things are worth remembering across ComfyUI restarts:

* the last model that generated successfully — so a freshly dragged-in node defaults to it
  instead of the alphabetically-first artifact in the scan list;
* the last engine binary that actually loaded — displayed in the node's info output as proof
  that the automatic ``dll_path`` resolution picked the right file.

The state file lives beside the pack (``_state.json``), is git-ignored, and every access is
failure-tolerant: a missing, corrupt or read-only file degrades to "no memory", never to an
error. The node must work even when the pack directory is not writable.
"""

from __future__ import annotations

import json
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(PACKAGE_DIR, "_state.json")


def load() -> dict:
    """The whole state dict, or ``{}`` when there is nothing usable there.

    Always a *fresh* dict: ``save()`` mutates what it gets, and returning a shared
    singleton would leak that mutation into every later "no memory" answer.
    """
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - missing/corrupt/locked file: no memory is fine
        return {}


def save(*, model: str | None = None, engine: str | None = None) -> None:
    """Merge the given fields into the state file. Best-effort, never raises."""
    state = dict(load())  # defensive copy: never mutate whatever load() returned
    if model is not None:
        state["last_model"] = model
    if engine is not None:
        state["last_engine"] = engine
    if not state:
        return
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001 - read-only pack dir: memory silently degrades
        pass


def last_model() -> str | None:
    value = load().get("last_model")
    return value if isinstance(value, str) and value else None


def last_engine() -> str | None:
    value = load().get("last_engine")
    return value if isinstance(value, str) and value else None
