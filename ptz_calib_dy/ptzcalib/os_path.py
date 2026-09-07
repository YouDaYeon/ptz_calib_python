"""Port of ``src/utils/os_path.{h,cc}``."""

from __future__ import annotations

import os
from typing import List, Tuple


def exists(path: str) -> bool:
    return os.path.exists(path)


def mkdir_ifnot_exist(directory: str) -> bool:
    if exists(directory):
        return True
    try:
        os.mkdir(directory, 0o755)
    except OSError:
        return False
    return True


def mkdirs(directory: str) -> bool:
    if not directory:
        return True
    try:
        os.makedirs(directory, 0o755, exist_ok=True)
    except OSError:
        return False
    return True


def listdir(directory: str) -> List[str]:
    """C++ ``Listdir``: full paths of the direct children (unsorted)."""
    if not exists(directory):
        return []
    return [os.path.join(directory, name) for name in os.listdir(directory)]


def basename(path: str) -> str:
    found = max(path.rfind("/"), path.rfind("\\"))
    if found < 0:
        return path
    return path[found + 1:]


def splitext(path: str) -> Tuple[str, str]:
    """C++ ``splitext``: returns ``(rootname, ext)`` where ``ext`` keeps the dot."""
    pend = len(path)
    i = pend - 1
    while i >= 0:
        c = path[i]
        if c in "/\\":
            return path, ""
        if c == ".":
            sep = i
            if sep == 0:
                return path, ""
            i -= 1
            while i >= 0 and path[i] == ".":
                i -= 1
            if i < 0:
                return path, ""
            return path[:sep], path[sep:pend]
        i -= 1
    return path, ""


def join_path(root_dir: str, filename: str) -> str:
    if root_dir.endswith("/") or root_dir.endswith("\\"):
        return root_dir + filename
    return root_dir + "/" + filename
