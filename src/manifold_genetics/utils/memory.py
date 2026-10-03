"""How much memory this process can still use.

A node's free memory is not the answer on a cluster or in a container: a Slurm
job or a container is capped by its cgroup, often far below what the machine
has (a 62.5 GiB JupyterHub job on a 251 GB node). So this takes the tightest of
the cgroup v2 limits on the process's own cgroup and its ancestors, and the
kernel's MemAvailable.
"""

import os
from pathlib import Path
from typing import Optional


def _cgroup_headroom(proc_cgroup: Path, cgroup_root: Path) -> Optional[int]:
    """Smallest ``memory.max - memory.current`` from this cgroup up, or None."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None
    rel = next((line[3:] for line in lines if line.startswith("0::")), None)
    if rel is None:
        return None

    headroom = None
    node = cgroup_root / rel.lstrip("/")
    while True:
        try:
            limit = (node / "memory.max").read_text().strip()
            current = int((node / "memory.current").read_text().strip())
        except (OSError, ValueError):
            limit, current = "max", 0
        if limit != "max":
            room = int(limit) - current
            headroom = room if headroom is None else min(headroom, room)
        if node == cgroup_root or node.parent == node:
            break
        node = node.parent
    return headroom


def _meminfo_available(meminfo: Path) -> Optional[int]:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def available_memory_bytes(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    meminfo: Path = Path("/proc/meminfo"),
) -> Optional[int]:
    """Bytes this process can still allocate, or None if it cannot be told.

    Linux: min(cgroup headroom, MemAvailable). Elsewhere: total physical
    memory, an upper bound -- enough to catch a request that cannot fit at all.
    """
    candidates = [
        v
        for v in (_cgroup_headroom(proc_cgroup, cgroup_root), _meminfo_available(meminfo))
        if v is not None
    ]
    if candidates:
        return max(0, min(candidates))
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
