# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the direct worker before exec and prove ownership of its Linux listener.

This script is also invoked directly with the container's Python, so it uses
only the standard library and no package-relative imports.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path


def process_record(path: Path) -> tuple[int, int]:
    """Return parent PID and start ticks without splitting the process comm field."""
    fields = path.read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def namespace_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def record_worker(path: Path, *, proc: Path = Path("/proc")) -> None:
    pid = os.getpid()
    _, start = process_record(proc / str(pid) / "stat")
    value = {
        "pid": pid,
        "start_ticks": start,
        "pid_namespace": namespace_identity(proc / "self/ns/pid"),
        "net_namespace": namespace_identity(proc / "self/ns/net"),
    }
    temporary = path.with_suffix(f".tmp-{pid}")
    with temporary.open("x") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def listening_inodes(port: int, *, proc: Path = Path("/proc")) -> set[str]:
    inodes = set()
    for table in ("tcp", "tcp6"):
        try:
            rows = (proc / "net" / table).read_text().splitlines()[1:]
        except FileNotFoundError:
            if table == "tcp6":
                continue
            raise
        for row in rows:
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def listener_owned(identity_path: Path, port: int, *, proc: Path = Path("/proc")) -> bool:
    """False means no listener yet; an unowned or unverifiable listener raises.

    PID/start time and namespace identity bind the worker across exec. All
    sockets on the direct port must belong to it or a current descendant.
    A same-model foreign service does not satisfy this check.
    """
    inodes = listening_inodes(port, proc=proc)
    if not identity_path.exists():
        if inodes:
            raise RuntimeError(f"Direct port {port} is occupied before the prepared worker starts")
        return False
    identity = json.loads(identity_path.read_text())
    for kind in ("pid", "net"):
        if tuple(identity[f"{kind}_namespace"]) != namespace_identity(proc / f"self/ns/{kind}"):
            raise RuntimeError(f"Prepared direct worker requires the host {kind} namespace")
    pid = identity["pid"]
    _, start = process_record(proc / str(pid) / "stat")
    if start != identity["start_ticks"]:
        raise RuntimeError("Prepared direct worker PID was reused")
    if not inodes:
        return False

    records = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            records[int(directory.name)] = process_record(directory / "stat")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {pid}
    while True:
        added = {child for child, (parent, born) in records.items() if parent in descendants and born >= start}
        if added <= descendants:
            break
        descendants.update(added)
    owned = set()
    owners = set()
    for child in descendants:
        try:
            if process_record(proc / str(child) / "stat") != records[child]:
                raise RuntimeError("Prepared worker descendant changed before socket inspection")
            child_inodes = set()
            for fd in (proc / str(child) / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    child_inodes.add(target[8:-1])
                    if inodes <= owned | child_inodes:
                        break
            if process_record(proc / str(child) / "stat") != records[child]:
                raise RuntimeError("Prepared worker descendant changed during socket inspection")
            owned.update(child_inodes)
            if child_inodes & inodes:
                owners.add(child)
            if inodes <= owned:
                break
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not inodes <= owned:
        raise RuntimeError(f"Direct port {port} has a listener outside the prepared worker process tree")
    # Confirm every credited socket owner still has the ancestry used to select it.
    checked = set()
    for child in owners:
        while child not in checked:
            if process_record(proc / str(child) / "stat") != records[child]:
                raise RuntimeError("Prepared worker listener ancestry changed during inspection")
            checked.add(child)
            if child == pid:
                break
            child = records[child][0]
    # Recheck the anchor after proc traversal to reject exit/PID reuse during inspection.
    if process_record(proc / str(pid) / "stat")[1] != start:
        raise RuntimeError("Prepared direct worker changed during listener inspection")
    return True


def wait_for_owned_listener(identity_path: Path, port: int, timeout: float, stop_event: threading.Event) -> bool:
    deadline = time.monotonic() + timeout
    while not stop_event.is_set():
        if listener_owned(identity_path, port):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        stop_event.wait(min(1.0, remaining))
    return False


def require_free_port(port: int) -> None:
    """Reject existing listeners before starting a model; ownership checks cover later races."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", port))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an explicit worker command is required")
    record_worker(args.identity)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
