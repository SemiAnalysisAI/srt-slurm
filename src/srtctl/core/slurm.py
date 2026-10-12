# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SLURM utilities and the Slurm launcher.

- Environment: get_slurm_job_id, get_slurm_nodelist
- Network: get_hostname_ip, get_node_ips
- Commands: run_command
- Container utilities: get_container_mounts_str
- SlurmLauncher: processes as srun steps inside an sbatch allocation
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .ip_utils import get_node_ip
from .launch_plan import record_srun_command
from .launcher import Launcher, LaunchSpec

if TYPE_CHECKING:
    from .runtime import Nodes, RuntimeContext

logger = logging.getLogger(__name__)


# ============================================================================
# SLURM Environment
# ============================================================================


def get_slurm_job_id() -> str | None:
    """Get the current SLURM job ID from environment."""
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def get_slurm_nodelist() -> list[str]:
    """Get list of nodes from SLURM_NODELIST environment variable.

    Returns:
        List of node hostnames, or empty list if not in SLURM.
    """
    return _expand_nodelist(os.environ.get("SLURM_NODELIST", ""))


def _expand_nodelist(nodelist_raw: str) -> list[str]:
    """Expand a SLURM ranged nodelist via ``scontrol show hostnames``."""
    if not nodelist_raw:
        return []

    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist_raw],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try simple parsing for non-ranged formats
        return [nodelist_raw]


def get_slurm_het_nodelists() -> list[list[str]] | None:
    """Per-component nodelists for a SLURM heterogeneous job, else None.

    Returns one expanded nodelist per het component when ``SLURM_HET_SIZE`` is
    set to a value greater than 1. Returns None for non-het jobs so callers can
    fall back to ``get_slurm_nodelist()``.
    """
    het_size_raw = os.environ.get("SLURM_HET_SIZE", "")
    if not het_size_raw:
        return None
    try:
        het_size = int(het_size_raw)
    except ValueError:
        return None
    if het_size < 2:
        return None

    groups: list[list[str]] = []
    for i in range(het_size):
        nodelist_raw = os.environ.get(f"SLURM_JOB_NODELIST_HET_GROUP_{i}", "")
        groups.append(_expand_nodelist(nodelist_raw))
    return groups


# ============================================================================
# Network Resolution
# ============================================================================


def get_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
    """The address other processes in the job use to reach ``hostname``, from the selected launcher."""
    from .launcher import get_launcher

    return get_launcher().node_ip(hostname, network_interface)


def resolve_slurm_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
    """Resolve hostname to routable IP address.

    Uses multiple resolution strategies:
    1. If inside a SLURM job, use srun to get the real IP from the target node
    2. Fall back to socket.gethostbyname() (may return loopback on some systems)

    Args:
        hostname: Node hostname to resolve
        network_interface: Optional network interface to prefer

    Returns:
        IP address as string
    """
    # If we're inside a SLURM allocation, use srun-based resolution
    # This gets the actual routable IP from the target node
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id:
        ip = get_node_ip(hostname, slurm_job_id, network_interface)
        if ip:
            return ip
        logger.warning(
            "srun-based IP resolution failed for %s, falling back to socket resolution",
            hostname,
        )

    # Fallback to socket resolution
    try:
        ip = socket.gethostbyname(hostname)
        # Warn if we got a loopback address
        if ip.startswith("127."):
            logger.warning(
                "socket.gethostbyname returned loopback %s for %s - this may cause cross-node issues",
                ip,
                hostname,
            )
        return ip
    except socket.gaierror:
        # Return hostname as-is (may be IP already)
        return hostname


def get_node_ips(
    nodes: list[str],
    slurm_job_id: str | None = None,
    network_interface: str | None = None,
) -> dict[str, str]:
    """Get IP addresses for multiple SLURM nodes.

    Args:
        nodes: List of node hostnames
        slurm_job_id: SLURM job ID for srun context
        network_interface: Specific network interface to use

    Returns:
        Dict mapping node hostname to IP address
    """
    ips = {}
    for node in nodes:
        ip = get_node_ip(node, slurm_job_id, network_interface)
        if ip:
            ips[node] = ip
        else:
            logger.warning("Could not resolve IP for node %s", node)
    return ips


# ============================================================================
# Process Launching
# ============================================================================

# enroot env var that remaps the unprivileged user to root inside the container
# at container-creation time. Injected (via srun --export) only on launches that
# install dynamo, whose cold build needs apt-get/pip-to-system as root. Passing an
# env var (not the pyxis --container-remap-root flag) degrades gracefully: srun
# never parses it, so an unsupporting cluster no-ops instead of failing the step.
CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "yes"}


def run_command(
    command: str,
    background: bool = False,
    stdout=None,
    stderr=None,
) -> subprocess.Popen | int:
    """Run a shell command.

    Args:
        command: Command string to run
        background: If True, return Popen object; if False, wait and return exit code
        stdout: Optional stdout file handle
        stderr: Optional stderr file handle

    Returns:
        Popen object if background=True, exit code if background=False
    """
    logger.debug("Running command: %s", command)

    if background:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout or subprocess.DEVNULL,
            stderr=stderr or subprocess.DEVNULL,
        )
        return proc
    else:
        result = subprocess.run(command, shell=True, check=False)
        return result.returncode


# ============================================================================
# Container Utilities
# ============================================================================


def get_container_mounts_str(mounts: dict[Path, Path]) -> str:
    """Convert container mounts dict to comma-separated string.

    Args:
        mounts: Dict mapping host paths to container paths

    Returns:
        Comma-separated string for --container-mounts
    """
    return ",".join(f"{host}:{container}" for host, container in mounts.items())


class SlurmLauncher(Launcher):
    """Processes are Slurm steps (``srun``) inside an ``sbatch`` allocation."""

    name = "slurm"

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        srun_cmd = ["srun"]

        # Run in the same job context.
        slurm_job_id = get_slurm_job_id()
        if slurm_job_id:
            srun_cmd.extend(["--jobid", slurm_job_id])

        if spec.overlap:
            srun_cmd.append("--overlap")

        # MPI options (for TRTLLM)
        if spec.mpi:
            srun_cmd.extend(["--mpi", spec.mpi])
        if spec.oversubscribe:
            srun_cmd.append("--oversubscribe")
        if spec.cpu_bind:
            srun_cmd.append(f"--cpu-bind={spec.cpu_bind}")

        # Arbitrary layouts derive their node count from the repeated host list.
        if not spec.srun_options or spec.srun_options.get("distribution") != "arbitrary":
            srun_cmd.extend(["--nodes", str(spec.nodes)])
        srun_cmd.extend(["--ntasks", str(spec.ntasks)])

        if spec.cpus_per_task:
            srun_cmd.extend(["--cpus-per-task", str(spec.cpus_per_task)])

        if spec.nodelist:
            srun_cmd.extend(["--nodelist", ",".join(spec.nodelist)])

        # Route this srun to a specific component of a SLURM heterogeneous job.
        if spec.het_group is not None:
            srun_cmd.append(f"--het-group={spec.het_group}")

        if spec.output:
            srun_cmd.extend(["--output", spec.output])

        if spec.container_image:
            srun_cmd.extend(["--container-image", str(spec.container_image)])
            srun_cmd.append("--no-container-entrypoint")
            srun_cmd.append("--no-container-mount-home")
            if spec.container_mounts:
                mount_str = ",".join(f"{host}:{container}" for host, container in spec.container_mounts.items())
                srun_cmd.extend(["--container-mounts", mount_str])

        if spec.srun_options:
            for key, value in spec.srun_options.items():
                if value:
                    srun_cmd.append(f"--{key}={value}")
                else:
                    srun_cmd.append(f"--{key}")

        if spec.step_name:
            srun_cmd.append(f"--job-name={spec.step_name}")

        # Set env vars in the task environment so the container runtime (enroot/pyxis)
        # sees them at container-creation time. Prefix ALL to preserve srun's normal
        # full-environment propagation and only add these on top.
        if spec.srun_export_env:
            exports = ",".join(f"{k}={v}" for k, v in spec.srun_export_env.items())
            srun_cmd.append(f"--export=ALL,{exports}")

        srun_cmd.extend(self.build_task_command(spec))

        # Demoted to debug — every worker srun line is multi-KB once the
        # fingerprint heredoc is inlined (see core/fingerprint.generate_capture_script).
        logger.debug("srun command: %s", shlex.join(srun_cmd))

        record_srun_command(
            srun_cmd,
            label=Path(spec.output).stem if spec.output else Path(spec.command[0]).name,
            output=spec.output,
            nodelist=list(spec.nodelist) if spec.nodelist else None,
            het_group=spec.het_group,
            container_image=str(spec.container_image) if spec.container_image else None,
            env_to_set=spec.env_to_set,
            srun_export_env=spec.srun_export_env,
        )

        return subprocess.Popen(
            srun_cmd,
            stdout=subprocess.PIPE if not spec.output else None,
            stderr=subprocess.STDOUT if not spec.output else None,
            env=None,  # Inherit environment
        )

    def list_step_ids(self, job_id: str | None = None) -> dict[str, str] | None:
        job_id = job_id or get_slurm_job_id()
        if not job_id or shutil.which("squeue") is None:
            return None
        try:
            result = subprocess.run(
                ["squeue", "--steps", f"--jobs={job_id}", "--noheader", "--format=%i %j"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("squeue --steps failed: %s", exc)
            return None
        if result.returncode != 0:
            logger.warning("squeue --steps exited %d: %s", result.returncode, result.stderr.strip())
            return None
        steps: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                steps.setdefault(parts[1], parts[0])
        return steps

    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """``scancel --signal=<sig> [--full] <job>.<step>``.

        ``srun`` turns a SIGTERM aimed at itself into a step abort that SIGKILLs the
        task, so a process that must flush on SIGTERM has to be signalled through
        Slurm. With ``full=False`` only the tasks receive the signal.
        """
        if shutil.which("scancel") is None:
            return False  # not under Slurm (tests, the mock): the caller signals srun directly
        if step_ids is not None:
            step_id = step_ids.get(step_name)
        else:
            steps = self.list_step_ids()
            step_id = steps.get(step_name) if steps else None
        if step_id is None:
            logger.warning("No running step named %s found; falling back to signalling srun", step_name)
            return False
        try:
            result = subprocess.run(
                ["scancel", f"--signal={sig}", *(["--full"] if full else []), step_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("scancel --signal=%s %s failed: %s", sig, step_id, exc)
            return False
        if result.returncode != 0:
            logger.warning(
                "scancel --signal=%s %s exited %d: %s", sig, step_id, result.returncode, result.stderr.strip()
            )
            return False
        logger.info("Sent SIG%s to step %s (%s)", sig, step_id, step_name)
        return True

    def nodes(
        self,
        *,
        frontend_dedicated_node: bool = False,
        client_dedicated_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
        colocate_dedicated_nodes: bool = True,
        engine_nodes: int | None = None,
        pools: Sequence[tuple[str, int]] = (),
    ) -> Nodes:
        from srtctl.core.runtime import Nodes

        return Nodes.from_slurm(
            frontend_dedicated_node=frontend_dedicated_node,
            client_dedicated_node=client_dedicated_node,
            etcd_nats_dedicated_node=etcd_nats_dedicated_node,
            colocate_dedicated_nodes=colocate_dedicated_nodes,
            engine_nodes=engine_nodes,
            pools=pools,
        )

    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        return resolve_slurm_hostname_ip(hostname, network_interface)

    def job_id(self) -> str | None:
        return get_slurm_job_id()

    def submit(self, script_path: Path) -> str:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]

    def shell_command(self, runtime: RuntimeContext, node: str) -> str:
        container_args = f"--container-image={runtime.container_image}"
        mounts_str = ",".join(f"{src}:{dst}" for src, dst in runtime.container_mounts.items())
        if mounts_str:
            container_args += f" --container-mounts={mounts_str}"
        return f"srun {container_args} --jobid {runtime.job_id} -w {node} --overlap --pty bash"

    def status_hint(self, job_id: str) -> str:
        return f"squeue --job {job_id}"
