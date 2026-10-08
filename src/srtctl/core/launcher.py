# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's processes run: Slurm steps, or Docker containers.

The ``SweepOrchestrator`` asks a ``Launcher`` for everything that depends on where
it runs: start a process (``launch``), signal a named step (``signal_step``), list
running steps (``list_step_ids``), the job's nodes (``nodes``) and their addresses
(``node_ip``), the job id (``job_id``), and how a rendered job script is submitted
(``submit`` / ``start``). ``SlurmLauncher`` (``core/slurm.py``) is the default:
srun steps inside an sbatch allocation. ``DockerLauncher`` (``core/docker.py``)
runs each process as a ``docker run`` container on its node, over ssh when the
node is not this machine.

The launcher is picked by the ``launcher`` key of ``srtslurm.yaml``
(``get_launcher``). Nothing outside the launcher modules branches on that name.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

DEFAULT_LAUNCHER = "slurm"


def _get_cluster_bash_preamble() -> str | None:
    """The cluster-wide ``default_bash_preamble`` from ``srtslurm.yaml``, if set."""
    from srtctl.core.config import get_srtslurm_setting

    value = get_srtslurm_setting("default_bash_preamble")
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class LaunchSpec:
    """One process launch, independent of where it runs.

    Fields:
        command: Command to run
        nodes, ntasks: Node and task counts; ``nodelist`` names the nodes
        cpus_per_task: CPUs per task
        output: File that receives the task's stdout and stderr
        container_image, container_mounts: Container to run in (host path -> container path)
        env_to_pass_through: Environment variable names to pass through
        env_to_set: Environment variables to export inside the task before the command
        env_to_unset: Environment variable names to unset before the preamble and command
        bash_preamble: Bash commands to run before the main command
        srun_options: Additional srun options (Slurm only)
        srun_export_env: Env vars the container runtime sees at creation time
            (``srun --export=ALL,K=V``, ``docker run -e``); ENROOT_REMAP_ROOT and the
            like must be set here rather than in ``env_to_set``
        overlap: ``srun --overlap`` (Slurm only)
        use_bash_wrapper: Wrap the command in the bash wrapper (``build_task_command``)
        mpi, oversubscribe, cpu_bind: MPI launch options (TRT-LLM; Slurm only)
        het_group: Component of a Slurm heterogeneous job
        step_name: Names the step so it can be found and signalled later
            (``signal_step``); signalling the tracking process itself only aborts it
    """

    command: list[str]
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    nodelist: Sequence[str] | None = None
    output: str | None = None
    container_image: str | None = None
    container_mounts: dict[Path, Path] | None = None
    env_to_pass_through: list[str] | None = None
    env_to_set: dict[str, str] | None = None
    env_to_unset: list[str] | None = None
    bash_preamble: str | None = None
    srun_options: dict[str, str] | None = None
    srun_export_env: dict[str, str] | None = None
    overlap: bool = True
    use_bash_wrapper: bool = True
    mpi: str | None = None
    oversubscribe: bool = False
    cpu_bind: str | None = None
    het_group: int | None = None
    step_name: str | None = None


class Launcher(ABC):
    """How and where the orchestrator's processes run."""

    name: ClassVar[str]

    def build_task_command(self, spec: LaunchSpec) -> list[str]:
        """The command the task runs: the bash wrapper around ``spec.command``, or the command itself.

        The wrapper exports ``env_to_set``, runs the cluster ``default_bash_preamble``,
        unsets ``env_to_unset``, runs the per-call preamble, then ``exec``s the command
        so it replaces bash and receives SIGTERM directly: a ``bash -c`` parent would
        hold the signal until its child exited, so the child was only ever SIGKILLed
        at the cleanup timeout.
        """
        cluster_preamble = _get_cluster_bash_preamble()
        if not spec.use_bash_wrapper:
            if cluster_preamble:
                logger.warning(
                    "Cluster default_bash_preamble is set but this launch bypasses the bash wrapper "
                    "(use_bash_wrapper=False); preamble will not be applied. command=%s",
                    shlex.join(spec.command),
                )
            return list(spec.command)

        bash_parts = []
        if cluster_preamble:
            bash_parts.append(cluster_preamble)
        if spec.env_to_set:
            for name, value in spec.env_to_set.items():
                bash_parts.append(f"export {name}={shlex.quote(value)}")
        if spec.env_to_unset:
            for name in spec.env_to_unset:
                bash_parts.append(f"unset -- {shlex.quote(name)}")
        if spec.bash_preamble:
            bash_parts.append(spec.bash_preamble)
        bash_parts.append("exec " + shlex.join(spec.command))
        return ["bash", "-c", " && ".join(bash_parts)]

    @abstractmethod
    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        """Start ``spec`` and return the local process that tracks it."""

    @abstractmethod
    def list_step_ids(self, job_id: str | None = None) -> dict[str, str] | None:
        """``{step name: id}`` for this job's running named steps; None when it cannot be asked."""

    @abstractmethod
    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """Deliver ``sig`` to the task of the step named ``step_name``; True when delivered.

        False tells the caller to signal the tracking process directly.
        """

    @abstractmethod
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
        """The job's nodes, carved into head, infra, worker and pool roles."""

    @abstractmethod
    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        """The address other processes in the job use to reach ``hostname``."""

    @abstractmethod
    def job_id(self) -> str | None:
        """The running job's id, or None outside a job."""

    @abstractmethod
    def submit(self, script_path: Path) -> str:
        """Hand the rendered job script over and return the new job id."""

    def start(self, job_id: str, job_output_dir: Path) -> int | None:
        """Run a submitted job once its output directory is staged; None when the scheduler runs it."""
        return None

    @abstractmethod
    def shell_command(self, runtime: RuntimeContext, node: str) -> str:
        """A command a user can paste to get a shell next to the job's processes on ``node``."""

    @abstractmethod
    def status_hint(self, job_id: str) -> str:
        """A command that shows whether ``job_id`` is still running."""

    def validate(self, config: SrtConfig) -> list[str]:
        """Reasons ``config`` cannot run under this launcher; empty when it can."""
        return []


@cache
def launchers() -> dict[str, type[Launcher]]:
    """Every launcher by its ``srtslurm.yaml`` name."""
    from srtctl.core.docker import DockerLauncher
    from srtctl.core.slurm import SlurmLauncher

    return {cls.name: cls for cls in (SlurmLauncher, DockerLauncher)}


@cache
def _launcher(name: str) -> Launcher:
    return launchers()[name]()


def get_launcher() -> Launcher:
    """The launcher named by ``launcher`` in ``srtslurm.yaml`` (default ``slurm``)."""
    from srtctl.core.config import get_srtslurm_setting

    name = get_srtslurm_setting("launcher") or DEFAULT_LAUNCHER
    if name not in launchers():
        raise ValueError(f"srtslurm.yaml launcher: {name!r} is not one of {', '.join(sorted(launchers()))}")
    return _launcher(name)


def launch(spec: LaunchSpec) -> subprocess.Popen:
    """Start ``spec`` with the selected launcher; the one entry point for every process the job runs."""
    return get_launcher().launch(spec)
