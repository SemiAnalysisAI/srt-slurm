# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``launcher: docker``: every process is a ``docker run`` container on its node, over ssh off this machine."""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.ip_utils import SCRIPTS_DIR, get_local_ip
from srtctl.core.launcher import Launcher, LaunchSpec

if TYPE_CHECKING:
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]*
_CONTAINER_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]")
JOB_ID_ENV = "SRTCTL_JOB_ID"
JOB_LABEL = "srtctl.job"
STEP_LABEL = "srtctl.step"
# Host env forwarded into every container: pyxis hands a Slurm container the whole
# submit environment, docker hands it nothing. ``-e NAME`` takes the value from the
# environment of the docker client, so a container on another node gets that node's
# login environment.
_PASSTHROUGH_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY")
_SSH = ("ssh", "-n", "-o", "BatchMode=yes")


@dataclass
class _DockerStep:
    popen: subprocess.Popen
    hosts: list[str]
    container: str | None  # docker container name on every host; None for a host process


def _this_host() -> str:
    return socket.gethostname()


def _is_this_host(host: str) -> bool:
    name = _this_host()
    return host in (name, name.split(".")[0], "localhost")


def _on(host: str, cmd: list[str], *, tty: bool = False) -> list[str]:
    """``cmd`` as run on ``host``: as is here, through ssh elsewhere.

    ``tty`` gives the remote command a terminal, so it gets SIGHUP when its ssh client dies.
    """
    if _is_this_host(host):
        return cmd
    return [*_SSH, *(["-tt"] if tty else []), host, shlex.join(cmd)]


class DockerLauncher(Launcher):
    """Each process is a ``docker run`` container on its node; ssh reaches the nodes that are not this machine.

    The job's nodes are ``docker_hosts`` in ``srtslurm.yaml`` (default: this machine),
    carved into roles exactly like a Slurm nodelist; the orchestrator runs where
    ``srtctl apply`` ran. A launch runs one task on each node of its ``nodelist``, as
    srun does. Containers share the host network and IPC namespace and see every
    GPU; each worker's ``CUDA_VISIBLE_DEVICES`` (set inside the container by the bash
    wrapper) selects its GPUs, as inside a Slurm step. A launch without an image runs
    on the node's host, like a container-less srun. Mounted paths must exist on every
    node (a shared filesystem), as under Slurm.
    """

    name = "docker"

    def __init__(self) -> None:
        self._steps: dict[str, _DockerStep] = {}
        self._lock = threading.Lock()
        self._counter = 0

    @staticmethod
    def hosts() -> list[str]:
        from srtctl.core.config import get_srtslurm_setting

        return list(get_srtslurm_setting("docker_hosts") or [_this_host()])

    # -- launching -----------------------------------------------------------

    def _job_label(self) -> str:
        return self.job_id() or "docker"

    def _container_name(self, step_name: str | None) -> str:
        # A relaunched step (restart, a second benchmark) gets a fresh name: the
        # previous ``--rm`` container may not be removed yet.
        with self._lock:
            self._counter += 1
            suffix = (
                step_name if step_name and step_name not in self._steps else f"{step_name or 'proc'}{self._counter}"
            )
        return _CONTAINER_NAME_UNSAFE.sub("_", f"srtctl_{self._job_label()}_{suffix}")

    @staticmethod
    def docker_image(image: str) -> str:
        """An enroot/pyxis image URI as docker expects it (``nvcr.io#nvidia/x`` -> ``nvcr.io/nvidia/x``)."""
        for prefix in ("docker://", "dockerd://"):
            image = image.removeprefix(prefix)
        return image.replace("#", "/", 1)

    def docker_command(self, spec: LaunchSpec, container_name: str) -> list[str]:
        from srtctl.core.config import get_srtslurm_setting

        assert spec.container_image
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{JOB_LABEL}={self._job_label()}",
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--entrypoint",
            "",
        ]
        if spec.step_name:
            cmd.extend(["--label", f"{STEP_LABEL}={spec.step_name}"])
        for host, container in (spec.container_mounts or {}).items():
            cmd.extend(["-v", f"{host}:{container}"])
        for name in (*_PASSTHROUGH_ENV, *(spec.env_to_pass_through or ())):
            if name in os.environ:
                cmd.extend(["-e", name])
        for key, value in (spec.srun_export_env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(get_srtslurm_setting("docker_args") or [])
        cmd.append(self.docker_image(str(spec.container_image)))
        cmd.extend(self.build_task_command(spec))
        return cmd

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        hosts = list(spec.nodelist or self.hosts()[:1])
        if spec.ntasks != len(hosts):
            raise ValueError(
                f"launcher: docker runs one task per node; got ntasks={spec.ntasks} on {len(hosts)} node(s) "
                f"for {spec.step_name or shlex.join(spec.command[:3])}"
            )
        container: str | None = None
        if spec.container_image:
            container = self._container_name(spec.step_name)
            cmd = self.docker_command(spec, container)
        else:
            cmd = self.build_task_command(spec)
            if spec.srun_export_env:
                cmd = ["env", *(f"{k}={v}" for k, v in spec.srun_export_env.items()), *cmd]

        # One shell line per node; several run in parallel, like srun's tasks.
        lines = []
        for host in hosts:
            # A host process on another node has no container to `docker kill`: it runs
            # under a terminal and stops when its ssh client is signalled.
            line = shlex.join(_on(host, cmd, tty=container is None))
            if spec.output:
                output = Path(spec.output.replace("%N", host))
                output.parent.mkdir(parents=True, exist_ok=True)
                line += f" > {shlex.quote(str(output))} 2>&1"
            lines.append(line)
        script = f"exec {lines[0]}" if len(lines) == 1 else " & ".join(lines) + " & wait"
        logger.info("docker launcher command: %s", script)

        # start_new_session: the task (or the ssh/docker clients) get their own process
        # group, so a host process's whole tree can be signalled.
        popen = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.DEVNULL if spec.output else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if spec.step_name:
            with self._lock:
                self._steps[spec.step_name] = _DockerStep(popen=popen, hosts=hosts, container=container)
        return popen

    # -- signalling ----------------------------------------------------------

    def list_step_ids(self, job_id: str | None = None) -> dict[str, str] | None:
        with self._lock:
            return {
                name: step.container or str(step.popen.pid)
                for name, step in self._steps.items()
                if step.popen.poll() is None
            }

    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """``docker kill --signal`` on each node for a container, ``killpg`` for a host process.

        ``docker kill`` reaches the container even when its ``docker run`` (or ssh)
        client is gone, and killing the client alone does not stop the container. A
        host process here gets ``sig``; one on another node gets SIGHUP when ``killpg``
        stops its ssh client.
        """
        with self._lock:
            step = self._steps.get(step_name)
        if step is None or step.popen.poll() is not None:
            return False
        if step.container is not None:
            for host in step.hosts:
                result = subprocess.run(
                    _on(host, ["docker", "kill", f"--signal={sig}", step.container]),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode != 0:
                    logger.warning(
                        "docker kill --signal=%s %s on %s failed: %s", sig, step.container, host, result.stderr.strip()
                    )
                    return False
        else:
            try:
                os.killpg(step.popen.pid, signal.Signals[f"SIG{sig}"])
            except (ProcessLookupError, PermissionError, KeyError) as exc:
                logger.warning("killpg SIG%s %s failed: %s", sig, step_name, exc)
                return False
        logger.info("Sent SIG%s to %s", sig, step_name)
        return True

    # -- placement -----------------------------------------------------------

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

        return Nodes.from_nodelist(
            self.hosts(),
            frontend_dedicated_node=frontend_dedicated_node,
            client_dedicated_node=client_dedicated_node,
            etcd_nats_dedicated_node=etcd_nats_dedicated_node,
            colocate_dedicated_nodes=colocate_dedicated_nodes,
            engine_nodes=engine_nodes,
            pools=pools,
        )

    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        # On one machine every container runs with --network host.
        if len(self.hosts()) == 1:
            return "127.0.0.1"
        return _host_ip(hostname, network_interface)

    def job_id(self) -> str | None:
        return os.environ.get(JOB_ID_ENV)

    # -- submission ----------------------------------------------------------

    def submit(self, script_path: Path) -> str:
        now = time.time()
        return f"docker-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}"

    def start(self, job_id: str, job_output_dir: Path) -> int:
        """Run the staged job script in the foreground, teeing it to ``logs/sweep_<job_id>.log``.

        Runs in the foreground so a sweep's jobs take the nodes' GPUs one at a time.
        Afterwards, any container of the job still up on any node is removed.
        """
        script = job_output_dir / "sbatch_script.sh"
        log_path = job_output_dir / "logs" / f"sweep_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        env[JOB_ID_ENV] = job_id
        try:
            with open(log_path, "a") as log:
                proc = subprocess.Popen(
                    ["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True
                )
                assert proc.stdout is not None
                while True:
                    try:
                        for line in proc.stdout:
                            log.write(line)
                            log.flush()
                            sys.stderr.write(line)
                        break
                    except KeyboardInterrupt:
                        # Ctrl+C also reached the orchestrator (same process group); keep
                        # streaming while it cleans up.
                        continue
                return proc.wait()
        finally:
            remove = f"docker ps -aq --filter label={JOB_LABEL}={job_id} | xargs -r docker rm -f"
            for host in self.hosts():
                subprocess.run(_on(host, ["bash", "-c", remove]), capture_output=True, timeout=60, check=False)

    def shell_command(self, runtime: RuntimeContext, node: str) -> str:
        shell = f"docker exec -it $(docker ps -q --filter label={JOB_LABEL}={runtime.job_id} | head -1) bash"
        return shell if _is_this_host(node) else f"ssh -t {node} {shlex.quote(shell)}"

    def status_hint(self, job_id: str) -> str:
        return f"docker ps --filter label={JOB_LABEL}={job_id}"

    # -- validation ----------------------------------------------------------

    def validate(self, config: SrtConfig) -> list[str]:
        from srtctl.core.config import get_srtslurm_setting

        errors = []
        hosts = self.hosts()
        if config.total_nodes > len(hosts):
            errors.append(
                f"needs {config.total_nodes} nodes; docker_hosts in srtslurm.yaml lists {len(hosts)} ({', '.join(hosts)})"
            )
        if config.topology.het_components(
            infra_dedicated=config.infra_dedicated_node,
            cluster_default=get_srtslurm_setting("use_het_jobs", False),
        ):
            errors.append("resources.het_jobs is a Slurm heterogeneous job; set it false")
        if config.backend.get_srun_config().launch_per_endpoint:
            errors.append(
                f"engine {config.backend_type} launches each endpoint as one multi-task MPI step, "
                "which launcher: docker does not support"
            )
        images = {"model.container": str(config.model.container)}
        images.update({f"roles.{role}.container": image for role, image in config.role_containers.items()})
        for key, image in images.items():
            if os.path.expandvars(image).startswith(("/", "./")):
                errors.append(f"{key} is an enroot image file ({image}); launcher: docker needs a docker image name")
        return errors


@cache
def _host_ip(host: str, network_interface: str | None) -> str:
    """The address ``host`` is reached at, resolved on ``host`` itself (``ip_utils/get_node_ip.sh``)."""
    if _is_this_host(host):
        return get_local_ip(network_interface)
    script = (SCRIPTS_DIR / "get_node_ip.sh").read_text()
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash", "-s"],
        input=f"{script}\nget_local_ip {shlex.quote(network_interface or '')}\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    ip = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not ip:
        raise RuntimeError(f"could not resolve the IP of {host} over ssh: {result.stderr.strip()}")
    return ip
