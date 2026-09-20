# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict one-job preparation and durable Slurm submission.

The pilot API deliberately does not expand sweeps. A claimed intent is never
resubmitted, even when the submitter dies before recording Slurm's response.
Reconciliation is read-only and may leave an outcome unknown.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from srtctl.benchmarks import get_runner
from srtctl.core.config import cluster_config_scope, expand_engine_config_defaults, resolve_config_with_defaults
from srtctl.core.schema import ClusterConfig, SrtConfig

CAPABILITIES = (
    "prepared-v1",
    "durable-intent-v1",
    "custom-argv-v1",
    "controller-observation-v1",
    "bounded-cleanup-v1",
    "prepared-direct-listener-ownership-v1",
)


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently choosing a recipe value."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    loader.flatten_mapping(node)
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def strict_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return raw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"


def durable_write(path: Path, text: str, *, exclusive: bool = False) -> None:
    """Flush bytes and directory entry before returning; never replace a claim."""
    durable_mkdir(path.parent)
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _fsync_directory(directory.parent)


def source_identity(root: Path) -> dict[str, str]:
    """Hash executable source and packaged resources, including dirty changes."""
    paths = [
        p
        for p in (root / "src" / "srtctl").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "_version.py"
    ]
    paths += [p for p in (root / "pyproject.toml", root / "uv.lock") if p.exists()]
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def verify_loader_source(root: Path) -> None:
    """Prove the installed loader matches the source bundle being qualified."""
    package_root = Path(__file__).resolve().parents[1]
    source_package = root / "src" / "srtctl"
    if not source_package.is_dir():
        raise ValueError("Explicit srtctl_root must contain the pinned source package")
    for name, digest in source_identity(root).items():
        path = Path(name)
        if path.parts[:2] != ("src", "srtctl") or path.name == "_version.py":
            continue
        installed = package_root.joinpath(*path.parts[2:])
        if not installed.is_file() or sha256(installed) != digest:
            raise ValueError(f"Installed runtime differs from pinned source: {name}")


_RUNTIME_IDENTITY_SCRIPT = r"""
import hashlib, importlib.metadata, json, pathlib, platform, sys
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()
distributions = {}
for d in importlib.metadata.distributions():
    name = d.metadata.get("Name", "unknown").lower()
    files = {}
    for item in d.files or []:
        if str(item).endswith((".pyc", ".pyo")) or "__pycache__" in str(item): continue
        path = pathlib.Path(d.locate_file(item))
        if path.is_file(): files[str(item)] = digest(path)
    distributions[name] = {"version": d.version, "files": files}
print(json.dumps({"python": platform.python_version(), "architecture": platform.machine(),
                  "executable_sha256": digest(sys.executable), "distributions": distributions}, sort_keys=True))
"""


def runtime_identity(python: str) -> dict[str, Any]:
    if not Path(python).is_absolute():
        raise ValueError("runtime_python must be an absolute installed compute interpreter path")
    result = subprocess.run(
        [python, "-I", "-c", _RUNTIME_IDENTITY_SCRIPT], capture_output=True, text=True, check=True, timeout=180
    )
    return json.loads(result.stdout)


def allocation_summary(config: SrtConfig, profile: dict[str, Any]) -> dict[str, int]:
    """Use the same physical demand as native sbatch and validate native packing."""
    r = config.resources
    het = r.het_components(
        infra_dedicated=config.infra.etcd_nats_dedicated_node, cluster_default=profile.get("use_het_jobs", False)
    )
    if het is not None:
        raise ValueError("Prepared pilot does not support heterogeneous allocations")
    dedicated = sum(
        (config.infra.etcd_nats_dedicated_node, config.frontend.dedicated_node, config.benchmark.client_dedicated_node)
    )
    nodes = config.total_nodes + (1 if dedicated and config.benchmark.colocate_with_frontend else dedicated)
    # Run the native allocator before requesting real nodes. These names are
    # topology placeholders only; no submit-host file/IP existence checks.
    endpoints = config.backend.allocate_endpoints(
        num_prefill=r.num_prefill,
        num_decode=r.num_decode,
        num_agg=r.num_agg,
        gpus_per_prefill=r.gpus_per_prefill,
        gpus_per_decode=r.gpus_per_decode,
        gpus_per_agg=r.gpus_per_agg,
        gpus_per_node=r.gpus_per_node,
        available_nodes=tuple(f"prepared-node-{i}" for i in range(config.total_nodes)),
        spread_workers=r.spread_workers,
    )
    return {
        "nodes": nodes,
        "gpus_per_node": r.gpus_per_node,
        "serving_gpus": r.num_prefill * r.gpus_per_prefill
        + r.num_decode * r.gpus_per_decode
        + r.num_agg * r.gpus_per_agg,
        "workers": len(endpoints),
        "cardinality": 1,
    }


def _freeze_paths(config: dict[str, Any], profile: dict[str, Any]) -> None:
    def expand(value: str) -> str:
        result = os.path.expanduser(os.path.expandvars(value))
        if re.search(r"\$(?:[A-Za-z_]|\{)", result):
            raise ValueError(f"Unresolved environment variable in prepared path: {value!r}")
        return result

    for name in ("path", "container", "stage_dir"):
        if config.get("model", {}).get(name):
            config["model"][name] = expand(config["model"][name])
    for key in ("srtctl_root", "output_dir"):
        if profile.get(key):
            profile[key] = expand(profile[key])
    for owner, name in ((config, "container_mounts"), (profile, "default_mounts")):
        if owner.get(name):
            owner[name] = {expand(k): expand(v) for k, v in owner[name].items()}
    if config.get("extra_mount"):
        config["extra_mount"] = [expand(value) for value in config["extra_mount"]]


@dataclass(frozen=True)
class PreparedJob:
    prepared_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "state": "prepared",
            "prepared_dir": str(self.prepared_dir),
            "manifest_sha256": self.manifest_sha256,
            "resources": self.manifest["resources"],
            "output_root": self.manifest["output_root"],
            "capabilities": list(CAPABILITIES),
        }


def prepare_job(
    recipe_path: Path, profile_path: Path, prepared_dir: Path, *, expected_nodes: int, runtime_python: str
) -> PreparedJob:
    """Publish a complete immutable one-point snapshot before any scheduler call."""
    from srtctl.cli.submit import generate_minimal_sbatch_script

    recipe_path, profile_path, prepared_dir = (Path(p).resolve() for p in (recipe_path, profile_path, prepared_dir))
    raw = strict_yaml(recipe_path)
    if any(k in raw for k in ("sweep", "overrides", "variants", "include", "includes", "base")):
        raise ValueError("Prepared submission requires exactly one already selected recipe")
    profile_schema = ClusterConfig.Schema()
    profile = profile_schema.dump(profile_schema.load(strict_yaml(profile_path)))
    resolved = resolve_config_with_defaults(raw, profile)
    expand_engine_config_defaults(resolved)
    _freeze_paths(resolved, profile)
    config = SrtConfig.Schema().load(resolved)
    if config.backend_type == "vllm":
        from srtctl.backends.vllm import find_vllm_orchestration_recipe_flags

        orchestration_flags = find_vllm_orchestration_recipe_flags(config.backend)
        if orchestration_flags:
            raise ValueError(f"Native orchestration owns these engine flags: {orchestration_flags}")
        for mode in ("agg", "prefill", "decode"):
            args = config.backend.get_config_for_mode(mode)
            normalized = [name.replace("_", "-") for name in args]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Conflicting equivalent engine argument aliases in {mode}")
    if config.benchmark.sweep is not None or len(config.benchmark.get_concurrency_list()) > 1:
        raise ValueError("Prepared submission cannot expand a nested benchmark sweep")
    if config.benchmark.type == "manual":
        raise ValueError("Prepared pilot requires an explicit client")
    errors = get_runner(config.benchmark.type).validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    resources = allocation_summary(config, profile)
    if resources["nodes"] != expected_nodes or expected_nodes < 1:
        raise ValueError(f"Native allocation requires {resources['nodes']} nodes; queued demand is {expected_nodes}")
    reserved = {
        "nodes",
        "ntasks",
        "ntasks-per-node",
        "gpus-per-node",
        "array",
        "job-name",
        "output",
        "comment",
        "requeue",
        "no-requeue",
        "export",
        "wrap",
        "chdir",
        "clusters",
        "cluster",
        "parsable",
    }
    if reserved.intersection(config.sbatch_directives):
        raise ValueError("Prepared recipes cannot override allocation, identity or export directives")
    root = Path(profile.get("srtctl_root") or Path(__file__).resolve().parents[3]).resolve()
    verify_loader_source(root)
    profile["srtctl_root"] = str(root)
    output_root = Path(profile.get("output_dir") or root / "outputs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "native-logs").mkdir(parents=True, exist_ok=True)
    runtime = runtime_identity(runtime_python)
    source = source_identity(root)
    prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    if prepared_dir.exists():
        raise FileExistsError(f"Prepared bundle already exists; never overwrite: {prepared_dir}")
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=prepared_dir.parent) as temporary:
        staging = Path(temporary)
        files = {
            "config.yaml": yaml.safe_dump(resolved, sort_keys=False),
            "profile.yaml": yaml.safe_dump(profile, sort_keys=False),
        }
        with cluster_config_scope(profile):
            script = generate_minimal_sbatch_script(
                config,
                prepared_dir / "config.yaml",
                output_dir=output_root,
                prepared_dir=prepared_dir,
                runtime_python=runtime_python,
            )
        files["job.slurm"] = script
        for name, contents in files.items():
            durable_write(staging / name, contents)
        manifest = {
            "schema": 1,
            "prepared_dir": str(prepared_dir),
            "resources": resources,
            "output_root": str(output_root),
            "runtime_python": runtime_python,
            "runtime_identity": runtime,
            "runtime_root": str(root),
            "runtime_source": source,
            "input_sha256": {"recipe": sha256(recipe_path), "profile": sha256(profile_path)},
            "files": {name: sha256(staging / name) for name in files},
        }
        durable_write(staging / "manifest.json", _json(manifest))
        for path in staging.iterdir():
            path.chmod(0o444)
        # mkdir claim prevents a concurrent preparer replacing an accepted
        # directory with the same name; files remain hidden until published.
        reservation = prepared_dir.with_name(prepared_dir.name + ".prepare-claim")
        durable_write(reservation, str(os.getpid()), exclusive=True)
        try:
            if prepared_dir.exists():
                raise FileExistsError(prepared_dir)
            os.rename(staging, prepared_dir)
            _fsync_directory(prepared_dir.parent)
        finally:
            reservation.unlink()
    return load_prepared(prepared_dir)


def load_prepared(prepared_dir: Path, *, verify_runtime: bool = False) -> PreparedJob:
    prepared_dir = Path(prepared_dir).resolve()
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or manifest.get("prepared_dir") != str(prepared_dir):
        raise ValueError("Invalid prepared bundle identity")
    if set(manifest["files"]) != {"config.yaml", "profile.yaml", "job.slurm"}:
        raise ValueError("Invalid prepared file inventory")
    for name, expected in manifest["files"].items():
        if sha256(prepared_dir / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    if source_identity(Path(manifest["runtime_root"])) != manifest["runtime_source"]:
        raise ValueError("Prepared runtime source changed")
    verify_loader_source(Path(manifest["runtime_root"]))
    if verify_runtime and runtime_identity(manifest["runtime_python"]) != manifest["runtime_identity"]:
        raise ValueError("Prepared installed runtime dependencies changed")
    return PreparedJob(prepared_dir, manifest, sha256(manifest_path))


def _scheduler_env() -> dict[str, str]:
    # sbatch option environment takes precedence over script directives.
    return {
        k: v
        for k, v in os.environ.items()
        if (
            k in {"SLURM_CONF", "SLURM_CONF_SERVER"}
            or not k.startswith(("SBATCH_", "SRUN_", "SALLOC_", "SLURM_", "SRTCTL_", "SRTSLURM_"))
        )
        and k not in {"EVAL_ONLY", "RUN_EVAL", "INFMAX_WORKSPACE", "PYTHONPATH"}
    }


def submit_prepared(prepared_dir: Path, intent_id: str, cluster: str, journal_dir: Path) -> dict[str, Any]:
    prepared = load_prepared(prepared_dir, verify_runtime=True)
    receipt_path = intent_receipt_path(intent_id, cluster, journal_dir)
    intent_dir = receipt_path.parent
    token = intent_dir.name
    durable_mkdir(intent_dir.parent)
    receipt = {
        "schema": 1,
        "state": "claimed",
        "intent_id": intent_id,
        "cluster": cluster,
        "generation": 0,
        "prepared_dir": str(prepared.prepared_dir),
        "manifest_sha256": prepared.manifest_sha256,
        "job_id": None,
        "output_dir": None,
        "output_root": prepared.manifest["output_root"],
        "receipt_path": str(receipt_path),
        "accepted_ids": [],
        "scheduler_comment": "srtctl:" + token,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scheduler_context": {k: os.environ[k] for k in ("SLURM_CONF", "SLURM_CONF_SERVER") if k in os.environ},
    }
    try:
        intent_dir.mkdir()
        _fsync_directory(intent_dir.parent)
    except FileExistsError:
        if not receipt_path.exists():
            raise ValueError(f"Intent claim has incomplete journal; reconciliation required: {intent_dir}") from None
        prior = json.loads(receipt_path.read_text())
        if prior["manifest_sha256"] != prepared.manifest_sha256:
            raise ValueError("Intent already belongs to another prepared bundle") from None
        return prior
    durable_write(receipt_path, _json(receipt), exclusive=True)
    # stdout is a separate journal, written by the child even if the parent is
    # killed after scheduler acceptance and before receipt bookkeeping.
    with (intent_dir / "sbatch.stdout").open("x") as stdout, (intent_dir / "sbatch.stderr").open("x") as stderr:
        try:
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--no-requeue",
                    f"--export=ALL,SRTCTL_PREPARED_MANIFEST_SHA256={prepared.manifest_sha256}",
                    f"--comment={receipt['scheduler_comment']}",
                    str(prepared.prepared_dir / "job.slurm"),
                ],
                env=_scheduler_env(),
                stdout=stdout,
                stderr=stderr,
                timeout=120,
                check=False,
            )
            stdout.flush()
            os.fsync(stdout.fileno())
            stderr.flush()
            os.fsync(stderr.fileno())
            receipt["sbatch_returncode"] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            receipt["diagnostic"] = str(exc)
    receipt["state"] = "unknown"
    parsed = (intent_dir / "sbatch.stdout").read_text().strip()
    match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", parsed)
    if match:
        receipt.update(
            state="accepted",
            job_id=match.group(1),
            accepted_ids=[match.group(1)],
            output_dir=str(Path(prepared.manifest["output_root"]) / match.group(1)),
            scheduler_cluster=match.group(2),
        )
        try:
            durable_write(intent_dir / "accepted.json", _json(receipt), exclusive=True)
        except OSError as exc:
            receipt.update(state="unknown", diagnostic=f"Acceptance journal incomplete: {exc}")
    try:
        durable_write(receipt_path, _json(receipt))
    except OSError as exc:
        receipt.update(state="unknown", diagnostic=f"Receipt journal incomplete: {exc}")
    return receipt


def reconcile_receipt(receipt_path: Path) -> dict[str, Any]:
    """Recover journaled acceptance, otherwise find all matching scheduler IDs."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    _validate_scheduler_context(receipt)
    if receipt["state"] == "accepted":
        return receipt
    accepted = receipt_path.with_name("accepted.json")
    if accepted.exists():
        value = json.loads(accepted.read_text())
        if value["manifest_sha256"] != receipt["manifest_sha256"]:
            raise ValueError("Accepted journal disagrees with intent")
        durable_write(receipt_path, _json(value))
        return value
    ids = set()
    stdout = receipt_path.with_name("sbatch.stdout")
    if stdout.exists():
        match = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", stdout.read_text().strip())
        if match:
            ids.add(match.group(1))
    for command in (
        ["squeue", "--noheader", "--format=%i|%k"],
        [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--starttime",
            datetime.fromisoformat(receipt["created_at"]).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "--format=JobIDRaw,Comment",
        ],
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False, env={**os.environ, "TZ": "UTC"}
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[1] == receipt["scheduler_comment"] and re.fullmatch(r"[0-9]+", parts[0]):
                ids.add(parts[0])
    receipt["accepted_ids"] = sorted(ids, key=int)
    receipt["state"] = "accepted" if len(ids) == 1 else "unknown"
    if len(ids) == 1:
        job_id = next(iter(ids))
        # Allocation ownership survives deleted/mutated bundles. Reconciliation
        # must not lose a known scheduler ID because execution inputs drifted.
        receipt.update(job_id=job_id, output_dir=str(Path(receipt["output_root"]) / job_id))
    durable_write(receipt_path, _json(receipt))
    return receipt


def validate_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(Path(receipt_path).read_text())
    _validate_scheduler_context(receipt)
    if (
        receipt.get("schema") != 1
        or receipt.get("state") != "accepted"
        or not re.fullmatch(r"[0-9]+", receipt.get("job_id") or "")
    ):
        raise ValueError("An accepted receipt with one numeric job ID is required")
    if receipt.get("accepted_ids") != [receipt["job_id"]]:
        raise ValueError("Receipt has ambiguous accepted allocation ownership")
    return receipt


def intent_receipt_path(intent_id: str, cluster: str, journal_dir: Path) -> Path:
    """Pure ownership-path calculation, safe to call before interruptible submit."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", intent_id) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", cluster):
        raise ValueError("intent and cluster must be nonempty safe identifiers")
    token = hashlib.sha256(f"{cluster}\0{intent_id}\0{0}".encode()).hexdigest()
    return Path(journal_dir).resolve() / cluster / token / "receipt.json"


def cancel_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = validate_receipt(receipt_path)
    return {**_cancel_owned_job(receipt["job_id"], receipt.get("scheduler_comment")), "receipt_path": str(receipt_path)}


def _cancel_owned_job(job_id: str, comment: str | None) -> dict[str, Any]:
    from srtctl.core.observation import observe_job

    if not comment or not re.fullmatch(r"srtctl:[0-9a-f]{64}", comment):
        raise ValueError("Cancellation requires the recorded intent comment")
    observed = observe_job(job_id, expected_comment=comment)
    if observed["state"] == "unknown" or observed.get("identity_mismatch"):
        return {**observed, "state": "unknown"}
    if observed.get("terminal"):
        return {
            **observed,
            "state": "cancellation_requested",
            "already_terminal": True,
        }
    result = subprocess.run(["scancel", job_id], capture_output=True, text=True, timeout=30, check=False)
    return {
        "state": "cancellation_requested" if result.returncode == 0 else "unknown",
        "job_id": job_id,
        "diagnostic": result.stderr.strip(),
    }


def known_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(Path(receipt_path).read_text())
    _validate_scheduler_context(receipt)
    ids = receipt.get("accepted_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(i, str) or not re.fullmatch(r"[0-9]+", i) for i in ids)
    ):
        raise ValueError("Receipt has no known accepted numeric job IDs")
    if len(ids) != len(set(ids)) or not re.fullmatch(r"srtctl:[0-9a-f]{64}", receipt.get("scheduler_comment", "")):
        raise ValueError("Invalid known-allocation intent identity")
    return receipt


def _validate_scheduler_context(receipt: dict[str, Any]) -> None:
    current = {k: os.environ[k] for k in ("SLURM_CONF", "SLURM_CONF_SERVER") if k in os.environ}
    if "scheduler_context" in receipt and receipt["scheduler_context"] != current:
        raise ValueError("Use the same recorded Slurm controller configuration to reconcile or cancel this intent")


def cancel_known_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = known_receipt(receipt_path)
    outcomes = []
    for job_id in receipt["accepted_ids"]:
        try:
            outcomes.append(_cancel_owned_job(job_id, receipt["scheduler_comment"]))
        except (OSError, subprocess.TimeoutExpired) as exc:
            outcomes.append({"state": "unknown", "job_id": job_id, "diagnostic": str(exc)})
    return {
        "state": "cancellation_requested"
        if all(x["state"] == "cancellation_requested" for x in outcomes)
        else "unknown",
        "receipt_path": str(receipt_path),
        "jobs": outcomes,
    }
