# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable entrypoints for strict prepared jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from srtctl.core.prepared import (
    CAPABILITIES,
    cancel_known_receipt,
    cancel_receipt,
    durable_write,
    intent_receipt_path,
    load_prepared,
    prepare_job,
    reconcile_receipt,
    submit_prepared,
)


def run_prepared(directory: str) -> None:
    """Compute-host entrypoint; validate before allocating any srun step."""
    import yaml

    from srtctl.cli.do_sweep import SweepOrchestrator, setup_logging
    from srtctl.core.config import cluster_config_scope, load_config
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.slurm import get_slurm_job_id

    prepared = load_prepared(Path(directory), verify_runtime=True)
    if os.environ.get("SRTCTL_PREPARED_MANIFEST_SHA256") != prepared.manifest_sha256:
        raise ValueError("Submitted manifest digest does not match the compute snapshot")
    job_id = get_slurm_job_id()
    if not job_id:
        raise ValueError("Prepared jobs must execute inside their Slurm allocation")
    output = Path(prepared.manifest["output_root"]) / job_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    scheduler_log = Path(prepared.manifest["output_root"]) / "native-logs" / f"{job_id}.log"
    log_link = output / "logs" / f"sweep_{job_id}.log"
    if not log_link.exists():
        log_link.symlink_to(scheduler_log)
    setup_logging()
    profile_path = prepared.prepared_dir / "profile.yaml"
    os.environ["SRTSLURM_CONFIG"] = str(profile_path)
    os.environ["SRTSLURM_CONFIG_SHA256"] = prepared.manifest["files"]["profile.yaml"]
    os.environ["SRTCTL_OUTPUT_DIR"] = str(output)
    os.environ["SRTCTL_SOURCE_DIR"] = prepared.manifest["runtime_root"]
    config = load_config(prepared.prepared_dir / "config.yaml", frozen=True)
    # Explicit node count is checked again against the actual allocation.
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != prepared.manifest["resources"]["nodes"]:
        raise ValueError("Actual Slurm allocation does not match prepared physical demand")
    profile = yaml.safe_load(profile_path.read_text())
    with cluster_config_scope(profile):
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()
    completion = {
        "schema": 1,
        "job_id": job_id,
        "manifest_sha256": prepared.manifest_sha256,
        "execution_success": exit_code == 0,
        "exit_code": exit_code,
        "cleanup_complete": getattr(orchestrator, "cleanup_complete", False),
        "restoration_success": getattr(orchestrator, "restoration_success", False),
    }
    durable_write(output / "completion.json", json.dumps(completion, sort_keys=True) + "\n")
    raise SystemExit(exit_code if all(completion[key] for key in ("cleanup_complete", "restoration_success")) else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--expected-nodes", type=int, required=True)
    prepare.add_argument("--runtime-python", required=True)
    submit = sub.add_parser("submit-prepared")
    submit.add_argument("--prepared-dir", type=Path, required=True)
    submit.add_argument("--intent", required=True)
    submit.add_argument("--cluster", required=True)
    submit.add_argument("--journal-dir", type=Path, required=True)
    intent = sub.add_parser("intent-path")
    intent.add_argument("--intent", required=True)
    intent.add_argument("--cluster", required=True)
    intent.add_argument("--journal-dir", type=Path, required=True)
    for name in ("wait", "cancel", "reconcile", "cancel-known", "wait-known"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        if name in {"wait", "wait-known"}:
            command.add_argument("--timeout", type=float, required=True)
            command.add_argument("--poll", type=float, default=5)
            command.add_argument("--until-terminal", action="store_true", help="Wait for physical closure after cancel")
    sub.add_parser("capabilities")
    run = sub.add_parser("run-prepared")
    run.add_argument("directory")
    for command in sub.choices.values():
        command.add_argument("--json", action="store_true", help="Emit one JSON record (also the default)")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-prepared":
            run_prepared(args.directory)
            return 0
        if args.command == "prepare":
            result = prepare_job(
                args.recipe,
                args.profile,
                args.output,
                expected_nodes=args.expected_nodes,
                runtime_python=args.runtime_python,
            ).to_dict()
        elif args.command == "intent-path":
            result = {
                "schema": 1,
                "state": "intent",
                "receipt_path": str(intent_receipt_path(args.intent, args.cluster, args.journal_dir)),
            }
        elif args.command == "submit-prepared":
            result = submit_prepared(args.prepared_dir, args.intent, args.cluster, args.journal_dir)
        elif args.command == "wait":
            from srtctl.core.observation import wait_receipt

            result = wait_receipt(
                args.receipt, timeout=args.timeout, poll=args.poll, until_terminal=args.until_terminal
            )
        elif args.command == "cancel":
            result = cancel_receipt(args.receipt)
        elif args.command == "cancel-known":
            result = cancel_known_receipt(args.receipt)
        elif args.command == "wait-known":
            from srtctl.core.observation import wait_known_receipt

            result = wait_known_receipt(args.receipt, timeout=args.timeout, poll=args.poll)
        elif args.command == "reconcile":
            result = reconcile_receipt(args.receipt)
        else:
            result = {"schema": 1, "state": "supported", "capabilities": list(CAPABILITIES)}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return (
            0
            if result["state"]
            in {"prepared", "accepted", "completed", "closed", "intent", "cancellation_requested", "supported"}
            else 2
        )
    except Exception as exc:  # noqa: BLE001 - CLI protocol emits structured failures for external inputs
        print(json.dumps({"schema": 1, "state": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
