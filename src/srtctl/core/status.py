# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fire-and-forget status reporter for external job tracking.

This module provides optional status reporting to one or more external API endpoints.
If no endpoints are configured or all are unreachable, operations silently continue.
The API contract is defined in srtctl.contract.

Configuration (in srtslurm.yaml or recipe YAML):
    # Single endpoint (backward-compatible)
    reporting:
      status:
        endpoint: "https://status.example.com"

    # Multiple endpoints
    reporting:
      status:
        endpoints:
          - "https://status.example.com"
          - "https://status2.example.com"

    # Both (merged, deduplicated)
    reporting:
      status:
        endpoint: "https://status.example.com"
        endpoints:
          - "https://status2.example.com"

    # Also push new log and metric output every 10 seconds
    reporting:
      status:
        endpoint: "https://status.example.com"
        logging-stream-interval: 10
"""

import codecs
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from srtctl.contract import JobCreatePayload, JobStage, JobStatus, JobUpdatePayload

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ReportingConfig, ReportingStatusConfig, SrtConfig

logger = logging.getLogger(__name__)

# Environment variable the reporter reads its bearer token from unless
# ``reporting.status.token_env`` names another one. The token itself never
# appears in a recipe or srtslurm.yaml: the resolved config is written to the
# lockfile and copied into the log directory that reporting.s3 uploads.
DEFAULT_TOKEN_ENV = "SRTCTL_STATUS_TOKEN"


def _token_env(status: "ReportingStatusConfig | None") -> str:
    return (status.token_env if status and status.token_env else None) or DEFAULT_TOKEN_ENV


def _auth_headers(token_env: str) -> dict[str, str]:
    """``Authorization: Bearer`` header when the token variable is set, else nothing."""
    token = os.environ.get(token_env)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _log_rejection(action: str, endpoint: str, status_code: int, token_env: str) -> None:
    """Auth failures and redirects are configuration errors, so they warn; anything else stays at DEBUG.

    Redirects matter because a collector behind a login page (an SSO proxy, for
    example) answers every request with a 302 to the sign-in page. Following it
    would land on an HTML page with HTTP 200 and look like success.
    """
    if 300 <= status_code < 400:
        logger.warning(
            "%s to %s was redirected (HTTP %d); the endpoint is behind a login page or proxy, nothing was recorded",
            action,
            endpoint,
            status_code,
        )
    elif status_code in (401, 403):
        logger.warning(
            "%s to %s rejected (HTTP %d); check the bearer token in $%s", action, endpoint, status_code, token_env
        )
    else:
        logger.debug("%s to %s failed: HTTP %d", action, endpoint, status_code)


def _cluster_setting() -> str | None:
    """The cluster name from srtslurm.yaml, or None; never raises (reporting must not break a run)."""
    try:
        from srtctl.core.config import get_srtslurm_setting  # lazy: core.config is heavier than this module

        value = get_srtslurm_setting("cluster")
    except Exception:  # noqa: BLE001 - any config problem just means "unknown cluster"
        return None
    return str(value) if value else None


def _resolve_endpoints(status: "ReportingStatusConfig | None") -> tuple[str, ...]:
    """Merge endpoint + endpoints into a deduplicated tuple with trailing slashes stripped.

    Args:
        status: ReportingStatusConfig (may be None)

    Returns:
        Tuple of unique endpoint URLs
    """
    if not status:
        return ()

    seen: dict[str, None] = {}
    if status.endpoint:
        seen[status.endpoint.rstrip("/")] = None
    if status.endpoints:
        for ep in status.endpoints:
            seen[ep.rstrip("/")] = None
    return tuple(seen)


@dataclass(frozen=True)
class StatusReporter:
    """Fire-and-forget status reporter.

    Reports job status to one or more external APIs if reporting.status endpoints
    are configured. All operations are non-blocking and failures are silently logged.

    Usage:
        reporter = StatusReporter.from_config(config.reporting, job_id="12345")
        reporter.report(JobStatus.WORKERS, stage=JobStage.WORKERS)
    """

    job_id: str
    api_endpoints: tuple[str, ...] = ()
    timeout: float = 5.0
    # Environment variable holding the bearer token; see DEFAULT_TOKEN_ENV.
    token_env: str = DEFAULT_TOKEN_ENV
    # Connection attempts per endpoint for each report; see _put.
    attempts: int = 2

    @classmethod
    def from_config(cls, reporting: "ReportingConfig | None", job_id: str) -> "StatusReporter":
        """Create reporter from reporting config.

        Args:
            reporting: ReportingConfig from srtslurm.yaml or recipe
            job_id: SLURM job ID

        Returns:
            StatusReporter instance (disabled if no endpoints configured)
        """
        status = reporting.status if reporting else None
        endpoints = _resolve_endpoints(status)
        if endpoints:
            logger.info("Status reporting enabled: %s", ", ".join(endpoints))

        return cls(job_id=job_id, api_endpoints=endpoints, token_env=_token_env(status))

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return len(self.api_endpoints) > 0

    def _now_iso(self) -> str:
        """Get current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _put(self, payload: dict) -> bool:
        """Send PUT to all endpoints, up to ``attempts`` tries each. Returns True if any succeeded.

        A lost PUT is a lost event in the collector, and a flaky egress path can
        eat one attempt's whole connect timeout on a dead address, so one retry
        buys a lot. The final failure is a WARNING in the sweep log; the run itself
        is never affected.
        """
        any_success = False
        headers = _auth_headers(self.token_env)
        for endpoint in self.api_endpoints:
            url = f"{endpoint}/api/jobs/{self.job_id}"
            for attempt in range(1, self.attempts + 1):
                try:
                    # allow_redirects=False: a 3xx is a failure, never something to follow (see _log_rejection).
                    response = requests.put(
                        url, json=payload, headers=headers, timeout=self.timeout, allow_redirects=False
                    )
                except requests.exceptions.RequestException as e:
                    if attempt < self.attempts:
                        logger.debug(
                            "Status report to %s failed (attempt %d/%d): %s", endpoint, attempt, self.attempts, e
                        )
                        continue
                    logger.warning("Status report to %s lost after %d attempts: %s", endpoint, self.attempts, e)
                    break
                if response.status_code == 200:
                    logger.debug("Status reported to %s", endpoint)
                    any_success = True
                else:
                    _log_rejection("Status report", endpoint, response.status_code, self.token_env)
                break
        return any_success

    def report(
        self,
        status: JobStatus,
        stage: JobStage | None = None,
        message: str | None = None,
    ) -> bool:
        """Report status update (fire-and-forget).

        Args:
            status: New job status
            stage: Current execution stage
            message: Optional human-readable message

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        payload = JobUpdatePayload(
            status=status.value,
            updated_at=self._now_iso(),
            stage=stage.value if stage else None,
            message=message,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_started(
        self,
        config: "SrtConfig",
        runtime: "RuntimeContext",
        resource_snapshot: dict | None = None,
    ) -> bool:
        """Report job started with initial metadata.

        Args:
            config: Job configuration
            runtime: Runtime context with computed values

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        resource_snapshot = resource_snapshot or {}
        metadata = {
            "model": {
                "path": str(config.model.path),
                "precision": config.model.precision,
            },
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
                "cpu_allocation": resource_snapshot.get("cpus"),
                "cpu_check": resource_snapshot.get("cpu_check"),
            },
            "benchmark": {
                "type": config.benchmark.type,
            },
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            "head_node": runtime.nodes.head,
            # Where the run writes its logs on the cluster filesystem. A collector
            # on the same filesystem (srtctl status-server on a login node) can
            # open them directly; logs_url only appears later if reporting.s3 is set.
            "log_dir": str(runtime.log_dir),
            # Identity, repeated from the submit-time POST. That POST is one attempt
            # from the login node; when it is lost (a flaky egress path), the
            # collector's placeholder row takes its name and cluster from here.
            "job_name": getattr(config, "name", None),
            "cluster": _cluster_setting(),
        }

        payload = JobUpdatePayload(
            status=JobStatus.STARTING.value,
            stage=JobStage.STARTING.value,
            message=f"Job started on {runtime.nodes.head}",
            started_at=self._now_iso(),
            updated_at=self._now_iso(),
            metadata=metadata,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_completed(self, exit_code: int, logs_url: str | None = None) -> bool:
        """Report job completed with exit code and optional logs pointer.

        Benchmark results themselves are intentionally not carried in this PUT.
        S3 is the source of truth for artifacts; the collector stores the
        pointer (``logs_url``) and consumers fetch the full rollup from S3.

        Args:
            exit_code: Process exit code (0 = success)
            logs_url: URL where logs were uploaded (e.g. s3://bucket/prefix/job/)

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled:
            return False

        status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
        message = "Benchmark completed successfully" if exit_code == 0 else f"Job failed with exit code {exit_code}"

        payload = JobUpdatePayload(
            status=status.value,
            stage=JobStage.CLEANUP.value,
            message=message,
            completed_at=self._now_iso(),
            updated_at=self._now_iso(),
            exit_code=exit_code,
            logs_url=logs_url,
        )

        return self._put(payload.model_dump(exclude_none=True))

    def report_artifacts(self, logs_url: str) -> bool:
        """Push an artifacts pointer (``logs_url``) eagerly, mid-run.

        Used after S3 sync completes but before later stages (AI analysis,
        cleanup) that can hang or fail. Keeps the status value at its current
        stage (``benchmark``) so this PUT is purely an artifact-pointer update,
        not a lifecycle transition. The collector merges ``logs_url`` in; a
        later ``report_completed`` will reassert it idempotently.

        Args:
            logs_url: URL where logs are being / have been uploaded

        Returns:
            True if reported to at least one endpoint, False otherwise
        """
        if not self.enabled or not logs_url:
            return False

        payload = JobUpdatePayload(
            status=JobStatus.BENCHMARK.value,
            stage=JobStage.CLEANUP.value,
            message="Artifacts uploaded",
            updated_at=self._now_iso(),
            logs_url=logs_url,
        )

        return self._put(payload.model_dump(exclude_none=True))


def create_job_record(
    reporting: "ReportingConfig | None",
    job_id: str,
    job_name: str,
    cluster: str | None = None,
    recipe: str | None = None,
    metadata: dict | None = None,
    attempts: int = 2,
    submitted_at: str | None = None,
) -> bool:
    """Create initial job record in status APIs (called at submission time).

    This is a standalone function used by submit.py before the job starts.
    Sends to all configured endpoints. Each endpoint gets up to ``attempts``
    tries: the login node's path to a collector can be flaky (a dead address
    behind a round-robin name eats the whole connect timeout), and this POST is
    the only message carrying the job name, cluster and recipe. A final failure
    is a WARNING because the person running ``srtctl apply`` is right there;
    the run still shows up once it starts, named from ``report_started``.

    Args:
        reporting: ReportingConfig from srtslurm.yaml or recipe
        job_id: SLURM job ID
        job_name: Job/config name
        cluster: Cluster name (optional)
        recipe: Path to recipe file (optional)
        metadata: Job metadata dict (may include "tags" list)
        attempts: Connection attempts per endpoint before giving up
        submitted_at: ISO 8601 submit time; defaults to now. Pass the real time when
            re-posting a job after the fact, the collector only ever moves it earlier.

    Returns:
        True if created on at least one endpoint, False otherwise
    """
    status = reporting.status if reporting else None
    endpoints = _resolve_endpoints(status)
    if not endpoints:
        return False
    token_env = _token_env(status)
    headers = _auth_headers(token_env)

    payload = JobCreatePayload(
        job_id=job_id,
        job_name=job_name,
        submitted_at=submitted_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        cluster=cluster,
        recipe=recipe,
        metadata=metadata,
    )
    payload_dict = payload.model_dump(exclude_none=True)

    any_success = False
    for endpoint in endpoints:
        url = f"{endpoint}/api/jobs"
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(url, json=payload_dict, headers=headers, timeout=5.0, allow_redirects=False)
            except requests.exceptions.RequestException as e:
                if attempt < attempts:
                    logger.debug("Job record creation on %s failed (attempt %d/%d): %s", endpoint, attempt, attempts, e)
                    continue
                logger.warning(
                    "Job record creation on %s failed after %d attempts: %s. The run still appears in the collector "
                    "once it starts; its name and cluster come from the started report.",
                    endpoint,
                    attempts,
                    e,
                )
                break
            if response.status_code == 201:
                logger.debug("Job record created on %s: %s", endpoint, job_id)
                any_success = True
            else:
                _log_rejection("Job record creation", endpoint, response.status_code, token_env)
            break

    return any_success


# Append-only text files under the log directory: engine and sweep logs, and
# metric samples (power samples.csv, host_samples.jsonl, profile_export.jsonl).
STREAM_SUFFIXES = (".out", ".err", ".log", ".csv", ".jsonl")
# Tachometer's capture is binary (a current.arrow snapshot plus out-N.parquet
# segments), so the streamer appends its new rows here as JSON lines.
TACHOMETER_STREAM_FILE = "tachometer_rows.jsonl"
# Raw bytes per POST; JSON escaping can grow text up to 6x, which stays under the collector's 1 MiB cap.
STREAM_REQUEST_BYTES = 128 * 1024
STREAM_SHUTDOWN_SECONDS = 5.0


class LogStreamer:
    """Every ``interval`` seconds, POST the bytes appended to the run's log and metric files.

    Each chunk carries its file (relative to ``log_dir``), byte ``offset`` and
    ``size``, so the collector stores deltas by position and a resend is a no-op.
    Offsets advance per endpoint only on success, so an unreachable collector
    catches up on a later flush. With ``tachometer_dir`` set, previously unseen
    observations are appended to ``TACHOMETER_STREAM_FILE``, which then
    streams like any other file.
    """

    def __init__(self, reporter: StatusReporter, log_dir: Path, interval: float, tachometer_dir: Path | None = None):
        self.reporter = reporter
        self.log_dir = log_dir
        self.interval = interval
        self.tachometer_dir = tachometer_dir
        self._offsets: dict[tuple[str, str], int] = {}
        self._pending: dict[tuple[str, str], bytes] = {}
        self._shutdown_deadline: float | None = None
        self._tachometer_done: set[Path] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="status-log-stream", daemon=True)

    @classmethod
    def from_config(
        cls,
        reporting: "ReportingConfig | None",
        reporter: StatusReporter,
        log_dir: Path,
        tachometer_dir: Path | None = None,
    ) -> "LogStreamer | None":
        interval = reporting.status.logging_stream_interval if reporting and reporting.status else None
        if not reporter.enabled or interval is None:
            return None
        return cls(reporter, log_dir, interval, tachometer_dir)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Give the worker a bounded final flush; telemetry cannot delay job completion indefinitely."""
        self._shutdown_deadline = time.monotonic() + STREAM_SHUTDOWN_SECONDS
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=STREAM_SHUTDOWN_SECONDS)
            if self._thread.is_alive():
                logger.warning(
                    "Log streaming did not finish within %.1fs; local logs are retained", STREAM_SHUTDOWN_SECONDS
                )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.flush()
        self.flush()

    def _expired(self) -> bool:
        return self._shutdown_deadline is not None and time.monotonic() >= self._shutdown_deadline

    def flush(self) -> None:
        """Upload a finite snapshot of each file, retaining unacknowledged bytes verbatim."""
        try:
            if self._expired():
                return
            if self.tachometer_dir is not None:
                self._export_tachometer()
            files = sorted(p for p in self.log_dir.rglob("*") if p.suffix in STREAM_SUFFIXES and p.is_file())
            for endpoint in self.reporter.api_endpoints:
                for path in files:
                    if self._expired():
                        return
                    rel = path.relative_to(self.log_dir).as_posix()
                    key = (endpoint, rel)
                    try:
                        with path.open("rb") as f:
                            # Do not chase an actively growing file forever and starve other files/endpoints.
                            end = os.fstat(f.fileno()).st_size
                            while not self._expired():
                                offset = self._offsets.get(key, 0)
                                data = self._pending.get(key)
                                if data is None:
                                    if offset >= end:
                                        break
                                    f.seek(offset)
                                    data = f.read(min(STREAM_REQUEST_BYTES, end - offset))
                                    # Keep a partial UTF-8 character for the next read/flush. Invalid
                                    # source bytes are replaced, but valid split characters stay intact.
                                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                                    decoder.decode(data, final=self._stop.is_set() and offset + len(data) >= end)
                                    pending_bytes, _ = decoder.getstate()
                                    if pending_bytes:
                                        data = data[: -len(pending_bytes)]
                                    if not data:
                                        break
                                    self._pending[key] = data
                                if not self._post(endpoint, rel, data):
                                    # One failed collector must not incur a timeout for every file.
                                    break
                                del self._pending[key]
                            else:
                                return
                            if key in self._pending:
                                break
                    except OSError as e:
                        logger.debug("Log stream skipped %s: %s", rel, e)
        except Exception:
            logger.warning("Log streaming flush failed; will retry", exc_info=True)

    def _post(self, endpoint: str, rel: str, data: bytes) -> bool:
        offset = self._offsets.get((endpoint, rel), 0)
        chunk = {"file": rel, "offset": offset, "size": len(data), "data": data.decode(errors="replace")}
        payload: dict = {"chunks": [chunk]}
        cluster = _cluster_setting()
        if cluster:
            payload["metadata"] = {"cluster": cluster}
        timeout = self.reporter.timeout
        if self._shutdown_deadline is not None:
            timeout = min(timeout, self._shutdown_deadline - time.monotonic())
            if timeout <= 0:
                return False
        try:
            response = requests.post(
                f"{endpoint}/api/jobs/{self.reporter.job_id}/logs",
                json=payload,
                headers=_auth_headers(self.reporter.token_env),
                timeout=timeout,
                allow_redirects=False,
            )
            if response.status_code != 200:
                _log_rejection("Log stream", endpoint, response.status_code, self.reporter.token_env)
                return False
            acknowledgment = response.json()
            if (
                not isinstance(acknowledgment, dict)
                or acknowledgment.get("job_id") != self.reporter.job_id
                or acknowledgment.get("stored") not in (0, 1)
            ):
                logger.debug("Log stream to %s returned an invalid acknowledgment", endpoint)
                return False
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.debug("Log stream to %s failed: %s", endpoint, e)
            return False
        self._offsets[(endpoint, rel)] = offset + len(data)
        return True

    def _export_tachometer(self) -> None:
        """Export observations once across snapshots, rotations and reordered compactions.

        Compaction replaces local segments and adds ``metric_name_clean``. A
        disk-backed identity index avoids both timestamp assumptions and keeping
        millions of sample identities in memory.
        """
        if self.tachometer_dir is None:
            return

        import pyarrow as pa
        from pyarrow import ipc

        from srtctl.dsight.metrics import batches, capture_files  # lazy: pyarrow

        try:
            files = [p for p in capture_files(self.tachometer_dir) if p not in self._tachometer_done]
            index_path = self.log_dir / ".tachometer-stream.sqlite3"
            with (
                closing(sqlite3.connect(index_path)) as index,
                (self.log_dir / TACHOMETER_STREAM_FILE).open("a+b", buffering=0) as out,
            ):
                index.execute("CREATE TABLE IF NOT EXISTS exported (identity BLOB PRIMARY KEY) WITHOUT ROWID")
                for path in files:
                    if self._expired():
                        return
                    if path.suffix == ".arrow":
                        # The writer truncates current.arrow in place. Copy before
                        # decoding: mmap can SIGBUS when the next snapshot shrinks.
                        data = path.read_bytes()
                        source = pa.BufferReader(data)
                        if data.startswith(b"ARROW1"):
                            reader = ipc.open_file(source)
                            capture_batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
                        else:
                            capture_batches = iter(ipc.open_stream(source))
                    else:
                        capture_batches = batches(path)
                    for batch in capture_batches:
                        if self._expired():
                            return
                        start = out.tell()
                        pending = bytearray()
                        try:
                            with index:
                                for row in batch.to_pylist():
                                    row.pop("metric_name_clean", None)
                                    encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                                    inserted = index.execute(
                                        "INSERT OR IGNORE INTO exported VALUES (?)", (hashlib.sha256(encoded).digest(),)
                                    ).rowcount
                                    if inserted:
                                        pending.extend(encoded)
                                if out.write(pending) != len(pending):
                                    raise OSError("Incomplete Tachometer JSONL write")
                        except Exception:
                            # Retry an incomplete batch without retaining half of
                            # its JSONL output or advancing its identity index.
                            out.seek(start)
                            out.truncate()
                            raise
                    if path.suffix == ".parquet":
                        self._tachometer_done.add(path)
        except Exception as e:  # noqa: BLE001 - unreadable/rotating captures must not stop streaming
            logger.debug("Tachometer export skipped: %s", e)
