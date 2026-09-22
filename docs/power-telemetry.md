# GPU Power Telemetry (`dcgm-power`)

The `dcgm-power` telemetry provider records raw per-GPU watts for every
allocated worker node, the topology needed to map each GPU to a `prefill`,
`decode`, or `agg` role, and the exact formal benchmark window for every
measured concurrency. It never integrates power into energy and never branches
on model, precision, or recipe; consumers integrate watts over the recorded
window themselves.

## How it works

- One DCGM exporter task runs on each allocated worker node, launched through
  the normal SLURM/process-registry path (one `srun` per heterogeneous group).
- A collector thread inside the orchestrator polls every exporter concurrently
  from the physical head node, so all sample timestamps and benchmark
  boundaries come from one clock.
- Only `DCGM_FI_DEV_POWER_USAGE` is parsed. Device identity comes from the
  `gpu` and `UUID` labels.
- **No in-tree benchmark stamps measurement windows yet**, so every run is
  currently unpublishable: it records `MEASUREMENT_WINDOW` reason codes, and
  `required: true` exits non-zero. The adapter belongs with the benchmark
  (for the current sa-bench path and its planned replacement alike): the
  benchmark child writes one window file per measured concurrency using the
  standalone `measurement_window.py` module and the windows directory passed
  in via `MEASUREMENT_WINDOW_DIR_ENV`.

## Configuration

```yaml
# NOTE: unsupported end-to-end until a benchmark adapter stamps windows —
# with this exact config every run is unpublishable and `required: true` fails.
benchmark:
  type: sa-bench          # future benchmark-side adapter must stamp the windows
  client_placement: head  # keeps sample and window clocks on one host
  isl: 8192
  osl: 1024
  concurrencies: [4]

telemetry:
  enabled: true
  provider: dcgm-power
  collect_interval_ms: 1000         # milliseconds between collector cycles; must be <= 3000
  storage_subdir: power             # relative to the run log directory
  required: true                    # exit non-zero when artifacts are unpublishable
  startup_timeout_seconds: 30
  request_timeout_seconds: 2
  collector_join_timeout_seconds: 12
  dcgm_exporter:
    container_image: dcgm-exporter  # alias, path, or registry URI
    port: 9401
```

`dcgm-power` needs **only** `dcgm_exporter`. Unlike `provider: scraper` it does
not require the top-level `container_image` or a `node_exporter`, because the
collector runs inside srtctl. Config loading validates the block and rejects
inconsistent values with actionable messages; in particular
`collect_interval_ms` must not exceed the 3-second max sample gap the audit
interpolates across (a window boundary farther than that from a sample is not
integrated). Gaps inside a window are reported per device as
`per_device_max_sample_gap_seconds` and never invalidate it. Telemetry stays
disabled by default and existing `provider: scraper` recipes are unchanged.
The collector join timeout must exceed two complete request-cycle budgets
(`2 * (2 * request_timeout_seconds + 1 second)`), covering a scrape already in
flight when shutdown starts plus the final bracketing scrape.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
├── scrape-timings.jsonl
└── windows/
    └── <benchmark-result-stem>.json
```

`samples.csv` has the exact header
`schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w`,
one row per observation, `(scrape_seq, hostname, gpu_index)` unique. Rows are
never interpolated, averaged, or role-attributed — role and heterogeneous
group live once in the manifest topology.

`manifest.json` records producer identity (version, git commit, exporter image
and its SHA-256), the sample interval, expected and observed device sets, the
topology mapping, the expected window list, the SHA-256 of the finalized
`samples.csv` bytes, terminal status, per-window coverage validation, and
reason codes. `status` is the lifecycle outcome;
`publication_valid` is the separate publication gate. Reason codes are stable
machine-readable strings enumerated in `srtctl/core/power/contract.py`.
The digest is required for offline publication validation, so packages created
before `samples_sha256` was recorded cannot be certified by this validator.

A window file records the formal benchmark boundaries on the head-node Unix
clock plus a monotonic `duration`, and points at the SA-Bench result it
brackets; result and window are boundary-identical.

With `required: true`, all artifacts are written first and the job then exits
non-zero whenever the terminal manifest is not publishable. With
`required: false`, measurement invalidity leaves the benchmark exit code
unchanged; an operational failure — a collector that cannot be joined, a
benchmark child that cannot be reaped, or an internal error while finalizing
telemetry — fails the job in either mode.

On `SIGTERM`/`SIGINT` or a critical-process death, the shared process registry
tears processes down before the collector finalizes, so the final scrape sees
dead endpoints. The manifest fails closed (`exporter_exited` /
`collector_interrupted` force `publication_valid=false`); the cost is that a
job that was simply cancelled can record `exporter_exited`.

## Diagnosing missed scrapes

The collector also writes `scrape-timings.jsonl`, one compact JSON record per
completed cycle (including the final bracketing cycle). This diagnostic sidecar
does not change the sample schema, cadence, timeouts, or publication policy.
Its `schema_version` is independent of the sample schema. Retain it alongside
the manifest and samples; join by `job_id`, `run_name`, `scrape_seq`, and endpoint
`hostname`. `collector_hostname` identifies the host running the collector.
Endpoint `sample_timestamp_unix` is the exact timestamp written to the CSV.

| Field | Meaning |
| --- | --- |
| `cycle_started_at_unix` | Head-host wall clock for correlation with other logs |
| `scheduled_monotonic`, `cycle_started_monotonic`, `schedule_lag_seconds` | Planned and actual cycle start on one monotonic clock; schedule/lag are null for manual and final scrapes |
| `endpoints[].request_start_delay_seconds` | Delay from cycle start to that worker entering the request, including dispatch and scheduling |
| `endpoints[].request_started_at_unix`, `request_finished_at_unix`, `request_duration_seconds` | Client-observed HTTP start/end and monotonic duration, including timeouts and HTTP errors |
| `endpoints[].http_status`, `reason_codes`, `row_count`, `settled` | Response status when available, parse/request outcome, parsed rows, and whether the worker returned before the cycle deadline |
| `endpoints[].error_type` | Exception class of a failed request (`ConnectTimeout`, `ReadTimeout`, `ConnectionError`, `HTTPError`); null on success. `ConnectTimeout` points at TCP connect (every scrape opens a new connection), `ReadTimeout` at the exporter's response |
| `poll_wall_seconds` | Wall time waiting for all endpoint workers, including request, parsing, and worker scheduling |
| `writer_lock_wait_seconds`, `sample_write_seconds`, `sample_write_completed` | Wait for the CSV writer lock, then append plus flush time and completion; flush is not fsync |
| `cycle_wall_seconds` | Time from cycle entry through CSV append/flush, excluding this cycle's diagnostic output |
| `previous_timing_write_seconds` | Previous cycle's JSON serialization/open/write/flush/close cost (null on the first cycle) |

A worker abandoned at the cycle deadline has `settled=false` and null request
timings/status, rather than an invented request duration. Request errors have
zero rows; no previous readings are reused. The manifest's
`max_scrape_duration_seconds` retains its historical population of successful
HTTP requests; use the sidecar to inspect failed requests.

Compare adjacent cycles: a long request versus a long CSV write identifies
where time was spent; an overdue cycle following either is not, by itself,
evidence of OS scheduling contention. A long client request can include network,
exporter, or collector-thread delays, so exporter/host evidence is still needed
to assign a root cause. Wall-clock adjustments can affect CSV timestamps; use
monotonic durations for elapsed-time comparisons.

The sidecar adds one open/write/flush/close per cycle under the existing writer lock,
after persisted samples have signalled readiness.
Inspect its measured cost on the actual storage before claiming negligible
overhead. Open/write/close I/O errors log a warning and disable diagnostics,
without changing collection validity. Blocked cycle diagnostic I/O uses the existing
bounded collector shutdown path. Initial creation is synchronous, like the existing
sample and manifest initialization; it has no separate I/O deadline.
A killed process, unhandled worker failure,
or still-blocked CSV write can leave the last cycle absent or the last JSON line
incomplete; absence is not proof that a request never started. No new monitoring
service or configuration is required.

## Re-validating a retained run

The artifact package is self-describing. The manifest supplies producer
identity, expected topology, runtime-only failure history, and a stored
verdict; the validator does not trust that verdict on its own:

```bash
srtctl-validate-power \
  --power-dir outputs/12345/logs/power \
  --result-root outputs/12345/logs \
  --expect-role prefill=4 --expect-role decode=4 \
  --require-distinct-het-groups
```

It recomputes every disk-derived claim from `samples.csv`, the result files,
and `windows/`, then requires the stored disk-derived reason subset and
`publication_valid` verdict to agree. Runtime-only reasons such as HTTP,
exporter-process, and collector failures cannot be reconstructed after the
live job is gone, so they are checked for a known v1 enum value and lifecycle
consistency instead. Exit status is `0` only when the recomputed package is
publishable, the stored verdict is `true`, and the two agree; otherwise it is
`1` and every failure is printed. The `--expect-*` flags optionally assert an
expected job shape for hardware canaries.
