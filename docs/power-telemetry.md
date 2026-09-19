# GPU Power Telemetry (`dcgm-power`)

The `dcgm-power` telemetry provider records raw per-GPU watts for every
allocated worker node, the topology needed to map each GPU to a `prefill`,
`decode`, or `agg` role, and the exact formal benchmark window for every
measured concurrency. It never integrates power into energy and never branches
on model, precision, or recipe; consumers integrate watts over the recorded
window themselves.

## How it works

- One configured GPU exporter task runs on each allocated worker node, launched through
  the normal SLURM/process-registry path (one `srun` per heterogeneous group).
- A collector thread inside the orchestrator polls every exporter concurrently
  from the physical head node, so all sample timestamps and benchmark
  boundaries come from one clock.
- The exporter's `power_profile` maps its watt metric and identity labels to
  the shared artifact fields. Omitting it preserves `DCGM_FI_DEV_POWER_USAGE`,
  `gpu`, `UUID`, and the existing NVIDIA scope. Optional utilization sources
  may change, but CSV columns and their units do not.
- The benchmark must stamp one measurement window per measured concurrency
  using `measurement_window.py` and `MEASUREMENT_WINDOW_DIR_ENV`. Without
  those windows the bundle is unpublishable and `required: true` exits non-zero.

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

GPU power needs **only** `dcgm_exporter`; the collector runs inside srtctl.
The provider name remains `dcgm-power` for compatibility. Tachometer is configured
separately under `observability.tachometer`. Config loading validates the block and rejects
inconsistent values with actionable messages; in particular
`collect_interval_ms` must not exceed the 3-second max sample gap the validator
accepts, or every window would fail `sample_gap_exceeded`. Telemetry stays
disabled by default.
The collector join timeout must exceed two complete request-cycle budgets
(`2 * (2 * request_timeout_seconds + 1 second)`), covering a scrape already in
flight when shutdown starts plus the final bracketing scrape.

### AMD SMI and other metric profiles

Keep the exporter command, image, and metric profile together in the cluster
configuration (`srtslurm.yaml`):

```yaml
default_gpu_exporter:
  container_image: amd-smi-image  # cluster alias for a ROCm image with amd-smi and python3
  port: 9401
  command: python3 /srtctl-runtime/amd_smi_exporter.py --port {port} --command-timeout 1
  power_profile:
    name: amd-smi-socket
    power_metric: amd_smi_socket_power_watts
    gpu_index_label: gpu
    gpu_uuid_label: uuid
    power_scope: gpu_socket_as_reported_by_amd_smi
    gpu_util_metric: amd_smi_gfx_activity_percent
    sm_active_metric: null
```

The recipe enables `telemetry: {enabled: true, provider: dcgm-power,
request_timeout_seconds: 3}` and supplies a window-producing benchmark as above.
The existing runtime mount provides `/srtctl-runtime`; no srtctl installation
inside the image is needed. Specify the image through the existing `containers`
alias map; do not infer it from the GPU type. Hardware execution and sampling
overhead of this adapter still require validation on the selected image/node.

Config loading copies the whole cluster exporter/profile into GPU power
telemetry. An explicit recipe `telemetry.dcgm_exporter` replaces it, including
an explicit `null`; cluster `default_gpu_exporter: null` disables the default.
Omitting or nulling `power_profile` inside an explicit exporter retains DCGM.
CPU-only telemetry never implicitly enables GPU power; combined CPU/GPU
collection explicitly supplies `telemetry.dcgm_exporter`. Enabled telemetry
with no remaining collector is rejected. A custom profile requires an explicit
exporter command, preventing an accidental DCGM launch.

The bundled adapter reads `amd-smi list --json` once at startup; device identity
is fixed for the exporter lifetime within an allocation. Each HTTP request runs
`amd-smi metric --power --usage --json` for fresh watts from all local GPUs. It
supports the ROCm 7.2 JSON shape, including the `gpu_data` envelope and explicit
units. Native command errors, timeouts, malformed JSON, and ambiguous or
partitioned identities fail closed; previous watts are never cached. Missing
labels and invalid watts retain the shared parser's reason codes. Its
metric-command timeout must fit within `request_timeout_seconds` with HTTP overhead.
The existing collector timestamps requests on the head node; AMD SMI owns the
sensor's sampling cadence. The adapter does not reproduce the native
InferenceX watch loop or its energy-accumulator sidecars.

**Measurement boundary:** `gpu_socket_as_reported_by_amd_smi` describes AMD SMI's
`power.socket_power` in watts. AMD documents this as current socket power on
MI300+ and average socket power on older supported GPUs; it is not a claim of
DCGM board-sensor equivalence, system power, UBB power, or wall power. See the
[AMD API reference](https://rocmdocs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html#amdsmi-get-power-info)
and the [ROCm 7.2 CLI mapping](https://github.com/ROCm/amdsmi/blob/rocm-7.2.0/amdsmi_cli/amdsmi_commands.py).

The InferenceX native AMD collector stays unchanged. Its separate srt-slurm
consumer currently pins DCGM's metric and scope and rejects these AMD bundles;
enabling downstream ingestion requires a separately reviewed consumer change.
This feature adds collection and offline validation only, with no consumer
migration or simultaneous native collection enabled.

A third Prometheus exporter needs only its command/image and a new mapping in
`power_profile`; no collector/parser dispatch or vendor registry is involved.
Metrics must already use watts, percent (`gpu_util_pct`) and fraction
(`sm_active`), respectively. Use null for unavailable utilization sources.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
└── windows/
    └── <benchmark-result-stem>.json
```

`samples.csv` has the exact header
`schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w,gpu_util_pct,sm_active`,
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

Non-default profiles add a self-contained `power_profile` object; the manifest's
`source_metric`, `power_scope`, and utilization provenance describe that profile.
The offline validator requires them to agree. NVIDIA defaults omit the additive
object and retain their previous artifact bytes. Schema versions, producer name,
the historical `dcgm_exporter` identity key, windows, and topology are unchanged.

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
