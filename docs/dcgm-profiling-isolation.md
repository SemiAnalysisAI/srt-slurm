# Keep power collection running during profiling recovery

The optional exporter build in `docker/Dockerfile.dcgm-power-exporter` patches
NVIDIA dcgm-exporter **4.6.0-4.8.3**, commit
`181290c399d46a9b905e083d0204348be63cb436`. It addresses a synchronous recovery
path: an HTTP scrape holds the collector mutex while replacing watches and
calling `UpdateAllFields()`. A slow profiling repair can therefore delay power
responses beyond the SRT request timeout.

The patch creates separate ordinary-field and profiling-field watches. Ordinary
watches, including label fields, remain installed. One background repair owns
only profiling resources and does not hold the scrape mutex or force a global
field update. Every scrape reads ordinary values again; it never substitutes
cached power. Profiling is omitted while repairing and becomes visible on later
scrapes after fresh samples arrive. Existing freshness checks and retry backoff
remain in place. Shutdown joins repair before releasing DCGM resources.

This fixes exporter recovery coupling. It does not explain why profiling first
became stale. Profiling reads and ordinary reads still use the same native DCGM
provider: a driver failure, globally blocked native call, or permanently stuck
repair can still affect collection or shutdown. This is not process isolation.

## Build, activate, and roll back

From the srt-slurm repository root, on a Docker builder for the target platform:

```sh
docker build --platform linux/arm64 \
  -f docker/Dockerfile.dcgm-power-exporter \
  -t dcgm-power-exporter:4.6.0-4.8.3-power-isolation .
```

The build checks patch application, runs the relevant Go unit packages, and
replaces only `/usr/bin/dcgm-exporter` in the matching NVIDIA runtime. Use
`linux/amd64` for x86 nodes. This recipe is opt-in; **changing the SRT git pin
alone does not activate it**. After image review and publication through the
usual cluster process, resolve the image by digest, import it as usual, and
record the resulting squashfs checksum. Point a cluster container alias at it:

```yaml
# srtslurm.yaml: replace the path with the approved immutable image artifact.
containers:
  dcgm-power-isolated: /path/to/dcgm-power-exporter.sqsh
```

Select that alias through the existing `telemetry.dcgm_exporter.container_image`
field; see [the example](../examples/features/power-profiling-isolation.yaml).
Retain existing counter-file arguments and mounts when applying the image to a
real recipe. The patch changes neither counters nor sampling configuration.
Keep exporter cadence at 100 ms, SRT cadence at 1 s, request timeout at 2 s,
and existing power-validity rules for a matched comparison.

Before rollout, run the original failing model/topology point with the new image
and inspect timing records, source timestamps, profiling recovery, and measured
windows. Roll back by restoring the previous exporter image alias/digest and
rerunning only affected points. Existing data and schemas need no migration.

## Reproduce the regressions

In a separate dcgm-exporter checkout at the commit above:

```sh
git apply --check /path/to/srt-slurm/docker/dcgm-exporter-profiling-isolation.patch
git apply /path/to/srt-slurm/docker/dcgm-exporter-profiling-isolation.patch
go test -short ./internal/pkg/collector ./internal/pkg/devicewatchlistmanager ./internal/pkg/server
go test -race -short ./internal/pkg/collector ./internal/pkg/devicewatchlistmanager
```

`TestProfilingRepairDoesNotBlockFreshPower` blocks profiling rewatch, verifies
changing power values while the ordinary watch remains installed, then verifies
profiling recovery and cleanup. Copying this test alone onto the unpatched base
fails with `profiling repair blocked a healthy power scrape`.

The patch also includes the opt-in `TestLiveProfilingIsolation`. It uses real
DCGM, collector, registry, and HTTP handler; only a profiling error status and a
five-second watcher delay are injected. Build its test executable on the target
architecture:

```sh
go test -c -o /path/to/server-probe ./internal/pkg/server
```

On an explicitly owned four-GPU allocation, run it in the matching DCGM runtime
with `POWERX_LIVE_PROBE_DIR` pointing at a fresh shared output directory:

```sh
POWERX_LIVE_PROBE_DIR=/shared/probe /path/to/server-probe \
  -test.run '^TestLiveProfilingIsolation$' -test.v -test.timeout=110s
```

Concurrently, on the same host, run the SRT consumer from this repository with
its Python dependencies. Mount the same output directory at the same path; the
HTTP endpoint uses loopback. Set provenance explicitly:

```sh
export SRT_PROBE_COMMIT=$(git rev-parse HEAD)
export SRT_PROBE_IMAGE=/path/to/dcgm-exporter.sqsh
export SRT_PROBE_COMMAND='/path/to/server-probe -test.run ^TestLiveProfilingIsolation$'
PYTHONPATH=src python tests/manual/consume_dcgm_probe.py /shared/probe
```

Repeat on an unpatched exporter checkout with only the live test file copied in.
The consumer writes a diagnostic window, CSV, manifest, and timing records through
the real SRT collection/validation path. It deliberately does not publish or
ingest them. Its artifact-validator result alone is not the isolation gate:
compare maximum sample gap, request errors, advancing original DCGM timestamps
during repair, and profiling samples after recovery. Diagnostic fixtures are not
model benchmark results.

Check retained candidate output with the offline gate (the unpatched arm should
fail this command):

```sh
python tests/manual/check_dcgm_probe.py /shared/probe
```

For normal-path comparison, start each real exporter binary with the same
counter file and `--collect-interval=100 --address :9401`; write
`http://127.0.0.1:9401` to `endpoint.txt`. Pass `natural` as the consumer's second
argument. This records a 60-second window with a second 1 Hz HTTP reader.

## Validation on GB200, 2026-09-22

Same node, four GPUs, DCGM 4.6.0, driver 580.126.20; SRT consumer `33fac260`.
Exporter cadence 100 ms, consumer cadence 1 s, timeout 2 s. No model workload.

| Injected five-second repair | Original | Patched |
| --- | ---: | ---: |
| Maximum sample gap | 5.5064 s | 1.0009 s |
| Request timeouts | 2 | 0 |
| Fresh DCGM timestamps per GPU during repair | 0 | 5 |
| Maximum power source age during repair | unavailable | 24.2 ms |
| GPUs with profiling recovered afterward | 4/4 | 4/4 |

Normal-path checks retained all 24 metric families with two concurrent readers.
Splitting fields adds a DCGM read per entity; overhead under model load is unverified.
The default counter file omits SM_ACTIVE; the fault probe checks its recovery.

Native ARM64 build and execution were verified. Docker build, registry publication,
the original multi-node model canary, x86/MIG, and production rollout remain unverified.
See the [full validation record](https://github.com/SemiAnalysisAI/srt-slurm/blob/63c9d2428a1e02e19933b4326b0eb614551825f9/docs/dcgm-profiling-isolation.md#validation-on-gb200-2026-09-22)
for normal-path timings, unit/race results, and baseline environment failures.
