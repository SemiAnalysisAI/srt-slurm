# Prepared single-job execution

The prepared API is an opt-in path for a consumer that has already selected one
benchmark point. Existing `apply`, dependency Bash templates, and cluster wrappers
remain supported. It starts at NVIDIA revision `984180e5b8755aef85e9995048b5a16cb5336bce`.

Install this revision into a dedicated compute-compatible environment using the
committed dependency lock, for example `uv sync --frozen --no-dev --no-editable`.
The environment and pinned source checkout must be readable on the compute host.
This provisioning step precedes submission; batch startup performs no installation.
Set the explicit profile's `srtctl_root` to that source checkout. Installed package
resources must match its source bytes. Generated package version metadata is
covered by the installed distribution inventory, separately from source identity.

```text
python -I -m srtctl.cli.submit capabilities --json
python -I -m srtctl.cli.submit prepare --recipe point.yaml --profile site.yaml \
  --output /shared/prepared/point --expected-nodes 1 \
  --runtime-python /shared/runtime/bin/python --json
python -I -m srtctl.cli.submit submit-prepared --prepared-dir /shared/prepared/point \
  --intent run-attempt-point --cluster site --journal-dir /shared/intents --json
python -I -m srtctl.cli.submit wait --receipt /shared/intents/site/<token>/receipt.json \
  --timeout 28800 --poll 5 --json
```

All commands emit one JSON object. `prepare` returns `prepared_dir`,
`manifest_sha256`, `resources`, `output_root`, and `capabilities`. It rejects
missing/malformed profiles, duplicate YAML keys, nested sweeps, orchestration flag
overrides, and queued demand that disagrees with native physical placement.
The resulting directory contains resolved `config.yaml`, frozen `profile.yaml`,
`job.slurm`, and a manifest. Source/profile path expansion is frozen before
publication. The full inputs exist before `sbatch` can start the batch script.

The manifest binds interpreter bytes, architecture, installed distribution files,
native source/resources, lockfile, recipe/profile input hashes, and allocation
summary. Compute startup validates these before creating any serving step and
checks actual allocated node cardinality. Mutable installed files or source edits
require preparation of a new bundle. Preserve the qualified environment and
source for every in-flight intent.

## Literal client hook

```yaml
benchmark:
  type: custom
  argv: [/shared/client/bin/python, -m, example.client, --policy, /ix/policy.json]
  cwd: /ix
  env: {EXPLICIT_SETTING: value}
  env_unset: [INHERITED_SIMULATION_SETTING]
  concurrencies: [28]
```

`argv` and the existing shell `command` are mutually exclusive. Arguments remain
literal across the native Slurm wrapper, including JSON, spaces, `$`, backticks,
and empty argument values. The selected container must expose the declared cwd
and executable through its mounts. The hook receives `SRT_ENDPOINT`,
`SRT_FRONTEND_HOST`, `SRT_FRONTEND_PORT`, `SRT_JOB_ID`, `SRT_LOG_DIR` (`/logs`),
`SRT_MODEL_NAME`, and existing logical-worker/metrics endpoint variables. Clients
cannot override or unset the runtime-owned `SRT_*` context.

For local Hugging Face snapshots with relative `../../blobs` shard links, mount
the complete cache root at the same canonical absolute path. Native
`RuntimeContext.worker_model_arg` preserves the snapshot path under that mount,
and vLLM uses it directly. This keeps blob links readable without copying model
weights. A snapshot-only `/model` mount does not preserve links outside it.

## Ownership, observation and recovery

The journal namespace is `(cluster, intent_id, generation=0)`. Atomic directory
claiming precedes `sbatch`; an existing claim never authorizes another allocation.
The CLI strips inherited scheduler option variables, requests `--no-requeue`, and
adds the intent token as the Slurm comment. Receipt states are `claimed`,
`accepted`, and `unknown`; accepted IDs remain available even if secondary
bookkeeping fails. Child stdout is an independent acceptance journal.

Before starting an interruptible submission, call `intent-path --intent ID
--cluster SITE --journal-dir DIR --json` to obtain its `receipt_path`. This pure
query creates nothing and lets a parent recover ownership even when interrupted
before submission stdout arrives. Preserve the recorded `SLURM_CONF` and
`SLURM_CONF_SERVER` context for recovery; changing it is rejected.

Use `reconcile --receipt PATH --json` after an unknown response or process crash.
It reads the independent journals and matching controller/accounting comments;
it never calls `sbatch`. Ambiguous or absent evidence remains unknown. Keep every
reported `accepted_ids` entry for operator resolution. A lost claim before its
receipt was published blocks resubmission and needs journal inspection; deleting
an unknown intent is not a retry procedure.

`cancel --receipt PATH --json` targets only the numeric job in the accepted
receipt. It reports a cancellation request, not completed cleanup. Follow it
with bounded `wait` and inspect the terminal state. Run these commands against
the same site's Slurm controller used to submit; the cluster field is the caller's
site namespace, not automatic cross-cluster routing.

After cancellation use `wait --until-terminal`; a requeue verdict can fail the
workload while the allocation is still active. For an ambiguous receipt with
multiple known IDs, use `cancel-known --receipt PATH --json`, followed by
`wait-known --receipt PATH --timeout 120 --poll 2 --json`. Each ID's comment is
verified independently. `wait-known` returns `state: closed, terminal: true`
only when every owned ID has terminal evidence. A foreign comment is never
authorized for cancellation.

Prepared mode opens scheduler logs under the precreated `output_root/native-logs`
directory, so Slurm does not need a job-ID-specific directory before executing
the batch. The job's `logs/sweep_<id>.log` points to its scheduler log.

Live `scontrol` state takes precedence over accounting. Accounting can establish
completion only after explicit controller absence, with exactly one original
generation record. Requeue, controller errors, missing accounting, or observation
timeout cannot become success. `COMPLETED/0:0` additionally requires matching
runtime completion evidence with successful execution, closed writers and node
restoration. Failed/cancelled runs retain diagnostics and never qualify as a
successful benchmark.

Critical server exit, including exit zero before the client finishes, fails the
workload. Signals request shutdown without acquiring cleanup locks; repeated
signals do not reenter cleanup. The registry uses one monotonic deadline across
all process tiers and reports incomplete cleanup conservatively. Required setup
scripts fail if absent. Host teardown failures remain a separate restoration
verdict and do not replace the original workload failure.

CPU tests cover real recording children and scheduler executables, concurrent
claims, a killed submitter followed by fresh-process reconciliation, and repeated
TERM during cleanup. Actual Pyxis device visibility, remote-step signal delivery,
H100 full curves, real evaluation, cancellation, and telemetry still require the
consumer's PR-linked cluster qualification; a CPU test is not that evidence.
