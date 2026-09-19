# Phase 1 native prerequisite validation

Implementation starts from `984180e5b8755aef85e9995048b5a16cb5336bce` in a separate
worktree. The original NVIDIA baseline remains clean. No Slurm allocation, GPU
run, workflow dispatch, push, or publication was performed by these checks.

The locked Python 3.12 source suite on macOS completed with **2453 passed,
2 skipped, 6 integration tests deselected, and 5 failures**. All five failures
were independently reproduced against the untouched baseline on the same host:

- Three `test_sa_bench_http_reuse_flag_reaches_warmup_and_formal` variants fail
  under macOS Bash 3 while resolving the sourced profiling helper.
- `test_probe_cpu_captures_affinity_and_slurm_allocation` assumes Linux's
  `os.sched_getaffinity`, absent on macOS.
- `test_apply_mock_emits_submission_json_and_spawns_worker` assumes a Linux CPU
  affinity count and sees `effective_for_check: null` on macOS.

Commands, using environments provisioned from committed `uv.lock`:

```text
uv sync --frozen --no-editable --python 3.12
PYTHONPATH=src python -m pytest tests/ -q --disable-warnings
ruff check src/srtctl/
ruff format --check src/srtctl/
PYTHONPATH=src python -m srtctl.cli.submit schema-docs --check
git diff --check
```

Ruff, formatting, generated schema checks and diff checks pass. The new native
behavior suite has **26 passing tests**, including real recording child argv,
environment and cwd; strict preparation before scheduler access; atomic duplicate
claim races; a killed submitter followed by recovery in a fresh interpreter;
required-server exit zero; repeated TERM during bounded cleanup; active controller
precedence; comment-scoped cancellation of multiple IDs; and a real relative HF
blob symlink read through the emitted model argument and native mount map.

A noneditable installed wheel was built with
`uv sync --frozen --no-dev --no-editable`; its isolated `python -I -m
srtctl.cli.submit prepare` succeeded using an explicit source profile and full
interpreter/distribution/source verification. Native config, model staging,
profiling and router regression checks passed after the model-path amendment.

These checks qualify CPU behavior only. The consumer still needs PR-linked H100
full-duration curves, real c28 evaluation, Pyxis mount/device/environment checks,
remote step cancellation and writer closure, and any enabled telemetry. The
cluster's filesystem must support the atomic directory claims and fsync durability
used by the intent journal. The six deselected tests use a real AIPerf integration
dependency; they are separate from the consumer's model/hardware acceptance.
