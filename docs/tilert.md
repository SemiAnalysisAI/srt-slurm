# TileRT with vLLM prefill

`engine.type: tilert` pairs a native vLLM prefill server with native TileRT decode
servers. `frontend.type: tilert-router` runs TileRT's own P/D router. It does not
start Dynamo, etcd, or NATS. The adapter implements `BackendProtocol`; the router
extends `StaticRouterFrontend` and uses its worker discovery, Slurm launch, and
process cleanup.

See [`examples/tilert/glm5-disagg.yaml`](https://github.com/SemiAnalysisAI/srt-slurm/blob/main/examples/tilert/glm5-disagg.yaml)
for a schema 2 recipe. This is a launch example, not a claim of a hardware-validated
benchmark result. The native interfaces follow
[TileRT's P/D implementation](https://github.com/tile-ai/TileRT/tree/main/tilert/pd_vllm).

## Images and setup

- `model.container` selects the TileRT decode image.
- `engine.prefill_container` selects the vLLM prefill image. It accepts the same
  cluster `containers:` aliases as `model.container`.
- `frontend.container_image` optionally selects a router image, otherwise it
  uses the decode image.

The example uses `ghcr.io/tile-ai/tilert:0.1.5` and
`vllm/vllm-openai:v0.26.0`. `setup_script: tilert_setup.sh` installs
TileRT `0.1.5.post3` and NIXL `1.3.1`; it adds the HTTP dependencies to decode and
router containers and Transformers 5 to the router for GLM-5.1 tokenizers.
The prefill install uses `--no-deps` for TileRT to preserve vLLM's dependencies.
The script accepts `TILERT_VERSION`, `TILERT_NIXL_VERSION`,
`TILERT_FASTAPI_VERSION`, `TILERT_UVICORN_VERSION`, and
`TILERT_TRANSFORMERS_VERSION` overrides. Omit setup when using prepared images.
The adapter supplies `TILERT_ROLE` to each worker and to the router setup.

Use `roles.prefill.args` / `roles.decode.args` and `roles.<role>.env` for normal
engine arguments and environment. srtctl owns ports, model paths, prefill TP,
KV transfer configuration, and MTP flags. Attempts to override those arguments
are rejected. `engine.served_model_name` overrides the OpenAI model name;
otherwise the final component of `model.path` is used.

## Weight conversion and mounts

TileRT requires converted weights. Before workers launch, the backend requests
one GPU on the first decode worker's node and runs TileRT's weight converter in
the decode image. The step is tracked in the normal process registry, including
signal cleanup, failure propagation, and a six-hour timeout. Its log is
`tilert_weight_conversion.out`.

Set `engine.weights_dir` to a dedicated directory under a shared writable mount,
visible at the same container path to decode workers **and the router**. For
example, add `/shared/tilert:/tilert_weights` to the cluster's mounts. Use a
separate directory for every model, revision, and converter version; the adapter
does not infer cache identity from model contents. Do not point it at source
weights or a directory with unrelated files.

Conversion uses a filesystem lock shared by concurrent jobs, writes to a sibling
temporary directory, copies tokenizer metadata and the chat template, and
publishes only after checking the index and every referenced shard. Complete
caches are reused. A cache with missing shards or tokenizer metadata is rebuilt.
Failed conversions leave the previous target untouched. The router loads its
tokenizer from this prepared directory, including for local model paths.
Hugging Face sources are completed through `snapshot_download` before conversion;
local sources use the worker's mounted model path.

## Topology, readiness, and metrics

The native router supports exactly one prefill worker and one or more decode
workers. Every worker must fit on one node. The GLM-5 decode kernels require
exactly eight GPUs per decode worker. Use one frontend
(`enable_multiple_frontends: false`); aggregate mode, Dynamo sidecars, and Torch
profiling are rejected. Nsys profiling remains available.

The backend configures the TileRT NIXL connector on vLLM, derives TP from the
allocated prefill GPUs, and supplies matching context length and transport to
both roles. `engine.with_mtp` controls MTP on both sides. Choose compatible
`kv_cache_dtype` and `prefill_kv_cache_dtype` values for the model profile.

The router is started after every worker `/health` endpoint is ready. Before
benchmark traffic, srtctl checks the router's `/health` plus all worker endpoints;
a healthy router alone does not prove the workers are ready. AIPerf and Tachometer receive
only vLLM prefill Prometheus URLs because TileRT decode does not expose `/metrics`.

## Backend extension points

TileRT uses two optional backend hooks alongside the required `BackendProtocol`
methods:

- `get_container_image_for_mode(mode, default)` selects a role image for worker
  and preparation launches. Backends without it keep `model.container`.
- `get_preparation(runtime, processes)` returns a `BackendPreparation` describing
  a finite pre-worker command, selected node/mode, log, timeout, and optional GPU
  request. The orchestrator runs it after HF download/model staging and before
  workers, using that role's image, setup, mounts, and environment.

`get_metrics_port(process)` optionally limits AIPerf discovery to real metrics
endpoints for both AIPerf and Tachometer. Returning `None` omits that process from metrics, without changing
its serving or profiling endpoint. Backends without it retain existing discovery.
A frontend may set `has_metrics = False` to omit its own Tachometer target.
