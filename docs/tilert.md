# TileRT with vLLM prefill

`engine.type: tilert` runs a native vLLM prefill server and native TileRT decode
servers. `frontend.type: tilert-router` runs TileRT's P/D router
(`tilert.pd_vllm.pd_router`). No Dynamo, etcd, or NATS is started.

Examples:

- [`examples/tilert/glm5-disagg.yaml`](https://github.com/SemiAnalysisAI/srt-slurm/blob/main/examples/tilert/glm5-disagg.yaml):
  B200, NIXL transport, stock TileRT and vLLM images prepared by `tilert_setup.sh`.
- [`examples/tilert/glm5-rocm-mooncake-disagg.yaml`](https://github.com/SemiAnalysisAI/srt-slurm/blob/main/examples/tilert/glm5-rocm-mooncake-disagg.yaml):
  MI355X, Mooncake transport, prepared ROCm prefill and decode images, weight
  conversion on first use.

## Images

- `model.container` is the decode image.
- `engine.prefill_container` is the vLLM prefill image. It accepts cluster
  `containers:` aliases like `model.container`.
- `frontend.container_image` optionally selects the router image; otherwise the
  router uses the decode image.

`setup_script: tilert_setup.sh` installs TileRT, NIXL, and the router's HTTP and
tokenizer dependencies into stock images, selected by `TILERT_ROLE`. srtctl sets
`TILERT_ROLE` to `prefill` or `decode` on workers and to `router` on the router.
Omit the setup script when the images already contain TileRT.

## Arguments

srtctl owns ports, the model path, prefill tensor parallelism, the KV transfer
config, and the MTP flags on both roles. Everything else goes in
`roles.prefill.args` (vLLM flags, for example `block-size`,
`gpu-memory-utilization`) and `roles.decode.args` (decode server flags, for
example `num-mtp`). Setting a managed argument there is rejected at load.

| Engine key | Prefill (vLLM) | Decode (TileRT) |
| --- | --- | --- |
| `model_profile` | `kv_connector_extra_config.tilert_model` | `--model` |
| `max_seq_len` | `--max-model-len`, `tilert_max_seq_len` | `--max-seq-len` |
| `prefill_kv_cache_dtype` / `kv_cache_dtype` | `--kv-cache-dtype` | `--kv-cache-dtype` |
| `transport` | `tilert_transport` | `--transport` |
| `with_mtp`, `speculative_tokens` | `--speculative-config` | `--with-mtp` |
| `weights_dir` | | `--model-weights-dir` |

The router passes each decode worker's control port to vLLM per request, so the
prefill connector config carries no decode address. `frontend.args` passes router
flags such as `parser`, `model-path`, and `queue-timeout`; `vllm-url`, `decode`,
`host`, and `port` are managed.

## Weights

TileRT decode reads converted weights from `engine.weights_dir`, a container path
that every decode worker sees (a shared mount such as `/models`). Use a separate
directory per model, revision, TileRT version, and TP size.

Without `engine.weight_converter`, the directory must already hold a complete
conversion. With it, each decode worker takes a lock on the directory, runs
`python -m <module> --model_dir <model> --save_dir <weights_dir> <args>` when
`ready_file` is missing, fails if the converter did not create `ready_file`, and
copies the model's tokenizer and chat template files next to the weights before
the server starts. Conversion needs a local `model.path`.

## Topology and readiness

The router supports one prefill worker and one or more decode workers; each
worker fits on one node. Use `enable_multiple_frontends: false`. Aggregate
workers, Dynamo sidecars, and torch profiling are rejected.

The router starts after every worker answers `/health`. Before the benchmark,
srtctl checks the router's `/health` and each worker's `/health` again. Only the
vLLM prefill worker is scraped for Prometheus metrics; TileRT decode and the
router serve no `/metrics`.
