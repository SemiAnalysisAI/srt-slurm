# TileRT decode

Use `roles.prefill.engine: vllm`, `roles.decode.engine: tilert`, and
`frontend.type: tilert-router`. Do not set a top-level `engine` alongside role engines.
Set `frontend.enable_multiple_frontends: false`.
Recipes:

- [B200 / NIXL](../examples/tilert/glm5-disagg.yaml)
- [MI355X / Mooncake](../examples/tilert/glm5-rocm-mooncake-disagg.yaml)

The supported pairing is **vLLM prefill + TileRT decode + TileRT router**.
Configuration loading rejects other engine/router combinations before Slurm
submission. Other prefill engines need a TileRT protocol integration and an
update to the frontend adapter's validation; using NIXL or Mooncake alone is not
enough. Set the model profile, sequence limit, KV layout, and transport consistently
in vLLM's `TileRTConnector` and the decode arguments. The configuration checks do
not verify the connector installed inside the image or prove KV-layout compatibility.

## Images and weights

Use images containing the engine and required connector. Set
`roles.prefill.container` and `roles.decode.container` for the worker images.
`model.container` supplies the shared image for non-worker tasks.
The router image defaults to `model.container`; use `frontend.container_image`
to override it. It must contain `tilert.pd_vllm.pd_router` and its dependencies.
The B200 recipe's image aliases must be defined in `srtslurm.yaml`.

Convert weights with TileRT before serving and mount them at
`roles.decode.args.model-weights-dir`. Set `frontend.args.model-path` to the
tokenizer path or Hugging Face model ID, including when `parser: none`.

## Limits

The router supports one prefill worker and one or more decode workers, each
decode worker on a single node. TileRT requires transferred KV state and cannot
serve aggregate requests through this adapter.

srtctl sets the decode HTTP/control ports and `--engine tilert`; other options
come from `roles.decode.args`. Workers must pass `/health` before the router
starts. TileRT decode has no `/metrics`, so only prefill metrics are scraped.

The [role-engine restrictions](config-reference.md#roles) also apply.
