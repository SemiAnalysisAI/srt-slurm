# AMD SMI fixture provenance

Synthetic values in the JSON shape emitted by ROCm 7.2.0. These files are not
a GPU capture. The deliberately non-contiguous indices test identity joins.

- [`list --json`](https://github.com/ROCm/amdsmi/blob/rocm-7.2.0/amdsmi_cli/amdsmi_commands.py): `list()` emits GPU index, UUID, BDF and partition identity.
- [`metric --power --usage --json`](https://github.com/ROCm/amdsmi/blob/rocm-7.2.0/amdsmi_cli/amdsmi_commands.py): `metric_gpu()` emits `power.socket_power` from `amdsmi_get_power_info` and `usage.gfx_activity`.
- [`AMDSMILogger`](https://github.com/ROCm/amdsmi/blob/rocm-7.2.0/amdsmi_cli/amdsmi_logger.py): `list` is an array; `metric` uses `combine_arrays_to_json()` and a `gpu_data` array.
- [AMD API sensor semantics](https://rocmdocs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html#amdsmi-get-power-info): GPU socket power in watts, current socket power on MI300+ or average socket power on older supported cards. This does not establish DCGM board-sensor equivalence or wall/system/UBB power.
