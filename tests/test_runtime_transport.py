"""Runtime config transport uses resolved data without evaluating recipe text."""

import base64
import re
from pathlib import Path


def test_shared_filesystem_runtime_config_transport_is_unchanged(monkeypatch) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: default)
    config = SrtConfig(
        name="shared-output-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
    )

    script = submit.generate_minimal_sbatch_script(config, Path("/tmp/not-required-for-shared.yaml"))

    assert "#SBATCH --output=" in script
    assert "/%j/logs/sweep_%j.log" in script
    assert ".srtctl-sweep-%j.log" not in script
    assert "base64 --decode" not in script


def test_embedded_transport_preserves_resolved_yaml_as_inert_data(monkeypatch, tmp_path) -> None:
    from srtctl.cli import submit
    from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig

    settings = {"runtime_config_transport": "embedded"}
    monkeypatch.setattr(submit, "get_srtslurm_setting", lambda key, default=None: settings.get(key, default))
    cluster_config_path = tmp_path / "srtslurm-source.yaml"
    cluster_config_text = "cluster: mi300x-amds\ncontainers:\n  rocm: /images/rocm.sqsh\n"
    cluster_config_path.write_text(cluster_config_text)
    monkeypatch.setattr(submit, "find_cluster_config_path", lambda: cluster_config_path)
    config = SrtConfig(
        name="node-local-output-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp16"),
        resources=ResourceConfig(gpu_type="mi300x", gpus_per_node=1, agg_nodes=1),
    )
    source_text = "name: source\nnote: original\n"
    runtime_text = 'name: resolved\nnote: "\'; touch /tmp/must-not-run; #"\n'
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(runtime_text)

    script = submit.generate_minimal_sbatch_script(
        config,
        config_path,
        runtime_config_filename="config_variant.yaml",
        runtime_config_text=runtime_text,
        source_config_text=source_text,
    )

    assert "#SBATCH --output=" in script
    assert "/.srtctl-sweep-%j.log" in script
    assert 'mv "${BOOTSTRAP_LOG}" "${LOG_DIR}/sweep_${SLURM_JOB_ID}.log"' in script
    assert '--ntasks-per-node=1 mkdir -p "${LOG_DIR}"' in script
    assert "#SBATCH --chdir=" in script
    assert 'export SRTSLURM_CONFIG="${OUTPUT_DIR}/srtslurm.yaml"' in script
    assert "touch /tmp/must-not-run" not in script

    embedded = re.findall(
        r'RUNTIME_CONFIG="\$\{OUTPUT_DIR\}/([^\"]+)".*?printf \'%s\' \'([^\']+)\' \| base64 --decode',
        script,
        flags=re.DOTALL,
    )
    decoded = {filename: base64.b64decode(payload).decode() for filename, payload in embedded}
    assert decoded == {
        "config.yaml": source_text,
        "config_variant.yaml": runtime_text,
        "srtslurm.yaml": cluster_config_text,
    }
