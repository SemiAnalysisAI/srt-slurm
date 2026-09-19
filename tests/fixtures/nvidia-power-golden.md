# NVIDIA artifact refactor guard

`nvidia-power-golden/` is the byte-level reference requested for this vendor-profile
refactor. It was generated from commit `03863ab23804f8a31811d9aaccc54abc8961a41c`
using the unchanged `package` fixture in `tests/test_power_validator.py`. The test
covers the manifest, samples CSV, measurement window, and benchmark result together.
These are synthetic artifacts, not a hardware capture.

Only `srtctl.core.power.manifest.PRODUCER_VERSION` is fixed to `golden-test` because
hatch-vcs otherwise changes it with the checkout. This test does not validate build
version generation. The CSV retains the writer's CRLF bytes; the adjacent
`.gitattributes` prevents checkout conversion and recognizes that line ending.

To reproduce the reference, run this from a checkout of the commit above with the
repository test dependencies installed. It prints a candidate directory outside the
checkout; compare its files before replacing an intentionally changed reference.
The same command on a newer checkout generates a candidate for a reviewed contract
change. Do not normalize or reformat the fixture files.

```bash
uv run python - <<'PY'
import runpy
import tempfile
from pathlib import Path
import srtctl.core.power.manifest as manifest

manifest.PRODUCER_VERSION = "golden-test"
package = runpy.run_path("tests/test_power_validator.py")["package"]
root = Path(tempfile.mkdtemp(prefix="srtctl-nvidia-golden-"))
log_dir, _ = package.__wrapped__(root)()
print(log_dir)
PY
```

Verify the retained reference with:

```bash
uv run pytest tests/test_power_validator.py::test_default_nvidia_bundle_matches_pre_profile_bytes -q
```
