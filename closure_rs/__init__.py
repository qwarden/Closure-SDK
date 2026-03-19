"""Local import shim for the Rust extension.

Import order:
1. bundled extension next to this package (installed wheel/editable build)
2. repo-local `_vendor/closure_rs/` build (developer workflow)

This prevents stale user-site packages from shadowing the current repository
when Python is launched from the repo root.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys


def _candidate_extensions() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    bundled = sorted(pkg_dir.glob("closure_rs*.so")) + sorted(pkg_dir.glob("closure_rs*.pyd"))
    if bundled:
        return bundled

    vendor_dir = pkg_dir.parent / "_vendor" / "closure_rs"
    vendored = sorted(vendor_dir.glob("closure_rs*.so")) + sorted(vendor_dir.glob("closure_rs*.pyd"))
    return vendored


def _load_extension():
    candidates = _candidate_extensions()
    if not candidates:
        raise ImportError(
            "Could not find a local closure_rs extension. "
            "Build it with: python -m pip install --no-build-isolation --no-deps "
            "--upgrade --target _vendor ."
        )

    ext_path = candidates[0]
    mod_name = "closure_rs.closure_rs"
    spec = util.spec_from_file_location(mod_name, ext_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load closure_rs extension from {ext_path}")

    module = util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


closure_rs = _load_extension()

for _name in dir(closure_rs):
    if not _name.startswith("_"):
        globals()[_name] = getattr(closure_rs, _name)

__doc__ = closure_rs.__doc__
if hasattr(closure_rs, "__all__"):
    __all__ = closure_rs.__all__
