import importlib, os, sys
import pkgutil
from pathlib import Path
from huggingface_hub import snapshot_download

def resolve_eds_model_path(eds_path: str | None) -> Path:
    """
    Resolve an EDS-PSEUDO model 'artifacts' path.

    Resolution order:
    1) CLI --eds_path: expects a folder that contains 'artifacts/'
    2) Relative to this file: <repo>/hf_cache/artifacts
    3) Download from Hugging Face into <repo>/hf_cache, then use artifacts
    """
    # 1) CLI argument provided
    if eds_path:
        cache_dir = Path(eds_path).expanduser().resolve()
        artifacts = cache_dir / "artifacts"
        if artifacts.exists() and artifacts.is_dir():
            return artifacts
        raise FileNotFoundError(
            f"--eds_path was provided but does not contain an 'artifacts' folder: {artifacts}"
        )

    # 2) Assume relative hf_cache next to the app entrypoint
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "hf_cache"
    artifacts = cache_dir / "artifacts"
    if artifacts.exists() and artifacts.is_dir():
        return artifacts

    # 3) Download model (same approach as test_eds.py)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="AP-HP/eds-pseudo-public",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
    )
    artifacts = cache_dir / "artifacts"
    if not artifacts.exists():
        raise FileNotFoundError(
            "Downloaded eds-pseudo cache, but could not find 'artifacts' folder at "
            f"{artifacts}. Check the Hugging Face snapshot structure."
        )
    return artifacts

def _import_recursive(package_name: str) -> None:
    """Import a package and all its submodules to trigger any registry decorators."""
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        raise RuntimeError(f"Could not import {package_name}: {e}") from e

    if hasattr(package, "__path__"):
        for _, name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
            try:
                importlib.import_module(name)
            except Exception as e:
                # Non-fatal: some optional submodules may fail, but the core registry should be loaded.
                # Keep going to mirror the behavior in test_eds.py.
                print(f"[eds-pseudo] Warning: failed to import {name}: {e}")

def prepare_eds_registry(cache_dir: Path) -> None:
    """Ensure custom eds_pseudo components are registered (per your test_eds.py)."""
    abs_cache = str(cache_dir.resolve())
    if abs_cache not in sys.path:
        sys.path.insert(0, abs_cache)
    _import_recursive("eds_pseudo")