"""Resolve portable analysis roots without touching data or creating directories."""
from pathlib import Path
import json
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FOODPREF_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
RUNS_ROOT = Path(os.environ.get("FOODPREF_RUNS_ROOT", REPO_ROOT / "outputs" / "runs")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("FOODPREF_ANALYSIS_OUTPUT_ROOT", REPO_ROOT / "outputs" / "analyses")).expanduser().resolve()
FSL_ROOT = Path(os.environ.get("FSLDIR", "/usr/local/fsl")).expanduser().resolve()
ATLAS_ROOT = Path(os.environ.get("FOODPREF_ATLAS_ROOT", FSL_ROOT / "data" / "atlases")).expanduser().resolve()

def analysis_root(name):
    return OUTPUT_ROOT / name

def config_root(name):
    return REPO_ROOT / "configs" / "analyses" / name

def config_path(name):
    env_name = "FOODPREF_" + name.upper() + "_CONFIG"
    return Path(os.environ.get(env_name, config_root(name) / "experiment.json")).expanduser().resolve()

def expand_paths(value):
    replacements = {"{repo}": REPO_ROOT, "{data}": DATA_ROOT, "{runs}": RUNS_ROOT,
                    "{outputs}": OUTPUT_ROOT, "{fsl}": FSL_ROOT, "{atlas}": ATLAS_ROOT}
    if isinstance(value, dict):
        return {key: expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_paths(item) for item in value]
    if isinstance(value, str):
        for token, path in replacements.items():
            value = value.replace(token, str(path))
    return value

def read_config(path):
    return expand_paths(json.loads(Path(path).read_text(encoding="utf-8")))

def add_dataset_path_args(parser):
    parser.add_argument("--beta-index-csv", default=None, help="Explicit relocated index; provide all three dataset paths together.")
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--brain-mask-root", default=None)
    parser.add_argument("--path-prefix-from", default="")
    parser.add_argument("--path-prefix-to", default="")


def explicit_dataset_paths(args):
    names = ("beta_index_csv", "models_root", "brain_mask_root")
    values = [getattr(args, name, None) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("Provide --beta-index-csv, --models-root, and --brain-mask-root together.")
    paths = {name: str(Path(value).expanduser().resolve()) for name, value in zip(names, values)}
    for name, value in paths.items():
        if not Path(value).exists():
            raise FileNotFoundError(f"Missing {name}: {value}")
    return {**paths, "path_prefix_from": args.path_prefix_from,
            "path_prefix_to": args.path_prefix_to, "resolution_mode": "explicit_cli_paths"}
