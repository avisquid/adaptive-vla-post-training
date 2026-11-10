"""Install stock BC and robot drivers in the pinned rsm deployment environment.

Run from the checkout: python -m real.scripts.setup_bc
The existing vla environment is never modified. aiofranka stays at version 0.4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
HANDOFF_REVISION = "a373756093472e145072bb9e3cfef3151022e0d8"
PATCH_SHA256 = "33b37d559c31823fc350fb15e4216a062273550fd580f890eaf1626063a8fec4"
CHECKPOINT_REPOSITORY = "rli14/rsm_cooking_v5_checkpoints"
CHECKPOINT_REVISION = "2f439458961853ecb03facf9f11ae1b13e97c685"
CHECKPOINT_NAME = "pi05_bc_rsm_real_stock_s1_29999"
PYTHON_VERSION = "3.11.13"
UV_VERSION = "0.11.15"


def run(*args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def output(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()


def apply_once(source: Path, patch: Path) -> None:
    applied = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not applied:
        run("git", "-C", source, "apply", "--check", patch)
        run("git", "-C", source, "apply", patch)


def install(root: Path, *, download_checkpoint: bool) -> None:
    repository = Path(__file__).resolve().parents[2]
    requirements = repository / "requirements/bc-inference.txt"
    build_requirements = repository / "requirements/bc-build.txt"
    if not requirements.is_file():
        raise RuntimeError("Run setup from an editable checkout containing requirements/bc-inference.txt")
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("Conda must be installed and available on PATH")
    environments = json.loads(output(conda, "env", "list", "--json"))["envs"]
    matches = [Path(path) for path in environments if Path(path).name == "rsm"]
    if not matches:
        run(conda, "create", "-y", "-n", "rsm", f"python={PYTHON_VERSION}", "pip")
        environments = json.loads(output(conda, "env", "list", "--json"))["envs"]
        matches = [Path(path) for path in environments if Path(path).name == "rsm"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rsm environment, found {matches}")
    python = matches[0] / "bin/python"
    version = output(python, "-c", "import platform; print(platform.python_version())")
    if version != PYTHON_VERSION:
        raise RuntimeError(f"rsm uses Python {version}; this installation requires {PYTHON_VERSION}")

    root.mkdir(parents=True, exist_ok=True)
    source = root / "openpi"
    if not source.exists():
        run("git", "clone", "https://github.com/Physical-Intelligence/openpi.git", source)
        run("git", "-C", source, "checkout", "--detach", OPENPI_REVISION)
    if output("git", "-C", source, "rev-parse", "HEAD") != OPENPI_REVISION:
        raise RuntimeError(f"{source} is not at the pinned OpenPI revision; refusing to overwrite it")
    patch = root / "cooking-inference.patch"
    # Use the checkout's authenticated remote: this repository can be private.
    present = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", HANDOFF_REVISION],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not present:
        run("git", "-C", repository, "fetch", "origin", HANDOFF_REVISION)
    payload = subprocess.check_output([
        "git", "-C", str(repository), "show",
        f"{HANDOFF_REVISION}:rise_stack/openpi/rise_on_pi05.patch",
    ])
    if hashlib.sha256(payload).hexdigest() != PATCH_SHA256:
        raise RuntimeError("Published cooking patch does not match its pinned SHA256")
    patch.write_bytes(payload)
    apply_once(source, patch)
    hardware_patch = repository / "requirements/openpi-real.patch"
    apply_once(source, hardware_patch)
    # PR #13's additive TrainConfigs for the 50-action fine-tunes. The original
    # cooking configs, model implementation, and checkpoint transforms stay intact.
    ft50_patch = repository / "requirements/openpi-ft50.patch"
    apply_once(source, ft50_patch)

    # Use uv only as a pip installer targeting this explicit conda interpreter.
    # Do not run uv sync here: that creates another environment and includes
    # training extras. All inference dependency versions come from the lock.
    run(python, "-m", "pip", "install", f"uv=={UV_VERSION}")
    installer = [python, "-m", "uv", "pip", "install", "--python", python]
    # ALOHA simulation requires MuJoCo <3; aiofranka 0.4 requires >=3.10.
    # The packaging-only patch makes ALOHA optional, without changing model code.
    run(python, "-m", "uv", "pip", "uninstall", "--python", python, "gym-aloha")
    run(*installer, "-r", requirements, "--build-constraints", build_requirements)
    run(*installer, "--no-deps", "-e", source, "-e", source / "packages/openpi-client", "-e", repository)
    run(python, "-m", "pip", "check")
    if download_checkpoint:
        script = (
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
            "allow_patterns=[sys.argv[3] + '/**'], local_dir=sys.argv[4], token=False)"
        )
        run(python, "-c", script, CHECKPOINT_REPOSITORY, CHECKPOINT_REVISION,
            CHECKPOINT_NAME, root / "checkpoints")
    manifest = {
        "python": str(python), "python_version": version, "uv_version": UV_VERSION,
        "openpi_source": str(source), "openpi_revision": OPENPI_REVISION,
        "handoff_revision": HANDOFF_REVISION, "patch_sha256": PATCH_SHA256,
        "hardware_patch_sha256": hashlib.sha256(hardware_patch.read_bytes()).hexdigest(),
        "ft50_config_patch_sha256": hashlib.sha256(ft50_patch.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "build_requirements_sha256": hashlib.sha256(build_requirements.read_bytes()).hexdigest(),
        "checkpoint_repository": CHECKPOINT_REPOSITORY, "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint": str(root / "checkpoints" / CHECKPOINT_NAME),
    }
    (root / "installation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nBC deployment installed. Activate conda environment rsm and use:")
    print(f"  rollout.checkpoint={root / 'checkpoints' / CHECKPOINT_NAME}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".local/share/rsm-real")
    parser.add_argument("--no-checkpoint", action="store_true", help="Install software only")
    args = parser.parse_args()
    install(args.root.expanduser().resolve(), download_checkpoint=not args.no_checkpoint)


if __name__ == "__main__":
    main()
