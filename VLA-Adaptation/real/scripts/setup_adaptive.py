"""Create the adaptive collector environment: vla plus aiofranka's native loop.

Run from the checkout: python -m real.scripts.setup_adaptive
adaptive is a clone of vla (LeRobot 0.3.3 collection stack) with aiofranka 0.9.0
and Numba for the nullspace control law. vla and rsm are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


PACKAGES = ["aiofranka==0.9.0", "numba==0.68.0", "llvmlite==0.50.0"]
# Robot drivers and the array runtime the native loop and LeRobot are built against.
UNCHANGED = {"pylibfranka": "0.21.3", "mujoco": "3.12.0", "numpy": "2.2.6"}


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def env_python(conda, name):
    environments = json.loads(subprocess.check_output([conda, "env", "list", "--json"], text=True))["envs"]
    matches = [Path(path) for path in environments if Path(path).name == name]
    return matches[0] / "bin/python" if len(matches) == 1 else None


def install(name: str, source: str) -> None:
    repository = Path(__file__).resolve().parents[2]
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("Conda must be installed and available on PATH")
    if env_python(conda, name) is None:
        if env_python(conda, source) is None:
            raise RuntimeError(f"Conda environment {source} is missing; create the collection environment first")
        run(conda, "create", "-y", "-n", name, "--clone", source)
    python = env_python(conda, name)
    # Constrain the drivers so pip fails rather than upgrading them.
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as constraints:
        constraints.write("".join(f"{package}=={version}\n" for package, version in UNCHANGED.items()))
        constraints.flush()
        run(python, "-m", "pip", "install", *PACKAGES, "-c", constraints.name)
    run(python, "-m", "pip", "install", "--no-deps", "-e", repository)
    check = ("import importlib.metadata as m, sys, json; import aiofranka.native as n; n.load_native(); "
             "print(json.dumps({p: m.version(p) for p in sys.argv[1:]}))")
    versions = json.loads(subprocess.check_output([python, "-c", check, *UNCHANGED], text=True))
    if versions != UNCHANGED:
        raise RuntimeError(f"Driver versions changed: {versions}; expected {UNCHANGED}")
    print(f"\n{name} is ready: aiofranka's native loop loads, drivers unchanged {versions}.")
    print(f"Run the collector with: conda activate {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="adaptive")
    parser.add_argument("--clone-from", default="vla")
    args = parser.parse_args()
    install(args.name, args.clone_from)


if __name__ == "__main__":
    main()
