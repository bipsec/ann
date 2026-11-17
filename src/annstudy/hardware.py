"""Hardware and environment snapshot for reproducibility."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True,
                timeout=5,
            )
            return sha if dirty.returncode == 0 else f"{sha}-dirty"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no-repo"


def _get_package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not-installed"


def snapshot() -> dict:
    """Capture a reproducible snapshot of the execution environment."""
    import psutil

    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass

    mem = psutil.virtual_memory()
    physical_cores = psutil.cpu_count(logical=False) or 0
    logical_cores = psutil.cpu_count(logical=True) or 0

    packages = [
        "faiss-cpu", "numpy", "pandas", "pyarrow", "scipy",
        "scikit-learn", "matplotlib", "psutil", "torch",
        "sentence-transformers", "openai", "hnswlib", "tqdm",
    ]
    library_versions = {pkg: _get_package_version(pkg) for pkg in packages}

    snap = {
        "timestamp": datetime.now(UTC).isoformat(),
        "cpu_model": cpu_model,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version,
        "git_commit": _get_git_commit(),
        "hostname": platform.node(),
        "library_versions": library_versions,
    }

    hw_key = f"{cpu_model}|{physical_cores}|{int(mem.total)}"
    snap["hardware_id"] = hashlib.sha256(hw_key.encode()).hexdigest()[:16]

    return snap


def save_snapshot(path: str = "results/hardware.json") -> dict:
    """Write the hardware snapshot to a JSON file and return it."""
    snap = snapshot()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    return snap


def check_ram_minimum(min_gb: int = 64) -> bool:
    """Return True if total RAM >= min_gb. Print a warning if not."""
    import psutil
    total_gb = psutil.virtual_memory().total / (1024**3)
    if total_gb < min_gb:
        print(
            f"WARNING: System has {total_gb:.1f} GB RAM, "
            f"but {min_gb} GB is recommended for full workloads. "
            f"Development and small-N runs are fine."
        )
        return False
    return True


def load_snapshot(path: str = "results/hardware.json") -> dict:
    """Load an existing hardware snapshot."""
    with open(path) as f:
        return json.load(f)


def main() -> None:
    """CLI: write results/hardware.json for this host and check the RAM minimum."""
    import argparse

    parser = argparse.ArgumentParser(description="Snapshot hardware and library versions.")
    parser.add_argument("--output", default="results/hardware.json")
    parser.add_argument("--min-ram-gb", type=int, default=64)
    args = parser.parse_args()

    snap = save_snapshot(args.output)
    print(json.dumps(snap, indent=2))
    check_ram_minimum(args.min_ram_gb)
    print(f"\nWrote {args.output} (hardware_id={snap['hardware_id']})")


if __name__ == "__main__":
    main()


def data_host_line() -> str:
    """Report provenance line naming the host the DATA came from, not the generating machine.

    Reports used to print ``snapshot()`` — whichever machine ran the generator — as "Hardware
    id", so regenerating on another box relabelled VM measurements as that box's. The data host
    is the analysis-scoped id (``ANNSTUDY_ANALYSIS_HARDWARE_ID`` or this host), described from
    ``results/hardware.json`` when that file is for the same host.
    """
    import json
    from pathlib import Path

    from annstudy.coordinate import current_hardware_id

    data_id = current_hardware_id()
    here = snapshot()
    desc = None
    p = Path("results/hardware.json")
    if p.exists():
        rec = json.loads(p.read_text())
        if rec.get("hardware_id") == data_id:
            desc = (f"{rec.get('cpu_model')}, {rec.get('physical_cores')} cores, "
                    f"{rec.get('ram_total_gb')} GB")
    if desc is None and here["hardware_id"] == data_id:
        desc = f"{here['cpu_model']}, {here['physical_cores']} cores, {here['ram_total_gb']} GB"
    line = f"- Data hardware id: `{data_id}`" + (f" ({desc})" if desc else "")
    if here["hardware_id"] != data_id:
        line += f"\n- Generated on: `{here['hardware_id']}` ({here['cpu_model']}; not the data host)"
    return line
