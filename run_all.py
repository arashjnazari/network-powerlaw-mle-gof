"""Execute the network degree-distribution analysis workflow."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "check_methods.py",
    "main.py",
    "xmin_sensitivity.py",
    "compare_models.py",
)
EXPECTED_OUTPUTS = (
    "summary.csv",
    "ba_degrees.csv",
    "er_degrees.csv",
    "ba_ccdf.png",
    "er_ccdf.png",
    "ba_xmin_sensitivity.csv",
    "ba_alpha_vs_xmin.png",
    "ba_ks_vs_xmin.png",
    "ba_tail_size_vs_xmin.png",
    "model_comparison.csv",
    "ba_model_comparison.png",
    "er_model_comparison.png",
)


def run_script(filename: str) -> None:
    """Stream a child script's output and stop the workflow on any failure."""
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    # The scripts save figures to PNG; no GUI event loop is needed.
    environment["MPLBACKEND"] = "Agg"
    process = subprocess.Popen(
        [sys.executable, "-u", "-B", str(ROOT / filename)],
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("Unable to read the child process output.")
        for line in process.stdout:
            print(line, end="", flush=True)
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if return_code != 0:
        raise RuntimeError(
            f"{filename} failed (exit code {return_code}). "
            "Workflow stopped; output may be incomplete."
        )


def main() -> None:
    missing_scripts = [name for name in SCRIPTS if not (ROOT / name).is_file()]
    if missing_scripts:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing_scripts))
    if not sys.executable:
        raise RuntimeError("Cannot locate the current Python interpreter.")

    print(f"Project: {ROOT}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print("Running analysis workflow; existing outputs will be overwritten.", flush=True)
    for index, filename in enumerate(SCRIPTS, start=1):
        print(f"\n[{index}/{len(SCRIPTS)}] {filename}", flush=True)
        print("-" * 60, flush=True)
        run_script(filename)

    output = ROOT / "output"
    missing_outputs = [
        name for name in EXPECTED_OUTPUTS
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError("Missing or empty outputs: " + ", ".join(missing_outputs))

    print("\nALL STEPS COMPLETED SUCCESSFULLY", flush=True)
    print(f"Generated {len(EXPECTED_OUTPUTS)} output files in: {output}", flush=True)
    print("CSV files: 5 | PNG figures: 7", flush=True)
    print("Interpretation limits and reference results are documented in README.md.", flush=True)


if __name__ == "__main__":
    main()
