"""Render all four assembled appendix figures with one command."""

from pathlib import Path
import subprocess
import sys
import time

from appendix import parse_args


def main():
    args = parse_args("")
    directory = Path(__file__).resolve().parent
    started = time.perf_counter()
    for script, name in (("compare_expanded.py", "expanded"), ("noise.py", "noise"),
                         ("root_sampling.py", "root_sampling"), ("degree.py", "degree")):
        figure_started = time.perf_counter()
        print(f"Starting {name}", flush=True)
        subprocess.run([sys.executable, str(directory / script), *sys.argv[1:],
                        "--out-dir", str(args.out_dir / name)], check=True)
        print(f"Completed {name} in {time.perf_counter() - figure_started:.1f}s", flush=True)
    print(f"All figures completed in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
