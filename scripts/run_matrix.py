"""Run scene x variant x seed for one benchmark config.

Finished runs (those with a metrics.json) are skipped, so the script can be
stopped and restarted. A failed run is logged to failures.jsonl and the matrix
continues; nothing is overwritten.

    python scripts/run_matrix.py configs/llff3.yaml --variants V0 V2 --scenes fern --seeds 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_command(cfg: dict, variant: str, scene: str, seed: int, out_dir: Path) -> list[str]:
    settings = dict(cfg.get("depth_defaults", {}))
    settings.update(cfg["variants"][variant])
    cmd = [sys.executable, "-m", "cwdepth.train",
           "-s", str(REPO_ROOT / cfg["data_root"] / scene), "-m", str(out_dir),
           "--eval", "--seed", str(seed)]
    for key, value in cfg.get("train_args", {}).items():
        cmd += [f"--{key}", str(value)]
    for key, value in settings.items():
        cmd += [f"--{key}", str(value)]
    if float(settings.get("depth_weight", 0.0)) > 0.0:
        cmd += ["--prior_dir", str(REPO_ROOT / cfg["prior_root"] / scene)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--tag", default=None, help="write under outputs/<benchmark>-<tag> (for tuning runs)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override depth_defaults, e.g. depth_weight=0.1")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        if not value:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        cfg.setdefault("depth_defaults", {})[key] = yaml.safe_load(value)
    variants = args.variants or list(cfg["variants"])
    scenes = args.scenes or cfg["scenes"]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    unknown = [v for v in variants if v not in cfg["variants"]] + [s for s in scenes if s not in cfg["scenes"]]
    if unknown:
        parser.error(f"not in {args.config}: {unknown}")

    name = cfg["benchmark"] + (f"-{args.tag}" if args.tag else "")
    root = REPO_ROOT / "outputs" / name
    jobs = [(v, s, k) for k in seeds for s in scenes for v in variants]
    failed = 0
    for index, (variant, scene, seed) in enumerate(jobs, start=1):
        out_dir = root / variant / scene / f"seed{seed}"
        label = f"[{index}/{len(jobs)}] {name} {variant} {scene} seed{seed}"
        if (out_dir / "metrics.json").exists():
            print(f"{label}: done, skipping")
            continue
        cmd = build_command(cfg, variant, scene, seed, out_dir)
        if args.dry_run:
            print(f"{label}: {' '.join(cmd)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with open(out_dir / "train.log", "w") as log:
            result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - start) / 60
        if result.returncode == 0 and (out_dir / "metrics.json").exists():
            mean = json.loads((out_dir / "metrics.json").read_text())["mean"]
            print(f"{label}: PSNR {mean['psnr']:.3f} SSIM {mean['ssim']:.4f} LPIPS {mean['lpips']:.4f} "
                  f"({minutes:.1f} min)", flush=True)
        else:
            failed += 1
            root.mkdir(parents=True, exist_ok=True)
            with open(root / "failures.jsonl", "a") as f:
                f.write(json.dumps({"variant": variant, "scene": scene, "seed": seed,
                                    "returncode": result.returncode, "log": str(out_dir / "train.log")}) + "\n")
            print(f"{label}: FAILED (exit {result.returncode}), see {out_dir / 'train.log'}", flush=True)
    if failed:
        print(f"{failed} of {len(jobs)} runs failed; see {root / 'failures.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
