"""Local 3D viewer for every trained model in ``outputs/``.

Serves ``viewer/index.html`` and a small JSON/binary API on 127.0.0.1:

    GET /api/runs                list of runs, grouped by dataset and scene
    GET /api/splat?id=<run id>   the run's Gaussians in the .splat format
    GET /api/cameras?id=<run id> the run's camera poses, with a look-at depth

Run ids are paths relative to ``outputs/`` (for example
``statues12/V2/statue05/seed0``); only ids found by the scan are served.
A .splat file stores per Gaussian position, scale, colour (degree-0 spherical
harmonics, so view-dependent colour is dropped), opacity and rotation in 32
bytes; conversions are cached in ``outputs/viewer_cache/``.

    python -m cwdepth.viewer            # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
CACHE = OUTPUTS / "viewer_cache"
INDEX = REPO_ROOT / "viewer" / "index.html"
SH_C0 = 0.28209479177387814

# How each output folder is described in the viewer; unknown folders keep their name.
GROUPS = {
    "llff3": ("LLFF", "3 training views"),
    "llff-ref": ("LLFF", "reference: all non-test views"),
    "statues12": ("Statues", "12 training views"),
    "statues-ref": ("Statues", "reference: 84 views"),
    "statues-art": ("Statues", "art: all 96 views"),
}
VARIANTS = {"V0": "baseline", "V1": "depth, uniform", "V2": "depth, confidence", "V3": "depth, confident half",
            "V4": "depth, shuffled confidence"}


def find_runs() -> dict[str, dict]:
    """Every run under outputs/ with a saved point cloud, keyed by run id."""
    runs = {}
    for ply in sorted(OUTPUTS.glob("**/point_cloud/iteration_*/point_cloud.ply")):
        run_dir = ply.parents[2]
        if CACHE in run_dir.parents:
            continue
        run_id = run_dir.relative_to(OUTPUTS).as_posix()
        iteration = int(ply.parent.name.split("_")[1])
        if run_id in runs and runs[run_id]["iteration"] > iteration:
            continue
        parts = run_id.split("/")
        group = parts[0]
        scene = next((p for p in parts[1:] if re.fullmatch(r"statue\d+", p) or (REPO_ROOT / "data" / "llff" / p).is_dir()),
                     None)
        if scene is None:
            continue
        sweep = re.fullmatch(r"llff3-w([0-9.]+)", group)
        if group in GROUPS:
            dataset, heading = GROUPS[group]
        elif sweep:
            dataset, heading = "LLFF", f"depth-weight sweep, λd = {sweep.group(1)}"
        else:
            dataset, heading = ("Statues" if scene.startswith("statue") else "LLFF"), group
        rest = [p for p in parts[1:] if p != scene]
        variant = next((p for p in rest if p in VARIANTS), None)
        seed = next((p for p in rest if p.startswith("seed")), None)
        name = " · ".join(x for x in (f"{variant} {VARIANTS[variant]}" if variant else None, seed) if x)
        name = name or ("full resolution, 30k iterations" if group == "statues-art" else "model")
        label = f"{name} · {heading}"
        info = {"id": run_id, "dataset": dataset, "scene": scene, "group": group, "heading": heading,
                "name": name, "label": label,
                "iteration": iteration, "ply": str(ply)}
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            mean = json.loads(metrics.read_text())["mean"]
            info["psnr"] = round(mean["psnr"], 2)
        run_json = run_dir / "run.json"
        if run_json.exists():
            info["gaussians"] = json.loads(run_json.read_text()).get("num_gaussians")
        runs[run_id] = info
    return runs


def to_splat(ply_path: Path) -> bytes:
    """Convert a 3DGS point_cloud.ply to the 32-byte-per-Gaussian .splat layout."""
    v = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    alpha = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    rgba = np.clip(np.concatenate([rgb, alpha[:, None]], axis=1) * 255.0, 0, 255).astype(np.uint8)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-12)
    rot_u8 = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)
    keep = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1)
    # Large, opaque Gaussians first, as in the reference converter.
    order = np.argsort(-(scale.prod(axis=1) * alpha))
    order = order[keep[order]]
    record = np.dtype([("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    out = np.empty(len(order), dtype=record)
    out["xyz"], out["scale"], out["rgba"], out["rot"] = xyz[order], scale[order], rgba[order], rot_u8[order]
    return out.tobytes()


def cached_splat(info: dict) -> Path:
    ply = Path(info["ply"])
    key = hashlib.sha1(f"{ply}:{ply.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    path = CACHE / f"{key}.splat"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(to_splat(ply))
        tmp.replace(path)
    return path


def cameras(info: dict) -> dict:
    """Camera centres and axes from cameras.json, plus the median depth of the Gaussians ahead of each camera."""
    run_dir = OUTPUTS / info["id"]
    cams = json.loads((run_dir / "cameras.json").read_text())
    v = PlyData.read(info["ply"])["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    xyz = xyz[np.random.default_rng(0).choice(len(xyz), size=min(len(xyz), 20000), replace=False)]
    out = []
    for cam in sorted(cams, key=lambda c: c["img_name"]):
        rot = np.asarray(cam["rotation"])          # camera-to-world rotation; columns are the camera axes
        pos = np.asarray(cam["position"])
        forward, down = rot[:, 2], rot[:, 1]
        depth = (xyz - pos) @ forward
        ahead = depth[depth > 0]
        out.append({"name": cam["img_name"], "position": pos.tolist(), "forward": forward.tolist(),
                    "up": (-down).tolist(), "depth": float(np.median(ahead)) if ahead.size else 1.0,
                    "fovy": float(np.degrees(2 * np.arctan(cam["height"] / (2 * cam["fy"]))))})
    up = np.mean([c["up"] for c in out], axis=0)
    return {"cameras": out, "up": (up / np.linalg.norm(up)).tolist()}


class Handler(BaseHTTPRequestHandler):
    runs: dict[str, dict] = {}
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # quiet
        pass

    def send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self.send(HTTPStatus.OK, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                with self.lock:
                    Handler.runs = find_runs()
                public = [{k: v for k, v in r.items() if k != "ply"} for r in Handler.runs.values()]
                self.send(HTTPStatus.OK, json.dumps(public).encode(), "application/json")
            elif url.path in ("/api/splat", "/api/cameras"):
                info = Handler.runs.get(query.get("id", [""])[0])
                if info is None:
                    self.send(HTTPStatus.NOT_FOUND, b"unknown run id", "text/plain")
                elif url.path == "/api/splat":
                    with self.lock:
                        path = cached_splat(info)
                    self.send(HTTPStatus.OK, path.read_bytes(), "application/octet-stream")
                else:
                    self.send(HTTPStatus.OK, json.dumps(cameras(info)).encode(), "application/json")
            else:
                self.send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as err:  # report to the page instead of dropping the connection
            self.send(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(err).__name__}: {err}".encode(), "text/plain")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.runs = find_runs()
    print(f"{len(Handler.runs)} runs; open http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
