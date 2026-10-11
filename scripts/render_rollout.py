"""Render saved poses only. Never import MJlab, Torch or Warp in this process."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path


def update_pose(model, data, positions, mujoco):
    """Replay observed positions, without applying controls or stepping dynamics."""
    data.qpos[:] = positions
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args()
    # Must be selected before importing MuJoCo/PyOpenGL.
    os.environ["MUJOCO_GL"] = args.backend
    os.environ["PYOPENGL_PLATFORM"] = args.backend
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    import numpy as np
    import mujoco
    import mediapy as media
    import imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    meta = json.loads((args.trace_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta["mujoco_version"] != mujoco.__version__:
        raise RuntimeError("Render the binary model with the same MuJoCo version that saved it")
    model = mujoco.MjModel.from_binary_path(str(args.trace_dir / "model.mjb"))
    with np.load(args.trace_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        qpos = trajectory["qpos"].copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0 or not np.isfinite(qpos).all():
        raise ValueError("Invalid trajectory shape or non-finite positions")
    camera_cfg = meta["camera"]
    model.vis.global_.offwidth = camera_cfg["width"]
    model.vis.global_.offheight = camera_cfg["height"]
    model.stat.extent = max(4.0, 1.5 * camera_cfg["distance"])
    if not camera_cfg["enable_shadows"]:
        model.light_castshadow[:] = False
    if not camera_cfg["enable_reflections"]:
        model.mat_reflectance[:] = 0.0
    if camera_cfg["fovy"] is not None:
        model.vis.global_.fovy = camera_cfg["fovy"]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = camera_cfg["body_id"]
    camera.distance = camera_cfg["distance"]
    camera.azimuth = camera_cfg["azimuth"]
    camera.elevation = camera_cfg["elevation"]
    camera.lookat[:] = camera_cfg["lookat"]
    data = mujoco.MjData(model)
    print(f"[render] Creating {args.backend} renderer", flush=True)
    renderer = mujoco.Renderer(model, height=camera_cfg["height"], width=camera_cfg["width"])
    frames = []
    try:
        for index, positions in enumerate(qpos):
            update_pose(model, data, positions, mujoco)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            if index == 0:
                print("[render] First frame rendered", flush=True)
    finally:
        renderer.close()
    print(f"[render] Encoding {len(frames)} frames to MP4", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(args.output), frames, fps=meta["fps"])
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("Video encoder produced no output")
    meta.update(video=str(args.output.resolve()), status="video_ready", render_backend=args.backend)
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[render] Video saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
