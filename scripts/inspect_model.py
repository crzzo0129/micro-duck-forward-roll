"""CPU model preview and nominal XML-PD pose audit (not a learned policy)."""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/model_audit"))
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    import mujoco
    import numpy as np
    model_path = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck/scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for trial in range(8):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key)
        data.qpos[7:] += rng.normal(0, 0.005, model.nu)
        for _ in range(round(3 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        tilt = math.degrees(math.acos(float(np.clip(data.xmat[trunk].reshape(3, 3)[2, 2], -1, 1))))
        rows.append({"trial": trial, "height_m": float(data.xpos[trunk, 2]),
                     "tilt_deg": tilt, "finite": bool(np.isfinite(data.qpos).all()),
                     "nominal_pose_stable": tilt < 20 and data.xpos[trunk, 2] > 0.09})
    report = {"model": str(model_path), "nq": model.nq, "nu": model.nu,
              "total_mass_kg": float(model.body_mass.sum()),
              "controller": "nominal XML PD; training uses BAM instead",
              "seconds": 3, "trials": rows}
    (args.output / "pose_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not args.no_render:
        from PIL import Image
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.12]
        camera.distance, camera.azimuth, camera.elevation = 0.65, 135, -15
        # Default framebuffer is 640x480; do not request a larger render target.
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(args.output / "preview.png")
        print(f"Untrained model preview: {args.output / 'preview.png'}")


if __name__ == "__main__":
    main()
