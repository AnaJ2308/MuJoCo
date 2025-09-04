#!/usr/bin/env python3
# live demo for heightfield with wave flow
# Live-updating heightfield demo for MuJoCo (Python bindings)
# Works with mujoco>=3.0.0. You'll also need GLFW (pip install glfw) on most systems.
#
# Run:
#   python live_heightfield_demo.py
#
# Controls:
#   - ESC or close window to quit.

import time
import numpy as np

import mujoco
from mujoco import viewer


def make_model_xml(nrow=256, ncol=256, size=(5.0, 5.0, 0.2, 0.0)):
    Lx, Ly, Hz, _ = size
    return f'''
<mujoco model="live_demo1">
  <option timestep="0.005"/>
  <asset>
    <!-- Programmatic heightfield: no <data> provided. We'll fill it from Python. -->
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{Lx} {Ly} {Hz} 0.001"/>
  </asset>

  <worldbody>
    <!-- The heightfield geom that uses the asset above. -->
    <geom type="hfield" hfield="terrain" pos="0 0 0" rgba="1.3 1.5 0.7 1"/>

    <!-- A simple free body you can watch interact with the terrain. -->
    <body name="ball" pos="0 0 1.2">
      <freejoint/>
      <geom type="sphere" size="0.08" density="2000" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
'''


def procedural_heights(nrow, ncol, t, speed=0.6):
    # Generate a smooth 0..1 height map that evolves with time t.
    # Combines a few sinusoids with different wavelengths and a slow drift.
    y = np.linspace(0.0, 2.0 * np.pi, nrow, dtype=np.float32)
    x = np.linspace(0.0, 2.0 * np.pi, ncol, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")

    Z = (
        0.6 * np.sin(1.0 * X + speed * t)
        + 0.4 * np.sin(1.7 * Y - 0.7 * speed * t)
        + 0.2 * np.sin(0.8 * (X + Y) + 0.33 * speed * t)
        + 0.05 * (X - np.pi) / np.pi  # mild slope so the ball rolls
    )
# Z is a 2D array of shape (nrow,ncol) Each entry Z[i, j] is the “raw” height value at grid cell (i, j)
    
    zmin = Z.min()
    zmax = Z.max()
    if zmax - zmin < 1e-8:
        return np.zeros_like(Z, dtype=np.float32)
    Z01 = (Z - zmin) / (zmax - zmin)
    return Z01.astype(np.float32, copy=False)


def set_heightfield(model, hfield_id, heights_01):
    # Write a (nrow x ncol) float32 array in 0..1 into model.hfield_data for the given hfield.
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    assert heights_01.shape == (nrow, ncol)
    adr = model.hfield_adr[hfield_id]
    model.hfield_data[adr : adr + nrow * ncol] = heights_01.ravel(order="C")


def upload_heightfield(v, model, hfield_id):
    # Newer python viewer exposes update_hfield; older versions require mjr_uploadHField.
    if hasattr(v, "update_hfield"):
        v.update_hfield(hfield_id)
    else:
        try:
            mujoco.mjr_uploadHField(model, hfield_id, v.context)
        except Exception:
            pass


def main():
    NROW, NCOL = 200, 200
    xml = make_model_xml(nrow=NROW, ncol=NCOL, size=(6.0, 6.0, 0.6, 0.0))

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")

    heights = procedural_heights(NROW, NCOL, t=0.0)
    set_heightfield(model, hfield_id, heights)

    mujoco.mj_forward(model, data)

    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = (0.0, 0.0, 0.4)
        v.cam.azimuth = 90
        v.cam.elevation = -20
        v.cam.distance = 5.5

        upload_heightfield(v, model, hfield_id)

        t0 = time.time()
        substep = 0
        while v.is_running():
            mujoco.mj_step(model, data)

            substep += 1
            if substep % 4 == 0:
                t = time.time() - t0
                heights = procedural_heights(NROW, NCOL, t)
                set_heightfield(model, hfield_id, heights)
                upload_heightfield(v, model, hfield_id)

            v.sync()


if __name__ == "__main__":
    main()
