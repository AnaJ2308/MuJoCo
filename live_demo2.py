#!/usr/bin/env python3
#live demo, fake data. looks like spiky grass moving around/updating
# Live-updating heightfield from a synthetic "point cloud" dataset (no cameras needed).
# - Every 5 seconds, we fetch a new random point cloud that simulates camera data.
# - We rasterize those points into an (nrow x ncol) grid using MAX z per cell.
# - We normalize 0..1 and write into model.hfield_data.
#
# macOS users: run with `mjpython live_demo2.py`
#
# pip install mujoco glfw numpy

import time
import numpy as np

import mujoco
from mujoco import viewer


# --------------------------
# Scene / model definition
# --------------------------
def make_model_xml(nrow=180, ncol=180, size=(6.0, 6.0, 6.0, 0.02)):
    Lx, Ly, Hz, base = size
    return f'''
<mujoco model="hf_from_pointcloud">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <asset>
    <!-- Blank heightfield: filled from Python. -->
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" size="{Lx} {Ly} {Hz} {base}"/>
  </asset>

  <worldbody>
    <!-- Place heightfield at world origin -->
    <geom type="hfield" hfield="terrain" pos="0 0 0" rgba="0.7 0.85 0.7 1"/>

    <!-- A ball so you can see interaction -->
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.08" density="2000" rgba="0.9 0.2 0.2 1"/>
    </body>
    <camera name="top" pos="0 0 8" quat="1 0 0 0"/>
  </worldbody>
</mujoco>
'''


# --------------------------
# Synthetic point cloud source
# --------------------------
class FakePointCloudSource:
    def __init__(self, xmin, xmax, ymin, ymax, zmin=0.0, zmax=0.5, n_points=80_000, seed=None):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        self.n_points = n_points
        self.rng = np.random.default_rng(seed)

    def get_points(self, t):
        x = self.rng.uniform(self.xmin, self.xmax, self.n_points)
        y = self.rng.uniform(self.ymin, self.ymax, self.n_points)

        # Base random z
        z = self.rng.uniform(self.zmin, self.zmax, self.n_points)

        # Mild time-dependent warp to make successive datasets differ
        z = z + 0.1 * np.sin(0.5 * x + 0.7 * t) * np.cos(0.3 * y - 0.2 * t)
        # Clamp
        z = np.clip(z, self.zmin, self.zmax)
        pts = np.column_stack([x, y, z]).astype(np.float32, copy=False)
        return pts


# --------------------------
# Rasterization: point cloud -> grid
# --------------------------
def rasterize_max(points_xyz, nrow, ncol, bounds, prev_grid=None):
    xmin, xmax, ymin, ymax = bounds
    # Map x,y to integer cell indices
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    ix = ((x - xmin) / (xmax - xmin) * ncol).astype(np.int32)
    iy = ((y - ymin) / (ymax - ymin) * nrow).astype(np.int32)

    # Clip into grid range
    np.clip(ix, 0, ncol - 1, out=ix)
    np.clip(iy, 0, nrow - 1, out=iy)

    # Prepare flat grid with -inf sentinel, then take max at indices
    flat = np.full(nrow * ncol, -np.inf, dtype=np.float32)
    flat_idx = np.ravel_multi_index((iy, ix), dims=(nrow, ncol))

    # Use np.maximum.at for in-place max reduction
    np.maximum.at(flat, flat_idx, z.astype(np.float32, copy=False))

    grid = flat.reshape(nrow, ncol)

    # Replace -inf (cells that received no points) with previous grid if available, else zeros
    if prev_grid is not None:
        mask_empty = ~np.isfinite(grid)
        grid[mask_empty] = prev_grid[mask_empty]
    else:
        grid[~np.isfinite(grid)] = 0.0

    return grid.astype(np.float32, copy=False)


def normalize01(grid):
    gmin = np.min(grid)
    gmax = np.max(grid)
    if not np.isfinite(gmin) or not np.isfinite(gmax) or (gmax - gmin) < 1e-8:
        return np.zeros_like(grid, dtype=np.float32)
    return ((grid - gmin) / (gmax - gmin)).astype(np.float32, copy=False)


# --------------------------
# HField helpers
# --------------------------

def set_heightfield(model, hfield_id, heights_01):
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    assert heights_01.shape == (nrow, ncol), (heights_01.shape, (nrow, ncol))
    adr = model.hfield_adr[hfield_id]
    model.hfield_data[adr : adr + nrow * ncol] = heights_01.ravel(order="C")


def upload_heightfield(v, model, hfield_id):
    # Newer viewer exposes update_hfield; fallback to OpenGL upload if needed.
    if hasattr(v, "update_hfield"):
        v.update_hfield(hfield_id)
    else:
        try:
            mujoco.mjr_uploadHField(model, hfield_id, v.context)
        except Exception:
            pass
FLOOR_FRAC = 0.20   # fraction of hfield range for the flat floor

def map_to_floor(h01, floor_frac=FLOOR_FRAC):
    # remap so 0 = floor_frac, 1 = 1
    return floor_frac + (1.0 - floor_frac) * h01


# --------------------------
# Main loop
# --------------------------
def main():
    # Grid resolution and world span
    NROW, NCOL = 180, 180
    Lx, Ly, Hz, base = 6.0, 6.0, 6.0, 0.02  # half-extents 6m -> total 12m square
    xml = make_model_xml(nrow=NROW, ncol=NCOL, size=(Lx, Ly, Hz, base))

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Find IDs and bounds
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    xmin, xmax = -Lx, Lx
    ymin, ymax = -Ly, Ly
    bounds = (xmin, xmax, ymin, ymax)

    # Synthetic dataset source: z ranges 0..0.45 m (within Hz=0.6 scale for headroom)
    source = FakePointCloudSource(xmin, xmax, ymin, ymax, zmin=0.0, zmax=0.45, n_points=80_000, seed=7)

    # Initialize grid from first dataset
    t0 = time.time()
    points = source.get_points(t=0.0)
    grid = rasterize_max(points, NROW, NCOL, bounds, prev_grid=None)
    heights01 = normalize01(grid)
    heights01 = map_to_floor(heights01)  
    set_heightfield(model, hfield_id, heights01)
    mujoco.mj_forward(model, data)

    FETCH_PERIOD = 2.0  # seconds
    last_fetch = time.time()

    with viewer.launch_passive(model, data) as v:
        # Camera
        v.cam.lookat[:] = (0.0, 0.0, 0.3)
        v.cam.azimuth = 90
        v.cam.elevation = -20
        v.cam.distance = 7.0

        # Ensure GPU sees first frame
        upload_heightfield(v, model, hfield_id)

        while v.is_running():
            mujoco.mj_step(model, data)

            now = time.time()
            # Fetch a new "point cloud" every FETCH_PERIOD seconds
            if (now - last_fetch) >= FETCH_PERIOD:
                t = now - t0
                new_points = source.get_points(t)
                grid = rasterize_max(new_points, NROW, NCOL, bounds, prev_grid=grid)
                heights01 = normalize01(grid)
                heights01 = map_to_floor(heights01) 
                set_heightfield(model, hfield_id, heights01)
                upload_heightfield(v, model, hfield_id)
                last_fetch = now

            v.sync()


if __name__ == "__main__":
    main()
