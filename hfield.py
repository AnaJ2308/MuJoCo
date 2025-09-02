#!/usr/bin/env python3
"""
Deterministic 4-step stairs -> pointcloud -> rasterize -> MuJoCo XML with <geom type="hfield">,
WITHOUT writing a PNG. Uses <hfield data="..."> inline instead.
"""

import numpy as np

# ----------------------------
# parameters
# ----------------------------
N_STEPS   = 4
RUN       = 0.25   # tread depth (m)
RISE      = 0.18   # riser height (m)
WIDTH_Y   = 1.00   # stairs width (y)
NROW      = 64     # grid rows (y)
NCOL      = 64     # grid cols (x)
DENSITY   = 120    # samples per surface for point cloud

LENGTH_X   = N_STEPS * RUN
MAX_HEIGHT = N_STEPS * RISE


def generate_stairs_pointcloud(n_steps, run, rise, width, density):
    pts = []
    # treads
    for k in range(n_steps):
        x_vals = np.linspace(k*run, (k+1)*run, density)
        y_vals = np.linspace(0.0, width, density)
        X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
        Z = np.full_like(X, (k+1)*rise)
        pts.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))
    # risers
    for k in range(1, n_steps+1):
        y_vals = np.linspace(0.0, width, density)
        z_vals = np.linspace((k-1)*rise, k*rise, density)
        Y, Z = np.meshgrid(y_vals, z_vals, indexing="xy")
        X = np.full_like(Y, (k-1)*run)
        pts.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))
    return np.concatenate(pts, axis=0)


def rasterize_max(pc, nrow, ncol, bounds):
    xmin, xmax, ymin, ymax = bounds
    ix = np.clip(((pc[:,0]-xmin)/(xmax-xmin)*(ncol-1)).astype(int), 0, ncol-1)
    iy = np.clip(((pc[:,1]-ymin)/(ymax-ymin)*(nrow-1)).astype(int), 0, nrow-1)
    H = np.zeros((nrow, ncol), dtype=np.float32)
    flat = H.ravel()
    np.maximum.at(flat, iy*ncol + ix, pc[:,2])
    return H


def main():
    # 1) make pointcloud
    pc = generate_stairs_pointcloud(N_STEPS, RUN, RISE, WIDTH_Y, DENSITY)

    # 2) rasterize to grid heights (in meters)
    bounds = (0.0, LENGTH_X, 0.0, WIDTH_Y)
    H = rasterize_max(pc, NROW, NCOL, bounds)

    # 3) normalize to [0,1] for mujoco’s <hfield data="">
    H_norm = H / MAX_HEIGHT
    H_norm = np.nan_to_num(H_norm)

    # flatten row-major
    h_data = " ".join(f"{val:.5f}" for val in H_norm.flatten())

    # 4) write XML with inline data
    xml = f"""<mujoco model="stairs_hfield">
  <asset>
    <hfield name="stairs"
            nrow="{NROW}" ncol="{NCOL}"
            size="{LENGTH_X/2:.3f} {WIDTH_Y/2:.3f} {MAX_HEIGHT:.3f} 0.0"
            data="{h_data}"/>
  </asset>

  <worldbody>
    <geom type="hfield" hfield="stairs" rgba="0.7 0.7 0.7 1"/>
    <camera name="iso" pos="{LENGTH_X*0.7:.3f} {-WIDTH_Y*0.8:.3f} {MAX_HEIGHT*1.6:.3f}" euler="35 0 25"/>
  </worldbody>
</mujoco>
"""
    with open("stairs_hfield.xml", "w") as f:
        f.write(xml)
    print("[ok] wrote stairs_hfield.xml")

if __name__ == "__main__":
    main()
