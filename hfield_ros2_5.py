#!/usr/bin/env python3
# exactly the same as hfield_ros2_2.py except I'm trying to optimize speed in processing
    # works fine, but honestly cannot tell the difference in speed
# Minimal ROS2 → MuJoCo live heightfield (single topic, no TF, no fancy filters)
# uses camera z as the height directly, takes the mean

import os, threading, time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import mujoco
from mujoco import viewer

# =========================== CONFIG (keep it simple) ===========================
TOPIC = '/camera/camera/depth/color/points'   # change if your topic is different
# TOPIC = '/topic_1/cam_1/depth/color/points'

# Heightfield grid & world size
NROW, NCOL = 160, 160                   # resolution of the heightfield
Lx, Ly     = 6.0, 6.0                   # meters covered in X and Y (world size in MuJoCo)
Hz, base   = 0.8, 0.05                  # vertical scale (0..1 -> base..base+Hz)

# ROI in incoming pointcloud coordinates (x,y in meters)
# We'll map ROI_x ∈ [-Lx/2,+Lx/2], ROI_y ∈ [-Ly/2,+Ly/2] directly to the grid
ROI_X_HALF = Lx / 2.0
ROI_Y_HALF = Ly / 2.0

# z-to-[0,1] mapping (very simple, fixed bounds) - used in normalizing
Z_MIN = 0.0      # meters → becomes heightfield value 0.0
Z_MAX = 1.0      # meters → becomes heightfield value 1.0
# ==============================================================================


def make_model_xml(nrow=NROW, ncol=NCOL, size=(Lx, Ly, Hz, base)):
    Lx_, Ly_, Hz_, base_ = size
    return f"""
<mujoco model='pc2_live_hfield'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <asset>
    <hfield name='terrain' nrow='{nrow}' ncol='{ncol}' size='{Lx_} {Ly_} {Hz_} {base_}'/>
  </asset>
  <worldbody>
    <light name="toplight" pos="0 0 5" dir="0 0 -1" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" directional="true"/>
    <geom type='hfield' hfield='terrain' rgba='0.7 0.7 0.7 1'/>
    
    <camera name='iso' pos='2.0 -2.5 1.8' euler='35 0 25'/>
  </worldbody>
</mujoco>
"""


def set_heightfield(model, hid, arr): # edits CPU memory. writes numpy grid
    nrow = model.hfield_nrow[hid]
    ncol = model.hfield_ncol[hid]
    assert arr.shape == (nrow, ncol)
    adr = model.hfield_adr[hid]
    model.hfield_data[adr:adr + nrow * ncol] = arr.ravel(order="C")


def upload_heightfield(v, model, hid): # updates teh visualization so you actually see the new surface
    if hasattr(v, "update_hfield"):
        v.update_hfield(hid)
    else:
        try:
            mujoco.mjr_uploadHField(model, hid, v.context)
        except:
            pass


def project_points_to_grid(points_xyz, heights01_out):
    """
    Minimal “bin to grid”:
      - Keep the **max z** for each (row, col).
      - Anything without data falls back to Z_MIN.
    Inputs:
      points_xyz: (N,3) numpy array of [x,y,z]
      heights01_out: (NROW,NCOL) float32 array to overwrite (in-place)
    """
    # Precompute grid mapping constants
    # Map x ∈ [-ROI_X_HALF, +ROI_X_HALF] → col ∈ [0, NCOL-1]
    # Map y ∈ [-ROI_Y_HALF, +ROI_Y_HALF] → row ∈ [0, NROW-1]
    dx = (2 * ROI_X_HALF) / NCOL
    dy = (2 * ROI_Y_HALF) / NROW

    # Start by filling with NaN (not a number) to ater tell which cells never got any points
    grid_z = np.full((NROW, NCOL), np.nan, dtype=np.float32)

    # Filter to ROI bounds (keeps this VERY cheap)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    
    z = -z + 1.0

    # Keep points inside the box and with real z values (not NaN/Inf)
    m = (
        (x >= -ROI_X_HALF) & (x < ROI_X_HALF) &
        (y >= -ROI_Y_HALF) & (y < ROI_Y_HALF) &
        np.isfinite(z)
    )
    if not np.any(m):
        # No valid points → write flat at Z_MIN
        heights01_out[:] = 0.0
        return

    x = x[m]; y = y[m]; z = z[m] # shrink to just the points we kept

    # Compute integer row/col indices (floor)
    cols = ((x + ROI_X_HALF) / dx).astype(np.int32)
    rows = ((y + ROI_Y_HALF) / dy).astype(np.int32)

    # Clip so we don’t ever step outside the grid due to rounding.
    np.clip(cols, 0, NCOL - 1, out=cols)
    np.clip(rows, 0, NROW - 1, out=rows)

    # use Numpy's np.mazimum.at
    lin = rows * NCOL + cols
    sums   = np.bincount(lin, weights=z, minlength=NROW*NCOL).astype(np.float32)
    counts = np.bincount(lin, minlength=NROW*NCOL).astype(np.float32)
    means  = sums / np.maximum(counts, 1.0)  # avoid divide-by-zero

    grid_z = means.reshape(NROW, NCOL)



    # Fill NaNs (cells with no points) with Z_MIN
    np.nan_to_num(grid_z, copy=False, nan=Z_MIN)

    # normalize
    denom = max(1e-6, (Z_MAX - Z_MIN))
    np.subtract(grid_z, Z_MIN, out=grid_z)
    np.divide(grid_z, denom, out=grid_z)
    # np.clip(grid_z, 0.0, 1.0, out=grid_z)

    # Write into output buffer (no reallocation)
    heights01_out[:] = grid_z


class PC2ToHFieldNode(Node):
    def __init__(self, model, data, hid, heights01, lock, viewer_obj):
        super().__init__('pc2_to_hfield_minimal')
        self.subscription = self.create_subscription(
            PointCloud2, TOPIC, self.callback, 10
        )
        self.model = model
        self.data = data
        self.hid = hid
        self.heights01 = heights01
        self.lock = lock
        self.viewer = viewer_obj # The mujoco viewer You’ll trigger a GPU refresh after updating the heightfield.
        self.new_frame = False # flag for 'a fresh height grid is available'

        print(f"[ROS2] Subscribed to {TOPIC}")

    def callback(self, msg: PointCloud2):
        # 1) Pull xyz from pointcloud (skip NaNs)
        #   field names often include x,y,z in RealSense organized clouds
       
        

        gen = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True) # read_points Iterates over the structured binary data inside the ROS PointCloud2 message and yields only good (x,y,z) triples.
        arr = np.fromiter(gen, dtype=[('x','<f4'), ('y','<f4'), ('z','<f4')]) # pulls values out of the iterator and builds an array from them
        n = arr.size
        print(f"[ROS2] Got pointcloud with {n} finite points")

       
        if n == 0:
            print("[ROS2] Empty pointcloud, skipping") 
            return
        pts = np.empty((n, 3), dtype=np.float32)
        pts[:, 0] = arr['x'] # take the x column
        pts[:, 1] = arr['y']
        pts[:, 2] = arr['z']
        
        
        start_time = time.time()

        # 2) Project to grid (edit heights01 in place). thread safety
        with self.lock:
            project_points_to_grid(pts, self.heights01)
            self.new_frame = True
        elapsed_time_ms = (time.time() - start_time) * 1000
        print(f"[ROS2] Elapsed time: {elapsed_time_ms:.2f} ms")


def main():
    # ---------------- MuJoCo setup ----------------
    xml = make_model_xml(nrow=NROW, ncol=NCOL, size=(Lx, Ly, Hz, base))
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    hid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")

    # Shared heightfield buffer
    heights01 = np.zeros((NROW, NCOL), dtype=np.float32)
    lock = threading.Lock()

    # Start viewer
    v = viewer.launch_passive(model, data)
    v.cam.distance = 3.2
    upload_heightfield(v, model, hid)

    # ---------------- ROS2 setup ----------------
    rclpy.init(args=None)
    node = PC2ToHFieldNode(model, data, hid, heights01, lock, v)

    # Spin ROS2 in a background thread (simple)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        # Render loop (just steps physics at an idle rate)
        while v.is_running():
            mujoco.mj_step(model, data)
            if node.new_frame:
                with lock:
                    set_heightfield(model, hid, heights01)
                    # mujoco.mj_forward(model, data)
                    # (We upload here to keep it simple; you could also set a flag and upload in the render loop)
                    upload_heightfield(v, model, hid)
                    node.new_frame = False
            v.sync()
            # Small sleep keeps CPU happy
            time.sleep(0.002)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        v.close()


if __name__ == '__main__':
    main()
