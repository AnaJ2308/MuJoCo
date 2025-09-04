# Creates a random heightfield PNG and visualizes it in MuJoCo.

import os
import sys
import numpy as np

# 1) Make sure we can save a PNG
try:
    from PIL import Image
except ImportError:
    print("\nMissing dependency: Pillow\nInstall it with:\n  pip install pillow\n")
    sys.exit(1)

# 2) MuJoCo + viewer
try:
    import mujoco
    import mujoco.viewer
except Exception as e:
    print("\nMuJoCo import failed. Make sure you installed it:\n  pip install mujoco\n")
    raise

# -----------------------------
# Settings you can tweak safely
# -----------------------------
H, W = 256, 256          # heightfield resolution (pixels)
SX, SY = 5.0, 5.0        # half-sizes in X,Y (meters) -> world spans [-SX,+SX] x [-SY,+SY]
SZ = 1.0                 # vertical scale (meters) -> pixel 1.0 => base + SZ
BASE = 0.001             # vertical offset (meters) (keep > 0)
PNG_NAME = "mj_hfield.png" # saved heightmap file

# -----------------------------
# Make a smooth random heightmap
# -----------------------------
rng = np.random.default_rng(0) # random number generator where 0 is the starting point. Don't leave the () empty or else you will get
# a different number every time

def blur_same(img: np.ndarray, k: int = 9, iters: int = 1) -> np.ndarray:
    # k: how wide the blur window is
    """Box blur that keeps the same shape (H,W)."""
    kern = np.ones((k,), dtype=np.float32) / float(k) # make a kernel
    out = img.astype(np.float32)
    for _ in range(iters):
        # horizontal
        out = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, out)
        # vertical
        out = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, out)
    return out

# multi-octave noise (a few blurred random layers added together)
h = np.zeros((H, W), dtype=np.float32)
for weight, k in [(1.00, 21), (0.50, 15), (0.25, 9), (0.12, 5)]:
    h += weight * blur_same(rng.random((H, W), dtype=np.float32), k=k, iters=1)

# normalize to 0..1 and soften extremes (optional)
h -= h.min()
h /= max(h.max(), 1e-6)
h = np.clip(h**1.2, 0.0, 1.0)

# # save 16-bit grayscale PNG for precision
# img16 = (h * 65535.0).astype(np.uint16)
# Image.fromarray(img16).save(PNG_NAME)

# sanity print
print(f"Saved {PNG_NAME} ({W}x{H})")
print(f"World size: {(2*SX):.2f}m x {(2*SY):.2f}m; height range: {SZ:.2f}m; base: {BASE:.3f}m")

# -----------------------------
# Build minimal MuJoCo scene
# -----------------------------
# NOTE: nrow = image HEIGHT (H), ncol = image WIDTH (W). hello. okay i am just testing my commits
xml = f"""
<mujoco>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.002"/>

  <asset>
    <hfield name="terrain" file="{PNG_NAME}"
            nrow="{H}" ncol="{W}"
            size="{SX} {SY} {SZ} {BASE}"/>
    <material name="mat" rgba="0.8 0.8 0.8 1"/>
  </asset>

  <worldbody>
    <!-- The terrain surface -->
    <geom type="hfield" hfield="terrain" material="mat"/>

    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
  </worldbody>
</mujoco>
"""

# -----------------------------
# Load & view
# -----------------------------
# Helpful: ensure the PNG is in the same folder where you run this script.
if not os.path.exists(PNG_NAME):
    print(f"\nError: {PNG_NAME} not found in {os.getcwd()}\n")
    sys.exit(1)

try:
    model = mujoco.MjModel.from_xml_string(xml)
except Exception as e:
    print("\nFailed to load MuJoCo model. Common causes:\n"
          " - PNG path wrong (use absolute path in file=\"/full/path/terrain.png\")\n"
          " - size values must all be > 0 (SX,SY,SZ,BASE)\n"
          " - nrow/ncol must match PNG (H,W)\n")
    raise

data = mujoco.MjData(model)

# If you get a window/back-end error, install glfw:  pip install glfw
mujoco.viewer.launch(model, data)
