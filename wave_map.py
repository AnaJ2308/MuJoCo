# wave_heightfield_demo.py
# Create a sine-wave heightfield (looks like ocean swells) and view in MuJoCo.

import os, sys
import numpy as np

# 1) PNG writing
try:
    from PIL import Image
except ImportError:
    print("Missing Pillow. Install with: pip install pillow")
    sys.exit(1)

# 2) MuJoCo
try:
    import mujoco
    import mujoco.viewer
except Exception:
    print("MuJoCo import failed. Install with: pip install mujoco")
    raise

# --------------------------------------------
# Config you can tweak
# --------------------------------------------
H, W = 256, 256          # image resolution (pixels)
SX, SY = 5.0, 5.0        # half-sizes in meters -> world is 10m x 10m
SZ = 0.8                 # vertical scale (meters) = wave amplitude range (peak-to-peak ~ SZ)
BASE = 0.001             # positive base offset to satisfy MuJoCo (>0)
PNG_NAME = "terrain.png"

# Wave parameters (feel free to play with these)
amp = 1.0                # amplitude (in 0..1 before scaling by SZ)
wavelength = 3.0         # wavelength in meters
direction_deg = 30.0     # direction the wave crests are oriented to (degrees)
phase = 0.0              # phase shift (radians). Change this to “move” the wave.

# Optional: add a second wave to make interference patterns
use_second_wave = True
amp2 = 0.6
wavelength2 = 2.0
direction2_deg = -20.0
phase2 = 0.7

# --------------------------------------------
# Build coordinates in meters
# --------------------------------------------
# Grid spans x in [-SX, +SX], y in [-SY, +SY]
xs = np.linspace(-SX, SX, W, dtype=np.float32)
ys = np.linspace(-SY, SY, H, dtype=np.float32)
X, Y = np.meshgrid(xs, ys)   # shapes (H, W)

def wave_height(X, Y, amp, wavelength, direction_deg, phase):
    """Return a sine wave height in 0..1 (we'll normalize later)."""
    # Direction unit vector (where the wave travels)
    theta = np.deg2rad(direction_deg)
    kx = np.cos(theta) / max(wavelength, 1e-6)   # cycles per meter along x
    ky = np.sin(theta) / max(wavelength, 1e-6)   # cycles per meter along y
    # 2π * cycles + phase
    arg = 2.0 * np.pi * (kx * X + ky * Y) + phase
    # Centered sine in [-1,1], then map to [0,1]
    return 0.5 + 0.5 * amp * np.sin(arg)

# Base wave
h = wave_height(X, Y, amp, wavelength, direction_deg, phase)

# Optional second wave (interference pattern)
if use_second_wave:
    h2 = wave_height(X, Y, amp2, wavelength2, direction2_deg, phase2)
    h = np.clip((h + h2) / 2.0, 0.0, 1.0)

# # Slight smoothing (optional – keeps shape but softens aliasing)
# # (You can comment this out if you want the raw sine.)
# def blur_same(img, k=5, iters=1):
#     kern = np.ones((k,), dtype=np.float32) / float(k)
#     out = img.astype(np.float32)
#     for _ in range(iters):
#         out = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, out)
#         out = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, out)
#     return out

# h = blur_same(h, k=5, iters=1)

# # Normalize to [0,1] just in case (SZ & BASE handle real-world meters)
# h -= h.min()
# h /= max(h.max(), 1e-6)

# Save 16-bit PNG (precision)
img16 = (h * 65535.0).astype(np.uint16)
Image.fromarray(img16).save(PNG_NAME)
print(f"Saved {PNG_NAME} with wave pattern ({W}x{H})")

# --------------------------------------------
# Minimal MuJoCo scene using this heightfield
# --------------------------------------------
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
    <geom type="hfield" hfield="terrain" material="mat"/>

    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
  </worldbody>
</mujoco>
"""

# Load & view
if not os.path.exists(PNG_NAME):
    print(f"Error: {PNG_NAME} not found at {os.getcwd()}")
    sys.exit(1)

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
