#!/usr/bin/env mjpython
# pip install mujoco glfw numpy
# A set of stairs using the method of MuJoCo API

import numpy as np
import mujoco
from mujoco import viewer

def make_model_xml(nrow=120, ncol=120, size=(3.0, 3.0, 0.6, 0.02)):
    Lx, Ly, Hz, base = size
    return f"""
<mujoco model='stairs_live'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <asset>
    <!-- Blank heightfield: we'll fill hfield_data from Python -->
    <hfield name='terrain' nrow='{nrow}' ncol='{ncol}' size='{Lx} {Ly} {Hz} {base}'/>
  </asset>
  <worldbody>
    <geom type='hfield' hfield='terrain' rgba='0.75 0.75 0.75 1'/>
    <body pos='-1.0 0 0.6'>
      <freejoint/>
      <geom type='sphere' size='0.07' density='2000' rgba='0.9 0.2 0.2 1'/>
    </body>
    <camera name='iso' pos='1.5 -2.0 1.2' euler='35 0 25'/>
  </worldbody>
</mujoco>
"""

def set_heightfield(model, hid, arr): # this arr, which is a 2D NumPy array of shape ends up being heights01
    # since we only have one terrain/heightfield, hid=0 all the time
    nrow = model.hfield_nrow[hid] # hfield_nrow is an array
    ncol = model.hfield_ncol[hid] # hfield_ncol is another array
    assert arr.shape == (nrow, ncol) # makes sure the number of columns and rows are corrent
    adr = model.hfield_adr[hid] # another array
    model.hfield_data[adr:adr + nrow*ncol] = arr.ravel(order="C")

def upload_heightfield(v, model, hid):
    if hasattr(v, "update_hfield"): v.update_hfield(hid)
    else:
        try: mujoco.mjr_uploadHField(model, hid, v.context)
        except: pass

# ---------------- stairs grid ----------------
NROW, NCOL = 120, 120
NUM_STEPS = 6               # how many risers across X
ORIENTATION = "x"           # "x" = steps along X, "y" = along Y

# heights01 must be shape (nrow, ncol), values in [0..1]
heights01 = np.zeros((NROW, NCOL), dtype=np.float32)

if ORIENTATION == "x":
    # split columns into NUM_STEPS bands, each band is a higher plateau
    cuts = np.linspace(0, NCOL, NUM_STEPS + 1, dtype=int)
    for k in range(NUM_STEPS):
        heights01[:, cuts[k]:cuts[k+1]] = k / (NUM_STEPS - 1)
else:
    # steps along Y instead
    cuts = np.linspace(0, NROW, NUM_STEPS + 1, dtype=int)
    for k in range(NUM_STEPS):
        heights01[cuts[k]:cuts[k+1], :] = k / (NUM_STEPS - 1)

# ---------------- build & show ----------------
xml = make_model_xml(nrow=NROW, ncol=NCOL, size=(3.0, 3.0, 0.6, 0.02))
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)
hid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")

# Write the heights and push to GPU
set_heightfield(model, hid, heights01) # i am just confused where heights01 makes an impact because this line is just there. no variable was saved and it was never used again.
mujoco.mj_forward(model, data)

with viewer.launch_passive(model, data) as v:
    v.cam.distance = 3.0
    upload_heightfield(v, model, hid)
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
