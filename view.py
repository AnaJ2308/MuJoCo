import mujoco
from mujoco import viewer

m = mujoco.MjModel.from_xml_path("stairs_hfield.xml")
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
