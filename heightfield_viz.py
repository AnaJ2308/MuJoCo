import os
import sys
import numpy as np

from PIL import Image
import mujoco
import mujoco.viewer

PNG_NAME = "test_7.png"


xml = f"""
<mujoco model="pc_heightfield">
  <asset>
    
    <hfield name="terrain"
            nrow="100" ncol="100"
            file="{PNG_NAME}"
            size="2.5 2.5 1.0 0.001"/>
    <material name="hfmat" rgba="0.7 0.7 0.7 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5"/>
    <geom type="hfield" hfield="terrain" material="hfmat" pos="0 0 0"/>
    <body name="probe" pos="0 0 0.6"><joint type="free"/><geom type="sphere" size="0.1"/></body>
    <camera name="top" pos="0 0 8" quat="1 0 0 0"/>
  </worldbody>
</mujoco>

"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)