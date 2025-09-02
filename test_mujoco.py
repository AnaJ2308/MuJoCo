import mujoco
import mujoco.viewer

# Load a simple model
xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <geom type="box" size="0.1 0.1 0.1" pos="0 0 0.1" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Launch interactive viewer
mujoco.viewer.launch(model, data)
