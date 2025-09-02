##
#
# Use Mujoco's Viewer to show the model.
#
# Note: need mujoco python
##

import mujoco
from mujoco.viewer import launch

# Load the model from XML
xml_file = "./models/g1/g1_12dof.xml"
# xml_file = "./models/go2/go2.xml"

# load and launch the model
model =  mujoco.MjModel.from_xml_path(xml_file)
launch(model)
