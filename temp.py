from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
fa.reset_pose()
fa.reset_joints()
reset_pose = fa.get_pose()
reset_pose.translation = np.array([0.65, 0, 0.58])
fa.goto_pose(reset_pose)