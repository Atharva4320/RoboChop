from frankapy import FrankaArm

fa = FrankaArm()
pose = fa.get_pose()
print(pose)
fa.open_gripper()