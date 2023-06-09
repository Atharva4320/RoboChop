import argparse
from utils import *
from frankapy import FrankaArm
from perception import CameraIntrinsics
from UDPComms import Subscriber, timeout, Scope
# import warning

# # import camera intrinsics and extrinsics
# REALSENSE_INTRINSICS = "vision_module/2D_vision/calib/realsense_intrinsics.intr"
# REALSENSE_EE_TF = "vision_module/2D_vision/calib/realsense_ee_shifted.tf"
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
# )
# parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
# args = parser.parse_args()

# realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
# realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

# fa = FrankaArm()

sub = Subscriber(5501, timeout=0.2)

while True:
	try: 
		message = sub.get()
		print(message)

		# x = message[0][0]
		# y = message[0][1]
		# z = message[0][2]

		# robot_pose = fa.get_pose()
		# com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform, robot_pose)

		# # --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
		# com = np.array([com[0], com[1], com[2]]) # should be the x,y,z position in robot frame
		# print("\nCOM: ", com)
		# robot_pose.translation = np.array([com[0], com[1], com[2] + 10])
		# # fa.goto_pose(robot_pose)

	except timeout:
		print("no message")
	# message = sub.get()
	# print('message: ', message)
	# try:
	# 	message = sub.get()
	# 	print('message: ', message)

	# except timeout:
	# 	# warning.warn("UDPComms timeout")
	# 	print("UDPComss timeout")
	# 	break