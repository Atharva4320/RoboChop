import argparse
import time
from utils import *
from frankapy import FrankaArm
from perception import CameraIntrinsics
from UDPComms import Subscriber, timeout, Scope
import UdpComms as U
# import warning

# import camera intrinsics and extrinsics
REALSENSE_INTRINSICS = "vision_module/2D_vision/calib/realsense_intrinsics_camera4.intr"
REALSENSE_EE_TF = "vision_module/2D_vision/calib/realsense_camera4.tf"
# REALSENSE_INTRINSICS = "vision_module/2D_vision/calib/realsense_intrinsics.intr"
# REALSENSE_EE_TF = "vision_module/2D_vision/calib/realsense_ee_shifted.tf"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
)
parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
args = parser.parse_args()

realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)

# print("\nReset pose...")
# fa = FrankaArm()
# fa.reset_pose()
# fa.reset_joints()
# pose = fa.get_pose()
# pose.translation = np.array([0.5, 0, 0.5])
# print("\nGo to observation pose...")
# fa.goto_pose(pose)

# sub = Subscriber(5502, timeout=0.2)
# updIP: This computer, SendIP: other computer 
udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)

print("entering loop....")
while True:
	try: 
		message = udp.ReadReceivedData()
		if message is None:
			continue
		if message == 'No centroid Detected!':
			continue
		# if message is ''
		# message = sub.get()
		# print(udp.rxThread)
		# udp.ReadUdpThreadFunc()
		# print(message.split(','))
		x = float(message.split(',')[0][2:-1])
		y = float(message.split(',')[1])
		z = float(message.split(',')[2].split(']')[0])
		# print(np.array([x,y,z]))
		# if message == 'No centroid Detected!':
		# 	print(message)
		# else:
		# 	x = message[0][0]
		# 	y = message[0][1]
		# 	z = message[0][2]

		com = get_object_center_point_in_world_realsense_static_camera(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform)
		print("COM: ", com)

		# 	robot_pose = fa.get_pose()
		# 	# com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform, robot_pose)

		# 	# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
		# 	com = np.array([com[0], com[1] - 0.05, com[2]]) # should be the x,y,z position in robot frame
		# 	print("COM: ", com)
		# 	robot_pose.translation = np.array([com[0], com[1], com[2] + 0.4])
		# 	# fa.goto_pose(robot_pose)
		# 	# time.sleep(5)

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