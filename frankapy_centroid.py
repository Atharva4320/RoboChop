import argparse
import time
from utils import *
from frankapy import FrankaArm
from perception import CameraIntrinsics
from UDPComms import Subscriber, timeout, Scope
import UdpComms as U
import ast
import re
from numpy import array
from skill_utils import *

# updIP: This computer, SendIP: other computer 
udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)

print("\nReset pose...")
fa = FrankaArm()
fa.reset_pose()
fa.reset_joints()
reset_pose = fa.get_pose() # [0.65, 0, 0.4]
reset_pose.translation = np.array([0.625, 0, 0.45]) # x was 0.55 before

skills = SkillUtils(fa)
obs_objects, obj_dict = skills.observe_scene(udp, reset_pose)

print("entering loop....")
# n_pieces = 4 # parameter to set to dictate number of target pieces
n_pieces = input("Target Number of Pieces: ")
n_pieces = int(n_pieces)

"""
Inputs: "5 pieces of apple, 2 pieces of cucumber, ...."
- should be able to track target pieces for each fruit type
- keep track of count and n_pieces as dictionary with all the different classes and their asssociated target pieces
"""

count = obs_objects
prev_cut_pos = np.array([0.65, 0, 0.4])
while n_pieces > obs_objects:

	# check if the expected count doesn't match the observed objects
	if count != obs_objects:
		if count > obs_objects:
			skills.disturb_scene()
			obs_objects, obj_dict = skills.observe_scene(udp, reset_pose)

		else:
			print("ERROR: we are incorrectly observing more objects than expected")
			skills.disturb_scene()
			obs_objects, obj_dict = skills.observe_scene(udp, reset_pose)

	# go to cut 
	else:
		# get cut centroid
		com, angle = skills.plan_cut(obj_dict)
		collisions = skills.check_cut_collisions(com, obj_dict, angle)
		# check for collisions
		while len(collisions > 0):
			for idx in collisions:
				push_obj_com = obj_dict[idx][0]
				print("\nPush obj com -- should just be x,y: ", push_obj_com)
				skills.push(com, push_obj_com)
			obs_objects, obj_dict = skills.observe_scene(udp, reset_pose)
			collisions = skills.check_cut_collisions(com, obj_dict, angle)
		# when no collisions, execute cut action
		count = skills.cut(count, com, angle)
		obs_objects, obj_dict = skills.observe_scene(udp, reset_pose)

print("Completing cutting task!")






# # --------------------- PREVIOUS VERSION THAT WAS WORKING (JUST IN CASE) ---------------------------
# import camera intrinsics and extrinsics
# REALSENSE_INTRINSICS = "vision_module/2D_vision/calib/realsense_intrinsics_camera4.intr"
# REALSENSE_EE_TF = "vision_module/2D_vision/calib/realsense_camera4.tf"
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

# # updIP: This computer, SendIP: other computer 
# udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)
# print("\nReset pose...")
# fa = FrankaArm()
# fa.reset_pose()
# fa.reset_joints()
# reset_pose = fa.get_pose()
# reset_pose.translation = np.array([0.65, 0, 0.4]) # x was 0.55 before
# print("\nGo to observation pose...")
# fa.goto_pose(reset_pose)
# udp.SendData("Segment")
# # TODO: RIGHT BEFORE OBSERVATION

# print("entering loop....")
# n_pieces = 4 # parameter to set to dictate number of target pieces
# count = 0
# prev_cut_pos = np.array([0.65, 0, 0.4])
# obs_objects = 0
# while n_pieces > obs_objects:
# 	try: 
# 		# sleep_time = 3*count
# 		# time.sleep(sleep_time) # TODO: scale sleep based on count (larger when count larger)
# 		message = udp.ReadReceivedData()
# 		if message is None:
# 			continue

# 		print("Message: ", message)

# 		# parse text-based list of arrays to actual list of arrays
# 		val = re.findall(r"\((.*?)\)", message)
# 		val = list(map(ast.literal_eval, val))
# 		objs = list(map(array, val))

# 		obs_objects = len(objs)
# 		if count == 0:
# 			count = obs_objects

# 		obj_dict = {}

# 		for i in range(len(objs)):
# 			obj = objs[i]
# 			x = obj[0]
# 			y = obj[1]
# 			z = obj[2]
# 			area = obj[3]

# 			robot_pose = fa.get_pose()
# 			com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform, robot_pose)

# 			# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
# 			com = np.array([com[0], com[1] + 0.04, com[2] + 0.02]) # should be the x,y,z position in robot frame
# 			print("COM: ", com)
# 			obj_dict[i] = (com, area)

# 		# check if the expected count doesn't match the observed objects
# 		if count != obs_objects:
# 			if count > obs_objects:
# 				# goto previous slice position and disturb the scene slightly
# 				robot_pose.translation = np.array([prev_cut_pos[0], prev_cut_pos[1], prev_cut_pos[2] + 0.03])
# 				fa.goto_pose(robot_pose)

# 				# TODO: update this to not include the 0.10 offset!

# 				# disturb the scene --> currently assuming cutting blade fixed, so only move in x-direction
# 				robot_pose.translation = np.array([prev_cut_pos[0] - 0.03, prev_cut_pos[1], prev_cut_pos[2] + 0.03])
# 				fa.goto_pose(robot_pose)
# 				robot_pose.translation = np.array([prev_cut_pos[0] + 0.03, prev_cut_pos[1], prev_cut_pos[2] + 0.03])
# 				fa.goto_pose(robot_pose)
				
# 				print("\nGo to observation pose...")
# 				fa.goto_pose(reset_pose)
# 				udp.SendData("Segment")
# 				# TODO: RIGHT BEFORE OBSERVATION

# 			else:
# 				# TODO: disturb scene here too
# 				print("ERROR: we are incorrectly observing more objects than expected")
		
# 		# check if observed correct number of objects
# 		elif n_pieces == obs_objects:
# 			# TODO: disturb scene
# 			print("Completing cutting task!")

# 		# go to cut 
# 		else:
# 			# return the com with the largest area and hover above that object
# 			idx = get_largest_area_idx(obj_dict)
# 			com = obj_dict[idx][0]
# 			prev_cut_pos = com
# 			robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])
# 			fa.goto_pose(robot_pose)
# 			time.sleep(0.5)

# 			# Executing cutting action
# 			print("\nCutting...")
# 			fa.goto_gripper(0, block=False)
# 			fa.apply_effector_forces_along_axis(1.0, 0.5, 0.06, forces=[0.,0.,-75.])
# 			time.sleep(1)
# 			print("\nGo to observation pose...")
# 			fa.goto_pose(reset_pose)
# 			count += 1
# 			udp.SendData("Segment")
# 			# TODO: RIGHT BEFORE OBSERVATION

		# # ------- TEST FOR WHEN COM PREDICTION IS OFF ---------
		# idx = get_largest_area_idx(obj_dict)
		# com = obj_dict[idx][0]
		# prev_cut_pos = com
		# robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		# fa.goto_pose(robot_pose)
		# time.sleep(0.5)
		# print("\nGo to observation pose...")
		# fa.goto_pose(reset_pose)
		# udp.SendData("Segment")
		# # TODO: RIGHT BEFORE OBSERVATION

	# except timeout:
	# 	print("no message")

		# -------- old message parsing ----------
		# x = float(message.split(',')[0].split('[')[2])
		# y = float(message.split(',')[1])
		# z = float(message.split(',')[2].split(']')[0])
		# area = float(message.split(',')[3])

		# com = get_object_center_point_in_world_realsense_static_camera(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform)
		# print("COM: ", com)

		# robot_pose = fa.get_pose()
		# com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), realsense_intrinsics, realsense_to_ee_transform, robot_pose)
		# # print("COM: ", com)

		# # --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
		# com = np.array([com[0], -com[1] + 0.02, com[2] + 0.02]) # should be the x,y,z position in robot frame
		# print("COM: ", com)
		# robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])

		# fa.goto_pose(robot_pose)
		# time.sleep(5)
	
	# 	if count >= 1:
	# 		# Cutting action: 
	# 		print("\nCutting...")
	# 		fa.goto_gripper(0, block=False)
	# 		fa.apply_effector_forces_along_axis(1.0, 0.5, 0.06, forces=[0.,0.,-75.])
	# 		time.sleep(1)

	# 		print("\nGo to observation pose after cutting...")
	# 		# fa.reset_pose()
	# 		# fa.reset_joints()
	# 		fa.goto_pose(reset_pose)
	# 		break

	# 	print("\nGo to observation pose...")
	# 	fa.goto_pose(reset_pose)

	# except timeout:
	# 	print("no message")

	# count += 1