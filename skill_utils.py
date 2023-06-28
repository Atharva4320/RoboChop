import numpy as np
from frankapy import FrankaArm
import ast
import re
import time
import argparse
import json
from utils import *
from numpy import array
from perception import CameraIntrinsics

class SkillUtils():
	def __init__(self, fa):
		self.fa = fa
		REALSENSE_INTRINSICS = "vision_module/2D_vision/calib/realsense_intrinsics.intr"
		REALSENSE_EE_TF = "vision_module/2D_vision/calib/realsense_ee_shifted.tf"
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"--intrinsics_file_path", type=str, default=REALSENSE_INTRINSICS
		)
		parser.add_argument("--extrinsics_file_path", type=str, default=REALSENSE_EE_TF)
		args = parser.parse_args()
		self.realsense_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
		self.realsense_to_ee_transform = RigidTransform.load(args.extrinsics_file_path)
	
	def get_largest_area_idx(self, dict):
		largest_area = float('-inf')
		area_idx = None
		for idx, (_, area) in dict.items():
			if area > largest_area:
				largest_area = area
				area_idx = idx
		return area_idx
	
	def get_largest_area_idx_multiclass(self, dict):
		largest_area = float('-inf')
		area_idx = None
		key = None
		for elem in dict:
			for idx, (_, area) in dict.get(elem).items():
				if area > largest_area:
					largest_area = area
					area_idx = idx
					key = elem
		return key, area_idx
	
	def observe_scene_multiclass(self, udp, obs_pose):
		# goto observation pose
		print("\nGo to observation pose...")
		self.fa.goto_pose(obs_pose)

		message = None
		while message is None:
			udp.SendData("Segment")
			message = udp.ReadReceivedData()
		print("Message: ", message)

		objs = json.loads(message)
		print("objs: ", objs)

		# # parse text-based list of arrays to actual list of arrays
		# val = re.findall(r"\((.*?)\)", message)
		# val = list(map(ast.literal_eval, val))
		# objs = list(map(array, val))

		# populate observation dictionary
		obs_objects = {}
		obj_dict = {}

		for key in objs:
			key_dict = {}
			i = 0
			for element in key:
				x = element[0]
				y = element[1]
				z = element[2]
				area = element[3]

				self.robot_pose = self.fa.get_pose()
				com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)

				# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
				com = np.array([com[0], com[1] + 0.04, com[2] + 0.02]) # should be the x,y,z position in robot frame
				print("COM: ", com)
				key_dict[i] = (com, area)
				i+=1
			# populate the object dict (i.e. all the associated COM's of the class in the scene)
			obj_dict[key] = key_dict

			# populate the obs_objects dict (i.e. all the classes in the scene and their frequency number)
			obs_objects[key] = i+1
		
		# return dictionary
		return obs_objects, obj_dict
	
	def check_dict_values_greater(self, dict1, dict2):
		for key in dict1:
			if dict1[key] > dict2.get(key, 0):
				return True
		return False
	
	def check_dict_values_not_equal(self, dict1, dict2):
		for key in dict1:
			if dict1[key] != dict2.get(key, 0):
				return True
		return False

	def observe_scene(self, udp, obs_pose):
		# goto observation pose
		print("\nGo to observation pose...")
		self.fa.goto_pose(obs_pose)
		# # send message
		# udp.SendData("Segment")
		# # get message
		message = None
		while message is None:
			# send message
			udp.SendData("Segment")
			# get message
			message = udp.ReadReceivedData()
		# if message is None:
		# 	message = udp.ReadReceivedData()
		print("Message: ", message)
		# parse text-based list of arrays to actual list of arrays
		val = re.findall(r"\((.*?)\)", message)
		val = list(map(ast.literal_eval, val))
		objs = list(map(array, val))
		# populate observation dictionary
		obs_objects = len(objs)
		obj_dict = {}
		for i in range(len(objs)):
			obj = objs[i]
			x = obj[0]
			y = obj[1]
			z = obj[2]
			area = obj[3]

			self.robot_pose = self.fa.get_pose()
			com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)

			# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
			com = np.array([com[0], com[1] + 0.04, com[2] + 0.02]) # should be the x,y,z position in robot frame
			print("COM: ", com)
			obj_dict[i] = (com, area)
		
		# return dictionary
		return obs_objects, obj_dict
	
	def cut_multiclass(self, obj_dict, count):
		# return the com with the largest area and hover above that object
		key, idx = self.get_largest_area_idx_multiclass(obj_dict)
		com = obj_dict[key][idx][0]
		self.prev_cut_pos = com
		# goto com with offset
		self.robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		self.fa.goto_pose(self.robot_pose)
		time.sleep(0.5)

		# Executing cutting action
		print("\nCutting...")
		self.fa.goto_gripper(0, block=False)
		# cut action
		self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.06, forces=[0.,0.,-75.])
		time.sleep(1)
		count[key] += 1
		# perhaps add in wiggle here
		return count

	def cut(self, obj_dict, count):
		# return the com with the largest area and hover above that object
		idx = self.get_largest_area_idx(obj_dict)
		com = obj_dict[idx][0]
		self.prev_cut_pos = com
		# goto com with offset
		self.robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		self.fa.goto_pose(self.robot_pose)
		time.sleep(0.5)

		# Executing cutting action
		print("\nCutting...")
		self.fa.goto_gripper(0, block=False)
		# cut action
		self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.06, forces=[0.,0.,-75.])
		time.sleep(1)
		count += 1
		# perhaps add in wiggle here
		return count

	def disturb_scene(self):
		"""
		"""
		# goto previous slice position and disturb the scene slightly
		self.robot_pose.translation = np.array([self.prev_cut_pos[0], self.prev_cut_pos[1], self.prev_cut_pos[2]])
		self.fa.goto_pose(self.robot_pose)
		# disturb the scene --> currently assuming cutting blade fixed, so only move in x-direction
		self.robot_pose.translation = np.array([self.prev_cut_pos[0] - 0.05, self.prev_cut_pos[1], self.prev_cut_pos[2]])
		self.fa.goto_pose(self.robot_pose)
		self.robot_pose.translation = np.array([self.prev_cut_pos[0] + 0.05, self.prev_cut_pos[1], self.prev_cut_pos[2]])
		self.fa.goto_pose(self.robot_pose)

	def clear_area(self, com):
		"""
		function to check if there are nearby centroids within the designated radius, and push objects out of the way
		"""
		pass