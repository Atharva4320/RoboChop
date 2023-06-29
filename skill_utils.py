import numpy as np
from frankapy import FrankaArm
import ast
import re
import time
import argparse
import json
import math
from utils import *
from numpy import array
from perception import CameraIntrinsics
from scipy.spatial.transform import Rotation

class SkillUtils():
	def __init__(self, fa):
		self.fa = fa
		self.og_pose = self.fa.get_pose()
		self.og_rotation = self.og_pose.rotation
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

	def observe_scene(self, udp, obs_pose):
		# goto observation pose
		print("\nGo to observation pose...")
		self.fa.goto_pose(obs_pose)
		time.sleep(0.5)
		# send message
		udp.SendData("Segment")
		print("Sent message...")
		message = None
		while message is None:
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
			bbox = obj[4]
			pts = obj[5]
			self.robot_pose = self.fa.get_pose()
			com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
			# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
			com = np.array([com[0], com[1] + 0.04, com[2] + 0.02]) # should be the x,y,z position in robot frame
			print("COM: ", com)
			obj_dict[i] = (com, area, bbox, pts)
		# return dictionary
		return obs_objects, obj_dict
	
	def cut_multiclass(self, obj_dict, count, object_class):
		# return the com with the largest area and hover above that object
		idx = self.get_largest_area_idx(obj_dict[object_class])
		com = obj_dict[object_class][idx][0]
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
		count[object_class] += 1
		# perhaps add in wiggle here
		return count
	
	def plan_cut(self, obj_dict):
		cut_idx = self.get_largest_area_idx(obj_dict)
		com = obj_dict[cut_idx][0]

		# TODO: access the sides in obj_dict
		sides = obj_dict[cut_idx][3]
		# convert to world coordinates
		pt1 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[0][0],sides[0][1],sides[0][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
		pt1 = np.array([pt1[0], pt1[1] + 0.04]) 
		pt2 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[1][0],sides[1][1],sides[1][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
		pt2 = np.array([pt2[0], pt2[1] + 0.04]) 
		# find perpendicular vector
		vec = pt2 - pt1
		perp_vector = self._get_perp_vector(vec)
		# get rotation of the gripper
		angle = math.degrees(math.arctan(perp_vector[1] / perp_vector[0]))
		return com, angle

	def cut(self, count, com, angle):
		self.prev_cut_pos = com
		pose = self.fa.get_pose()
		rot = self._get_rot_matrix(self.og_rotation, angle) 
		self.prev_cut_rot = rot
		self.prev_cut_angle = angle
		# goto com with offset
		pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		pose.rotation = rot
		self.fa.goto_pose(pose)
		# self.robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		# self.fa.goto_pose(self.robot_pose)
		time.sleep(0.5)

		# Executing cutting action
		print("\nCutting...")
		self.fa.goto_gripper(0, block=False)
		# cut action
		# TODO: specify max height with object height
		self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.055, forces=[0.,0.,-75.])
		# self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.055, forces=[0.,0.,-75.])
		time.sleep(1)
		count += 1
		# rotate blade back to original rotation
		pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)
		return count

	def disturb_scene(self):
		"""
		"""
		print("\nDisturbing scene...")
		pose = self.fa.get_pose()
		# goto previous slice position and rotation disturb the scene slightly perpendicular to rotation
		dist = 0.035
		vec = np.array([math.sin(self.prev_cut_angle) * 0.08, math.cos(self.prev_cut_angle) * 0.08])
		xy_dists = dist * self._get_perp_vector(vec)
		self.fa.goto_gripper(0, block=False)
		pose.translation = np.array([self.prev_cut_pos[0], self.prev_cut_pos[1], 0.12])
		pose.rotation = self.prev_cut_rot
		self.fa.goto_pose(pose)
		# disturb the scene --> currently assuming cutting blade fixed, so only move in x-direction
		pose.translation = np.array([self.prev_cut_pos[0] - xy_dists[0], self.prev_cut_pos[1] - xy_dists[1], 0.12])
		# pose.translation = np.array([self.prev_cut_pos[0] - 0.03, self.prev_cut_pos[1], 0.12])
		self.fa.goto_pose(pose)
		pose.translation = np.array([self.prev_cut_pos[0] + xy_dists[0], self.prev_cut_pos[1] + xy_dists[1], 0.12])
		# pose.translation = np.array([self.prev_cut_pos[0] + 0.03, self.prev_cut_pos[1], 0.12])
		self.fa.goto_pose(pose)
		# reset to original rotation
		pose.translation = np.array([self.prev_cut_pos[0], self.prev_cut_pos[1], 0.22])
		pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)

	def _intersects(self, bbox1, bbox2):
		if (bbox1[0] <= bbox2[1] or bbox1[1] >= bbox2[0]) and (bbox1[2] <= bbox2[3] or bbox1[3] >= bbox2[2]):
			return True
		return False

	def check_cut_collisions(self, blade_com, obj_dict, rotation):
		"""
		- generate simple bounding box (4 corners) based on blade rotation
		- iterate through obj_dict and check for collisions between blade and obj bounding boxes
		- return list of obj_idxs that result in collisions of the bounding boxes
		"""
		tool_dim = 0.16 # x,y in [m]
		# NOTE: rotation expected in degrees
		Lx = 0.5*tool_dim*math.cos(rotation)
		Ly = 0.5*tool_dim*math.sin(rotation)
		x1 = blade_com[0] - Lx
		x2 = blade_com[0] + Lx
		y1 = blade_com[1] - Ly
		y2 = blade_com[1] + Ly
		tool_bb = (x1, x2, y1, y2)

		collision_idxs = []
		for idx in obj_dict:
			obj_bb = obj_dict[idx][2]
			if self._intersects(tool_bb, obj_bb):
				collision_idxs.append(idx)
		return collision_idxs

	def _get_perp_vector(self, vec):
		unit = np.array([vec[1], -vec[0]])
		unit = unit / np.linal.norm(unit)
		return unit
	
	def _get_rot_matrix(self, starting_rot, z_rot):
		orig = Rotation.from_matrix(starting_rot)
		orig_euler = orig.as_euler('xyz', degrees=True)
		rot_vec = np.array([0, 0, z_rot])
		new_euler = orig_euler + rot_vec
		r = Rotation.from_euler('xyz', new_euler, degrees=True)
		rotation = r.as_matrix()
		return rotation

	def push(self, cut_obj_com, push_obj_com):
		"""
		"""
		# NOTE: cut_obj_com and push_obj_com should only have x and y coordinates (no z)
		dir_vector = (push_obj_com - cut_obj_com) / np.linalg.norm(push_obj_com - cut_obj_com)
		perp_vector = self._get_perp_vector(dir_vector)
		rot_angle = math.degrees(math.arctan(perp_vector[1] / perp_vector[0]))
		pose = self.fa.get_pose()
		rot_matrix = self._get_rot_matrix(self.og_rotation, rot_angle) 
		# rotate blade for push angle
		pose = self.fa.get_pose()
		pose.rotation = rot_matrix
		self.fa.goto_pose(pose)
		# goto start position for push
		offset = None
		xy_offset = offset * dir_vector
		pose.translation = np.array([cut_obj_com[0] + xy_offset[0], cut_obj_com[1] + xy_offset[1], 0.12])
		self.fa.goto_pose(pose)
		# goto final position for push
		push_dist = None
		xy_push = push_dist * dir_vector
		pose.translation = np.array([xy_push[0], xy_push[1], 0.12])
		self.fa.goto_pose(pose)
		# goto intermediate z pose to then reset the gripper rotation
		pose.translation = np.array([xy_push[0], xy_push[1], 0.25])
		self.fa.goto_pose(pose)
		pose.rotation = og_rot
		self.fa.goto_pose(pose)