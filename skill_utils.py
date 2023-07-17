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
	
	def _get_largest_area_idx(self, dict):
		"""
		Iterate through object dictionary and return the key associated with the
		object with the largest mask area.
		"""
		largest_area = float('-inf')
		area_idx = None
		for idx, (_, area, _, _) in dict.items():
			if area > largest_area:
				largest_area = area
				area_idx = idx
		return area_idx
	
	def _parse_multiclass_message(self, raw_message, classes):
		"""
		Parse the text-based message for the multiclass version into
		dictionary format.
		"""
		print("\nRaw Message: ", raw_message)
		objs = ast.literal_eval(raw_message)
		obj_dict = {}
		for i in range(len(objs)):
			class_dict = {}
			for j in range(len(objs[i])):
				class_dict[j] = objs[i][j]
			obj_dict[classes[i]] = class_dict
		return obj_dict
	

	def _intersects(self, bbox1, bbox2):
		"""
		Determines if two bounding boxes (format [[xmin, ymin],[xmax, ymax]]) intersect.
		"""
		x11 = bbox1[0][0]
		y11 = bbox1[0][1]
		x12 = bbox1[1][0]
		y12 = bbox1[1][1]
		x21 = bbox2[0][0]
		y21 = bbox2[0][1]
		x22 = bbox2[1][0]
		y22 = bbox2[1][1]
		if x11 > x22 or x12 < x21 or y11 > y22 or y12 < y21:
			return False
		else:
			return True
		
	def _get_perp_vector(self, vec):
		"""
		Returns the unit vector perpendicular to the input vector.
		"""
		unit = np.array([-vec[1], vec[0]])
		print("Lingal norm: ", np.linalg.norm(unit))
		unit = unit / np.linalg.norm(unit)
		return unit
	
	def _get_rot_matrix(self, starting_rot, z_rot):
		"""
		Given the starting rotation we'd like to set as 0 degree rotation,
		and a new z rotation, return the rotation matrix.
		"""
		orig = Rotation.from_matrix(starting_rot)
		orig_euler = orig.as_euler('xyz', degrees=True)
		rot_vec = np.array([0, 0, z_rot])
		new_euler = orig_euler + rot_vec
		r = Rotation.from_euler('xyz', new_euler, degrees=True)
		rotation = r.as_matrix()
		return rotation
	
	def _axis_push(self, trans, rot, dir):
		"""
		Simple function to push objects along x or y axis only.
		"""
		# rotate blade for push angle
		pose = self.fa.get_pose()
		pose.rotation = rot
		self.fa.goto_pose(pose)
		# goto start position for push
		self.fa.goto_gripper(0, block=False)
		pose.translation = trans
		self.fa.goto_pose(pose)
		# goto final position for push
		push_dist = 0.075 
		xy_push = push_dist * dir
		pose.translation += np.array([xy_push[0], xy_push[1], 0])
		self.fa.goto_pose(pose)
		# goto intermediate z pose to then reset the gripper rotation
		pose.translation = np.array([trans[0], trans[1], 0.25])
		self.fa.goto_pose(pose)
		pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)
	
	def observe_scene_multiclass(self, udp, obs_pose, classes):
		"""
		Move the robot to observation pose, communicate with vision system of UDP Comms,
		iterate through the communicated data and transform the COM positions from camera
		to robot frame.

		return:		obs_objects:	Dictionary with keys of the object classes and values of the number of each class observed in scene.
					obj_dict: 		Dictionary of dictionaries of format dict[object_class][object_index] = (com, area, bbox, longest line pts)
		"""
		# goto observation pose
		print("\nGo to observation pose...")
		self.fa.goto_pose(obs_pose)
		time.sleep(0.5)
		# send message
		udp.SendData("Segment")
		print("Sent message...")
		message = None
		while message is None:
			message = udp.ReadReceivedData()
		print("Message: ", message)

		objs = self._parse_multiclass_message(message, classes)

		# populate observation dictionary
		obs_objects = {}
		obj_dict = {}

		for key in objs:
			key_dict = {}
			i = 0
			for idx in objs[key]:
				element = objs[key][idx]
				x = element[0]
				y = element[1]
				z = element[2]
				area = element[3]
				cut_angle = element[4]
				
				self.robot_pose = self.fa.get_pose()
				com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
				# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
				com = np.array([com[0], com[1] + 0.065, com[2] + 0.02]) # should be the x,y,z position in robot frame
				if len(element) > 4:
					collisions = []
					
					for j in range(5, len(element), 4):
						print("j: ", j)
						print("len(element)", len(element))
						push_x = element[j]
						push_y = element[j+1]
						push_z = element[j+2]
						push_angle = element[j+3]

						push_com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([push_x,push_y,push_z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
						push_com = np.array([push_com[0], push_com[1] + 0.065, push_com[2] + 0.02])
						collisions.append((push_com, push_angle))

						print("\nPush COM: ", push_com)

					key_dict[i] = (com, area, cut_angle, collisions)
				else:
					key_dict[i] = (com, area, cut_angle)
				i+=1
			# populate the object dict (i.e. all the associated COM's of the class in the scene)
			obj_dict[key] = key_dict
			# populate the obs_objects dict (i.e. all the classes in the scene and their frequency number)
			obs_objects[key] = i 
		# return dictionary
		print("\nobj dict: ", obj_dict)
		print("\nobs objects: ", obs_objects)
		return obs_objects, obj_dict

	# # ========= OLD VERSION WHEN COLLISION DETECTION IS ON THIS SIDE ============
	# def observe_scene_multiclass(self, udp, obs_pose, classes):
	# 	"""
	# 	Move the robot to observation pose, communicate with vision system of UDP Comms,
	# 	iterate through the communicated data and transform the COM positions from camera
	# 	to robot frame.

	# 	return:		obs_objects:	Dictionary with keys of the object classes and values of the number of each class observed in scene.
	# 				obj_dict: 		Dictionary of dictionaries of format dict[object_class][object_index] = (com, area, bbox, longest line pts)
	# 	"""
	# 	# goto observation pose
	# 	print("\nGo to observation pose...")
	# 	self.fa.goto_pose(obs_pose)
	# 	time.sleep(0.5)
	# 	# send message
	# 	udp.SendData("Segment")
	# 	print("Sent message...")
	# 	message = None
	# 	while message is None:
	# 		message = udp.ReadReceivedData()
	# 	print("Message: ", message)

	# 	objs = self._parse_multiclass_message(message, classes)

	# 	# populate observation dictionary
	# 	obs_objects = {}
	# 	obj_dict = {}

	# 	for key in objs:
	# 		key_dict = {}
	# 		i = 0
	# 		for idx in objs[key]:
	# 			element = objs[key][idx]
	# 			x = element[0]
	# 			y = element[1]
	# 			z = element[2]
	# 			area = element[3]
	# 			bbox = [[element[4], element[5], element[6]], [element[7], element[8], element[9]]]
	# 			pts = [[element[10], element[11], element[12]], [element[13], element[14], element[15]]]
	# 			self.robot_pose = self.fa.get_pose()
	# 			com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
	# 			# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
	# 			com = np.array([com[0], com[1] + 0.065, com[2] + 0.02]) # should be the x,y,z position in robot frame
	# 			print("COM: ", com)
	# 			key_dict[i] = (com, area, bbox, pts)
	# 			i+=1
	# 		# populate the object dict (i.e. all the associated COM's of the class in the scene)
	# 		obj_dict[key] = key_dict
	# 		# populate the obs_objects dict (i.e. all the classes in the scene and their frequency number)
	# 		obs_objects[key] = i 
	# 	# return dictionary
	# 	print("\nobj dict: ", obj_dict)
	# 	print("\nobs objects: ", obs_objects)
	# 	return obs_objects, obj_dict
	

	def plan_cut_multiclass(self, obj_dict, object_class, even_heuristic):
		"""
		Given the object dictionary and the target cut class, and the cut heuristic, plan the cut action.

		return:		com:		Center of mass in robot frame of the target cut object.
					angle: 		Angle (in degrees) of the gripper to achieve target cut.
					cut_idx:	Index corresponding to the object selected to cut.
		"""
		cut_idx = self._get_largest_area_idx(obj_dict[object_class])
		print("Cut index: ", cut_idx)
		com = obj_dict[object_class][cut_idx][0]
		angle = obj_dict[object_class][cut_idx][2]
		
		return com, angle, cut_idx

	# ========= OLD VERSION ============
	# def plan_cut_multiclass(self, obj_dict, object_class, even_heuristic):
	# 	"""
	# 	Given the object dictionary and the target cut class, and the cut heuristic, plan the cut action.

	# 	return:		com:		Center of mass in robot frame of the target cut object.
	# 				angle: 		Angle (in degrees) of the gripper to achieve target cut.
	# 				cut_idx:	Index corresponding to the object selected to cut.
	# 	"""
	# 	cut_idx = self._get_largest_area_idx(obj_dict[object_class])
	# 	print("Cut index: ", cut_idx)
	# 	com = obj_dict[object_class][cut_idx][0]
	# 	sides = obj_dict[object_class][cut_idx][3]
	# 	# convert to world coordinates
	# 	pt1 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[0][0],sides[0][1],sides[0][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
	# 	pt1 = np.array([pt1[0], pt1[1] + 0.065]) 
	# 	pt2 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[1][0],sides[1][1],sides[1][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
	# 	pt2 = np.array([pt2[0], pt2[1] + 0.065])
	# 	vec = pt2 - pt1


	# 	print("vec: ", vec)

	# 	vector = vec / np.linalg.norm(vec)
	# 	# get rotation of the gripper
	# 	angle = math.degrees(math.atan(vector[1] / vector[0]))
	# 	print("angle: ", angle)	
	# 	# TODO: a bug somewhere around the angle of rotation calculation --> even heuristic not rotating to perpendicular rotation of longest line
	# 	if even_heuristic:
	# 		# should cut at a perpendicular rotation
	# 		# angle += 90
	# 		print("perpendicular angle: ", angle)
	# 	# angle = 0
		
	# 	return com, angle, cut_idx

	def cut_multiclass(self, count, com, angle, obj_class):
		"""
		Execute the cut action given the planned cut action, and update the count dictionary with the assumtion that the 
		cut action was successful.

		return:		count:	Dictionary with keys of the object classes and values of the expected number of slices in the scene.
		"""
		self.prev_cut_pos = com
		pose = self.fa.get_pose()
		rot = self._get_rot_matrix(self.og_rotation, angle) 
		self.prev_cut_rot = rot
		self.prev_cut_angle = angle
		# goto com with offset
		pose.translation = np.array([com[0], com[1], com[2] + 0.10])
		pose.rotation = rot
		self.fa.goto_pose(pose)
		time.sleep(0.5)

		# # Executing cutting action
		# print("\nCutting...")
		# self.fa.goto_gripper(0, block=False)
		# # cut action
		# # TODO: specify max height with object height
		# self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.065, forces=[0.,0.,-75.])
		# # self.fa.apply_effector_forces_along_axis(1.0, 0.5, 0.055, forces=[0.,0.,-75.])
		# time.sleep(1)

		# # TODO: potentailly add a pose check and if the gripper isn't at the cutting board height, re-try??? or return count unchanged???
		# count[obj_class] += 1
		# # rotate blade back to original rotation
		# pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)
		return count

	def disturb_scene(self):
		"""
		Move the robot to the previous cut pose, and apply slight rotations to separate the two pieces.
		"""
		print("\nDisturbing scene...")
		pose = self.fa.get_pose()
		# goto previous slice position and rotation disturb the scene slightly 
		self.fa.goto_gripper(0, block=False)
		pose.translation = np.array([self.prev_cut_pos[0], self.prev_cut_pos[1], 0.125])
		# pose.rotation = self.prev_cut_rot # NOTE: don't want to rotate, because we rotate the gripper after cutting
		self.fa.goto_pose(pose)
		# perform rotational disturbance
		th_delta = 20 # [degrees]
		rot = self._get_rot_matrix(self.prev_cut_rot, th_delta) 
		pose.rotation = rot
		self.fa.goto_pose(pose)
		rot = self._get_rot_matrix(self.prev_cut_rot, 2*th_delta) 
		pose.rotation = rot
		self.fa.goto_pose(pose)
		pose.translation = np.array([self.prev_cut_pos[0], self.prev_cut_pos[1], 0.22])
		pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)

	def push_away_from_wall(self, com, rotation):
		"""
		This function detects if the blade will come into contact with the workspace
		boarders during cut.

		return:		Bool:	True if there was a predicted collision with the wall, and we moved an object in the scene, False if no action.
		"""
		tool_dim = 0.18 # x,y in [m]
		# NOTE: rotation expected in degrees
		print("Rotation: ", rotation)
		Lx = abs(0.5*tool_dim*math.sin(rotation))
		# print("math.sin(rotation) ", math.sin(rotation))
		# print("LX: ", Lx)
		Ly = abs(0.5*tool_dim*math.cos(rotation))
		# print("math.cos(rotation) ", math.cos(rotation))
		# print("Ly: ", Ly)
		# print("COM: ", com)
		x1 = com[0] - Lx
		x2 = com[0] + Lx
		y1 = com[1] - Ly
		y2 = com[1] + Ly
		tool_bb = [[x1, y1], [x2, y2]] # (x1, x2, y1, y2)
		# print("\nTool Bounding Box: ", tool_bb)
		# print("Cut EE Pose: ", self.fa.get_pose().translation)

		# ------ Measured 3D Printed Boundary Positions (with some margin) -------
		minx = 0.335 
		maxx = 0.69 
		miny = -0.235
		maxy = 0.26

		# return pose and orientation of the gripper and the push direction
		if tool_bb[0][0] <= minx or tool_bb[1][0] <= minx:
			print("less than x")
			trans = np.array([minx, com[1], 0.145])
			rot = self.og_rotation
			dir = np.array([1, 0])
			self._axis_push(trans, rot, dir)
			return True
		elif tool_bb[0][0] >= maxx or tool_bb[1][0] >= maxx:
			print("greater than x")
			trans = np.array([maxx, com[1], 0.145])
			rot = self.og_rotation
			dir = np.array([-1, 0])
			self._axis_push(trans, rot, dir)
			return True
		elif tool_bb[0][1] <= miny or tool_bb[1][1] <= miny:
			print("less than y")
			trans = np.array([com[0], miny, 0.145])
			angle = 90
			dir = np.array([0, 1])
			rot = self._get_rot_matrix(self.og_rotation, angle) 
			self._axis_push(trans, rot, dir)
			return True
		elif tool_bb[0][1] >= maxy or tool_bb[1][1] >= maxy:
			print("greater than y")
			trans = np.array([com[0], maxy, 0.145])
			angle = 90
			dir = np.array([0, -1])
			rot = self._get_rot_matrix(self.og_rotation, angle) 
			self._axis_push(trans, rot, dir)
			return True
		return False
	
	# def check_cut_collisions_multiclass(self, blade_com, obj_dict, rotation, cut_idx):
	# 	"""
	# 	Predicts collisions between the blade and objects in the scene that are not the target
	# 	cutting object. First, generate a bounding box for the blade given the target COM and 
	# 	rotation, then iterate through object dictionary to generate a bounding box for each
	# 	object in the scene and check for collisions with the blade bounding box.

	# 	return:		collision_idxs:	List of objects that are estimated to be in collision with proposed blade action.
	# 	"""
	# 	print("\n\n-------collision detection----------")

	# 	tool_dim = 0.16 # x,y in [m]
	# 	# NOTE: rotation expected in degrees
	# 	Lx = abs(0.5*tool_dim*math.sin(rotation))
	# 	Ly = abs(0.5*tool_dim*math.cos(rotation))
	# 	x1 = blade_com[0] - Lx
	# 	x2 = blade_com[0] + Lx
	# 	y1 = blade_com[1] - Ly
	# 	y2 = blade_com[1] + Ly
	# 	tool_bb = [[x1, y1], [x2, y2]] # (x1, x2, y1, y2)
	# 	print("Tool BB: ", tool_bb)

	# 	# TODO: current bug --> we are getting identical bounding boxes as we loop through obj_dict (perhaps vision system issue?)
	# 	collision_idxs = []
	# 	for obj_class in obj_dict:
	# 		for idx in obj_dict[obj_class]:
	# 			# do not want to check for blade collision with the target cutting object --> we want this collision!
	# 			if idx != cut_idx:
	# 				bb_cam_frame = obj_dict[obj_class][idx][2]
	# 				print("bb cam frame: ", bb_cam_frame)
	# 				pt1 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([bb_cam_frame[0][0],bb_cam_frame[0][1],bb_cam_frame[0][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
	# 				l1 = [pt1[0], pt1[1] + 0.065]
	# 				pt2 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([bb_cam_frame[1][0],bb_cam_frame[1][1],bb_cam_frame[1][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
	# 				l2 = [pt2[0], pt2[1] + 0.065]
	# 				obj_bb = [l1, l2]
	# 				print("Object BB: ", obj_bb)
	# 				if self._intersects(tool_bb, obj_bb):
	# 					print("Intersection detected...")
	# 					collision_idxs.append(obj_dict[obj_class][idx][0:2])
	# 					# collision_idxs.append([obj_class, idx])
	# 	return collision_idxs
	

	def push(self, cut_obj_com, push_com, push_angle):
		"""
		"""
		dir_vector = (push_com[0:2] - cut_obj_com[0:2]) / np.linalg.norm(push_com[0:2] - cut_obj_com[0:2])
		print("dir vector: ", dir_vector)
		pose = self.fa.get_pose()
		rot_matrix = self._get_rot_matrix(self.og_rotation, push_angle) 
		# rotate blade for push angle
		pose = self.fa.get_pose()
		pose.rotation = rot_matrix
		self.fa.goto_pose(pose)
		print("...rotating blade...")
		# goto start position for push
		self.fa.goto_gripper(0, block=False)
		pose.translation = np.array([push_com[0], push_com[1], 0.135])
		print("\npush pose: ", pose.translation)
		self.fa.goto_pose(pose)
		# goto final position for push
		push_dist = 0.05 
		xy_push = push_dist * dir_vector
		print("xy push: ", xy_push)
		new_pose = pose.translation + np.array([xy_push[0], xy_push[1], 0])
		pose.translation = new_pose
		self.fa.goto_pose(pose)
		# goto intermediate z pose to then reset the gripper rotation
		pose.translation = np.array([new_pose[0], new_pose[1], 0.25])
		self.fa.goto_pose(pose)
		pose.rotation = self.og_rotation
		self.fa.goto_pose(pose)

	# ============= OLD VERSION ===================
	# def push(self, cut_obj_com, push_obj_com):
	# 	"""
	# 	Given the target object and push object coms, plan and execute the push action to move the push object further away.
	# 	"""
	# 	print("\n\n-------PUSH ACTION---------")
	# 	print("Cut COM: ", cut_obj_com)
	# 	print("Push obj com: ", push_obj_com)
	# 	# NOTE: cut_obj_com and push_obj_com should only have x and y coordinates (no z)
	# 	dir_vector = (push_obj_com - cut_obj_com) / np.linalg.norm(push_obj_com - cut_obj_com)
	# 	print("dir vector: ", dir_vector)
	# 	perp_vector = self._get_perp_vector(dir_vector)
	# 	print("perp vector: ", perp_vector)
	# 	rot_angle = math.degrees(math.atan(perp_vector[1] / perp_vector[0]))
	# 	print("rot angle: ", rot_angle)
	# 	pose = self.fa.get_pose()
	# 	rot_matrix = self._get_rot_matrix(self.og_rotation, rot_angle) 
	# 	# rotate blade for push angle
	# 	pose = self.fa.get_pose()
	# 	pose.rotation = rot_matrix
	# 	self.fa.goto_pose(pose)
	# 	print("...rotating blade...")
	# 	# goto start position for push
	# 	self.fa.goto_gripper(0, block=False)
	# 	offset = 0.02 # TODO: potentially change this to the object diameter???
	# 	xy_offset = -offset * dir_vector 
	# 	print("xy offset: ", xy_offset)
	# 	pose.translation = np.array([push_obj_com[0] + xy_offset[0], push_obj_com[1] + xy_offset[1], 0.135])
	# 	print("start pose: ", pose.translation)
	# 	self.fa.goto_pose(pose)
	# 	print("...go to starting pose...")
	# 	# goto final position for push
	# 	print("...go to final pose...")
	# 	push_dist = 0.05 
	# 	xy_push = push_dist * dir_vector
	# 	print("xy push: ", xy_push)
	# 	new_pose = pose.translation + np.array([xy_push[0], xy_push[1], 0])
	# 	print("final push loc: ", new_pose)
	# 	pose.translation = new_pose
	# 	self.fa.goto_pose(pose)
	# 	# goto intermediate z pose to then reset the gripper rotation
	# 	print("...go to intermediate pose...")
	# 	pose.translation = np.array([new_pose[0], new_pose[1], 0.25])
	# 	self.fa.goto_pose(pose)
	# 	pose.rotation = self.og_rotation
	# 	self.fa.goto_pose(pose)








# -------- FUNCTIONS FOR SINGLE CLASS ONLY -----------
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
			bbox = [[obj[4], obj[5], obj[6]], [obj[7], obj[8], obj[9]]]
			pts = [[obj[10], obj[11], obj[12]], [obj[13], obj[14], obj[15]]]
			self.robot_pose = self.fa.get_pose()
			com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
			# --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
			com = np.array([com[0], com[1] + 0.065, com[2] + 0.02]) # should be the x,y,z position in robot frame
			print("COM: ", com)
			obj_dict[i] = (com, area, bbox, pts)
		# return dictionary
		return obs_objects, obj_dict
	
	def plan_cut(self, obj_dict):
		"""
		Observed offset:
		0.025 in y (too much <-- which is -y)
		"""
		cut_idx = self._get_largest_area_idx(obj_dict)
		com = obj_dict[cut_idx][0]
		sides = obj_dict[cut_idx][3] # [[x1, y1, z1], [x2, y2, z2]]
		# convert to world coordinates
		pt1 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[0][0],sides[0][1],sides[0][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
		pt1 = np.array([pt1[0], pt1[1] + 0.065]) 
		pt2 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([sides[1][0],sides[1][1],sides[1][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
		pt2 = np.array([pt2[0], pt2[1] + 0.065]) 
		# find perpendicular vector
		vec = pt2 - pt1
		perp_vector = self._get_perp_vector(vec)
		# get rotation of the gripper
		angle = math.degrees(math.atan(perp_vector[1] / perp_vector[0]))
		# angle = 0
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

	def check_cut_collisions(self, blade_com, obj_dict, rotation, cut_idx):
		"""
		This function checks for collisions of blade with non-target objects.

		- generate simple bounding box (4 corners) based on blade rotation
		- iterate through obj_dict and check for collisions between blade and obj bounding boxes
		- return list of obj_idxs that result in collisions of the bounding boxes
		"""
		tool_dim = 0.16 # x,y in [m]
		# NOTE: rotation expected in degrees
		Lx = 0.5*tool_dim*math.sin(rotation)
		Ly = 0.5*tool_dim*math.cos(rotation)
		x1 = blade_com[0] - Lx
		x2 = blade_com[0] + Lx
		y1 = blade_com[1] - Ly
		y2 = blade_com[1] + Ly
		tool_bb = [[x1, y1], [x2, y2]] # (x1, x2, y1, y2)

		collision_idxs = []
		for idx in obj_dict:
			if idx != cut_idx:
				bb_cam_frame = obj_dict[idx][2]
				pt1 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([bb_cam_frame[0][0],bb_cam_frame[0][1],bb_cam_frame[0][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
				l1 = [pt1[0], pt1[1] + 0.065]
				pt2 = get_object_center_point_in_world_realsense_3D_camera_point(np.array([bb_cam_frame[1][0],bb_cam_frame[1][1],bb_cam_frame[1][2]]), self.realsense_intrinsics, self.realsense_to_ee_transform, self.robot_pose)
				l2 = [pt2[0], pt2[1] + 0.065]
				obj_bb = [l1, l2]
				if self._intersects(tool_bb, obj_bb):
					collision_idxs.append(idx)
		return collision_idxs