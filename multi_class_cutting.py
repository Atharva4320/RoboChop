from utils import *
from frankapy import FrankaArm
import UdpComms as U
from skill_utils import *
import time


# --------- DEFINE TARGET PIECES HERE --------
n_pieces = {
	"Apple": 3,
	"Cucumber": 1,
	"Carrot": 0
}
EVEN = True # heuristic for slice type 

# updIP: This computer, SendIP: other computer 
udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)

# send message of the target classes to detect
object_string = ''
classes = []
for key in n_pieces:
	classes.append(key)
	object_string+=key
	object_string+=','
object_string = object_string[0:len(object_string)-1]
print("object_string: ", object_string)

udp.SendData(object_string)

print("\nReset pose...")
fa = FrankaArm()
fa.reset_pose()
fa.reset_joints()
reset_pose = fa.get_pose()
reset_pose.translation = np.array([0.65, 0, 0.58]) # x was 0.55 before

skills = SkillUtils(fa)
obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)

print("entering loop....")
count = obs_objects
prev_cut_pos = np.array([0.65, 0, 0.4]) # random initialization
object_list = list(n_pieces.keys())
for object in object_list:

	while n_pieces[object] > obs_objects[object]:

		# check if the expected count doesn't match the observed objects
		if count[object] != obs_objects[object]:
			if count[object] > obs_objects[object]:
				skills.disturb_scene()
				obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)

			else:
				print("ERROR: we are incorrectly observing more objects than expected")
				skills.disturb_scene()
				obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)

		# go to cut 
		else:
			# plan cut action (get com and angle)
			com, angle, cut_idx = skills.plan_cut_multiclass(obj_dict, object, even_heuristic=EVEN) 
			print("Cut idx: ", cut_idx)
			# check for collisions with boundary walls
			wall_collision = skills.push_away_from_wall(com, angle)
			while wall_collision:
				obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)
				com, angle, cut_idx = skills.plan_cut_multiclass(obj_dict, object, even_heuristic=EVEN) 
				wall_collision = skills.push_away_from_wall(com, angle)
			# get collision check with other objects in scene
			print("Dict: ", obj_dict[object][cut_idx])
			collisions = obj_dict[object][cut_idx][3]
			# while len(obj_dict[object][cut_idx]) > 3:
			while len(collisions) > 0:
				print("collisions: ", collisions)
				skills.push(com, collisions[0][0], collisions[0][1])
				obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)
				com, angle, cut_idx = skills.plan_cut_multiclass(obj_dict, object, even_heuristic=EVEN) 
				collisions = obj_dict[object][cut_idx][3]


			# ======= OLD COLLISION DETECTION VERSION =======
			# # check for collisions with other objects
			# collisions = skills.check_cut_collisions_multiclass(com, obj_dict, angle, cut_idx)  
			# while len(collisions) > 0:
			# 	print("Found ", len(collisions), " collisions")
			# 	for push_obj_com, _ in collisions:
			# 		print("\nPush obj com -- should just be x,y: ", push_obj_com)
			# 		skills.push(com, push_obj_com)
			# 	obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)
			# 	collisions = skills.check_cut_collisions_multiclass(com, obj_dict, angle, cut_idx)
			# # when no collisions, execute cut action
			# print("No found collisions")
			count = skills.cut_multiclass(count, com, angle, object)
			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose, classes)

print("Completing cutting task!")
print("Final oberved object states: ", obs_objects)

	# while n_pieces[object] > obs_objects[object]:
	# 	if count[object] != obs_objects[object]:
	# 		if count[object] > obs_objects[object]:
	# 			skills.disturb_scene()
	# 			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)
	# 		else:
	# 			print("ERROR: we are incorrectly observing more objects than expected")
	# 			skills.disturb_scene()
	# 			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)
	# 	else:
	# 		count = skills.cut_multiclass(obj_dict, count, object)


# while skills.check_dict_values_greater(n_pieces, obs_objects):

# 	# check if the expected count doesn't match the observed objects
# 	if skills.check_dict_values_not_equal(count, obs_objects):
# 		if skills.check_dict_values_greater(count, obs_objects):
# 			skills.disturb_scene()
# 			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

# 		else:
# 			print("ERROR: we are incorrectly observing more objects than expected")
# 			skills.disturb_scene()
# 			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

# 	# go to cut 
# 	else:
# 		# TODO: ensure the cut skill is cutting the correct object
# 		count = skills.cut_multiclass(obj_dict)
# 		obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

print("Completing cutting task!")
print("Final oberved object states: ", obs_objects)