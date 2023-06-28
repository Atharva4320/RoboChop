from utils import *
from frankapy import FrankaArm
import UdpComms as U
from skill_utils import *
import time


# --------- DEFINE TARGETS HERE --------
n_pieces = {
	'Apple': 2,
	'Cucumber': 5
}

# updIP: This computer, SendIP: other computer 
udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)

# send message of the target classes to detect
object_string = ''
for key in n_pieces:
	object_string+=key
	object_string+=','
object_string = object_string[0:len(object_string)-1]
print("object_string: ", object_string)
udp.SendData(object_string)
time.sleep(20) # TODO: determine how long we need to sleep here

print("\nReset pose...")
fa = FrankaArm()
fa.reset_pose()
fa.reset_joints()
reset_pose = fa.get_pose()
reset_pose.translation = np.array([0.65, 0, 0.4]) # x was 0.55 before

skills = SkillUtils(fa)
obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

print("entering loop....")
count = obs_objects
prev_cut_pos = np.array([0.65, 0, 0.4])


while skills.check_dict_values_greater(n_pieces, obs_objects):

	# check if the expected count doesn't match the observed objects
	if skills.check_dict_values_not_equal(count, obs_objects):
		if skills.check_dict_values_greater(count, obs_objects):
			skills.disturb_scene()
			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

		else:
			print("ERROR: we are incorrectly observing more objects than expected")
			skills.disturb_scene()
			obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

	# go to cut 
	else:
		count = skills.cut_multiclass(obj_dict)
		obs_objects, obj_dict = skills.observe_scene_multiclass(udp, reset_pose)

print("Completing cutting task!")
print("Final oberved object states: ", obs_objects)