from frankapy import FrankaArm
import numpy as np

# from actions import move_arm, cut, push
# from vision import detect_peach, detect_slices

from skill_library import SkillLibrary

fa = FrankaArm()
skills = SkillLibrary(fa)
fa.reset_pose()
fa.reset_joints()
pose = fa.get_pose()
pose.translation = np.array([0.65, 0, 0.4])
fa.goto_pose(pose)

"""
def cut_peach_into_n(n):
    peach_position, peach_size = skills.detect_object()
    
    # Detect slices
    slices = skills.detect_slices()
    while len(slices) < n:
        # Find the largest slice and cut it at the side
        largest_slice_position = max(slices, key=lambda x: x[1]) # Assuming detect_slices() returns a list of tuples in the format (position, size)
        
        skills.move_arm(largest_slice_position[0] + peach_size/4, largest_slice_position[1]) # Move a quarter of the peach's size to the side to avoid the pit
        skills.cut(5, 0, 0.5)
        
        # Detect slices
        slices = skills.detect_slices()
        print(f"Peach cut into {n} slices.")
"""

def cut_apple(num_slices):
    # Define orange size
    apple_radius = 0.2  # assuming orange radius is 0.2 feet
    blade_length = 0.5  # the length of the blade is 0.5 feet

    # Detect the orange
    apple_position = skills.detect_object() # "apple")

    # # Cut off both ends of the orange
    # skills.move_arm(apple_position[0], apple_position[1] - apple_radius)
    # skills.cut(blade_length, 0, 1)
    # skills.move_arm(apple_position[0], apple_position[1] + apple_radius)
    # skills.cut(blade_length, 0, 1)

    # Calculate the rotation angle for each slice
    rotation_angle = 360 / num_slices

    # Cut the orange into slices
    for i in range(num_slices):
        # Move the arm to the cutting position
        skills.move_arm(apple_position[0], apple_position[1])

        # Make the cut with the appropriate rotation angle
        skills.cut(blade_length, rotation_angle * i, 1)

    # Push the slices away to separate them
    skills.push(apple_position[0], apple_position[0] + 2 * apple_radius)

"""
def cut_orange(num_slices):
    # Define orange size
    orange_radius = 0.2  # assuming orange radius is 0.2 feet
    blade_length = 0.5  # the length of the blade is 0.5 feet

    # Detect the orange
    orange_position = skills.detect_object("orange")

    # Hold the orange in place to prevent it from rolling
    skills.hold_object(orange_position[0], orange_position[1])

    # Cut off both ends of the orange
    skills.move_arm(orange_position[0], orange_position[1] - orange_radius)
    skills.cut(blade_length, 0, 1)
    skills.move_arm(orange_position[0], orange_position[1] + orange_radius)
    skills.cut(blade_length, 0, 1)

    # Calculate the rotation angle for each slice
    rotation_angle = 360 / num_slices

    # Cut the orange into slices
    for i in range(num_slices):
        # Move the arm to the cutting position
        skills.move_arm(orange_position[0], orange_position[1])

        # Make the cut with the appropriate rotation angle
        skills.cut(blade_length, rotation_angle * i, 1)

    # Release the orange
    skills.hold_object(orange_position[0], orange_position[1], release=True)

    # Push the slices away to separate them
    skills.push(orange_position[0], orange_position[0] + 2 * orange_radius)

def cut_cucumber_into_slices():
    # Define cucumber length and thickness of slice
    cucumber_length = 2  # assuming cucumber length is 2 feet
    slice_thickness = cucumber_length / 10  # slicing cucumber into 10 slices
    blade_length = 0.5  # the length of the blade is 0.5 feet
    
    # Detect the cucumber
    cucumber_position = skills.detect_object("cucumber")
    # Cut the cucumber into slices
    for i in range(10):  # we want 10 slices
        # Move the robot arm to the cucumber
        skills.move_arm(cucumber_position[0], cucumber_position[1] + i * slice_thickness)
        
        # Cut the cucumber
        # skills.cut(slice_thickness, 0, 1)
        
        # Push the slices away to separate them
        skills.push(cucumber_position[0], cucumber_position[0] + cucumber_length)
"""

cut_apple(2)