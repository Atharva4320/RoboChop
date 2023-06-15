from frankapy import FrankaArm

# from actions import move_arm, cut, push
# from vision import detect_peach, detect_slices

from skill_library import SkillLibrary

fa = FrankaArm()
skills = SkillLibrary(fa)

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
