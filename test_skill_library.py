
class SkillLibrary():
    def __init__(self, fa):
        pass

    def move_arm(self, x_pos, y_pos):
        """
        Inputs: x_position, y_position 
        Outputs: None, simply moves robot to target position
        """
        pass

    # def goto_grasp(fa, x, y, z, rx, ry, rz, d):
    #     """
    #     Parameterize a grasp action by the position [x,y,z] Euler angle rotation [rx,ry,rz], and width [d] of the gripper.
    #     This function was designed to be used for clay moulding, but in practice can be applied to any task.

    #     :param fa:  franka robot class instantiation
    #     """
    #     pose = fa.get_pose()
    #     starting_rot = pose.rotation
    #     orig = Rotation.from_matrix(starting_rot)
    #     orig_euler = orig.as_euler('xyz', degrees=True)
    #     rot_vec = np.array([rx, ry, rz])
    #     new_euler = orig_euler + rot_vec
    #     r = Rotation.from_euler('xyz', new_euler, degrees=True)
    #     pose.rotation = r.as_matrix()
    #     pose.translation = np.array([x, y, z])

    #     fa.goto_pose(pose)
    #     fa.goto_gripper(d, force=60.0)
    #     time.sleep(3)

    def cut(self, height, rotation, force):
        """
        assume rotation is in degrees
        """
        # cur_joints = self.fa.get_joints()
        # cur_joints[6] += math.radians(rotation)
        # self.fa.goto_joints(cur_joints)

        pass

    def push(self, start, end):
        """
        """
        pass

    def detect_object(self):
        """
        Queries YOLO/SAM vision system to return all detected fruit/vegetable objects in the scene.
        Upon completion, this function will return: 
            object classes
            object positions
            total number of detected objects
        
        TODO: modify funtion to return all necessary features of detected objects
        """
        pass

    def detect_slices(self):
        """
        """
        pass

    def hold_object(x_pos, y_pos):
        """
        """
        pass