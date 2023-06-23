import math
import argparse
import numpy as np
import UdpComms as U
from utils import *
from perception import CameraIntrinsics
from scipy.spatial.transform import Rotation

class SkillLibrary():
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

        self.udp = U.UdpComms(udpIP='172.26.5.54', sendIP='172.26.69.200', portTX=5501, portRX=5500, enableRX=True)

    def move_arm(self, x_pos, y_pos):
        """
        Inputs: x_position, y_position 
        Outputs: None, simply moves robot to target position
        """
        pose = self.fa.get_pose()
        z_pos = 0.17
        pose.translation = np.array([x_pos, y_pos, z_pos])
        self.fa.goto_pose(pose)

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

        pose = self.fa.get_pose()
        starting_rot = pose.rotation
        orig = Rotation.from_matrix(starting_rot)
        orig_euler = orig.as_euler('xyz', degrees=True)
        rot_vec = np.array([0, 0, rotation])
        new_euler = orig_euler + rot_vec
        r = Rotation.from_euler('xyz', new_euler, degrees=True)
        pose.rotation = r.as_matrix()
        self.fa.goto_pose(pose)

        self.fa.apply_effector_forces_along_axis(1.0, 0.5, height, forces=[0.,0.,-force]) # NOTE: original z_force = -75.

    def push(self, start, end):
        """
        """
        pose = self.fa.get_pose()
        z_constant = 0.12 # offset from zero for tool height
        start_pose = pose
        start_pose.translation = np.array([start[0], start[1], z_constant])
        self.fa.goto_pose(start_pose)

        end_pose = pose
        end_pose.translation = np.array([end[0], end[1], z_constant])
        self.fa.goto_pose(end_pose)

    def detect_object(self):
        """
        Queries YOLO/SAM vision system to return all detected fruit/vegetable objects in the scene.
        Upon completion, this function will return: 
            object classes
            object positions
            total number of detected objects
        
        TODO: modify funtion to return all necessary features of detected objects
        """
        no_obj = True
        while no_obj:
            try: 
                message = self.udp.ReadReceivedData()
                if message is None:
                    continue

                print(message)

                x = float(message.split(',')[0].split('[')[2])
                y = float(message.split(',')[1])
                z = float(message.split(',')[2].split(']')[0])

                robot_pose = self.fa.get_pose()
                com = get_object_center_point_in_world_realsense_3D_camera_point(np.array([x,y,z]), self.realsense_intrinsics, self.realsense_to_ee_transform, robot_pose)

                # --------- FINAL 3D POINT IN FRANKA WORLD FRAME ----------
                com = np.array([com[0], com[1] + 0.02, com[2] + 0.02]) # should be the x,y,z position in robot frame
                print("COM: ", com)
                # robot_pose.translation = np.array([com[0], com[1], com[2] + 0.10])

                no_obj = False
                
            except:
                print("no message")
        
        return np.array([com[0], com[1]])

    def detect_slices(self):
        """
        """
        pass

    def hold_object(x_pos, y_pos):
        """
        """
        pass