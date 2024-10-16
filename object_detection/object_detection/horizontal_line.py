#!/usr/bin/python3

import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import torch
import pathlib
from ultralytics import YOLO
import math
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import easyocr
from geometry_msgs.msg import Twist
import time
from skimage import data, transform

########################################################################################################################
########################################################################################################################
########################################################################################################################

# import cv2
# from operator import itemgetter
# from glob import glob
# import matplotlib.pyplot as pltpaper = cv2.imread('./Photos/book.jpg')
# # Coordinates that you want to Perspective Transform
# pts1 = np.float32([[219,209],[612,8],[380,493],[785,271]])
# # Size of the Transformed Image
# pts2 = np.float32([[0,0],[500,0],[0,400],[500,400]])
# for val in pt1:
#     cv2.circle(paper,(val[0],val[1]),5,(0,255,0),-1)
# M = cv2.getPerspectiveTransform(pt1,pts2)
# dst = cv2.warpPerspective(paper,M,(500,400))
# plt.imshow(dst)

########################################################################################################################
########################################################################################################################
##########################################################################################################################


pi = 3.14
tilt_angle = 0.296706
camera_common_radius = 0.03
camera_subtended_angle = 1.91986
caster_pos_y = 0.45
cone_radius = 0.1
rot_angle = -1*(-pi/2 + camera_subtended_angle/2)

def cal_rel_cam(v, tilt_angle, rot_angle):
    rx = np.asarray(
            [[1, 0, 0],
            [0, np.cos(0), -np.sin(0)],
            [0, np.sin(0), np.cos(0)]]
        )
        
    ry = np.asarray(
        [[np.cos(tilt_angle), 0 ,np.sin(tilt_angle)],
        [0, 1, 0],
        [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]]
    )

    rz = np.asarray(
        [[np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle), np.cos(rot_angle), 0],
        [0, 0, 1]]
    )
    
    y, z, x = v[0], v[1], v[2]
    v = np.dot(rx, np.array([x, y, z]))
    v = np.dot(ry, v)
    v = np.dot(rz, v)

    return v

b = tilt_angle
g = rot_angle
a = 0

x0 = 0
y0 = 0
z0 = 0

T = np.array(
        [[ math.cos(b)*math.cos(g), (math.sin(a)*math.sin(b)*math.cos(g) + math.cos(a)*math.sin(g)), (math.sin(a)*math.sin(g) - math.cos(a)*math.sin(b)*math.cos(g)), x0],
        [-math.cos(b)*math.sin(g), (math.cos(a)*math.cos(g) - math.sin(a)*math.sin(b)*math.sin(g)), (math.sin(a)*math.cos(g) + math.cos(a)*math.sin(b)*math.sin(g)), y0],
        [math.sin(b), -math.sin(a)*math.cos(b), math.cos(a)*math.cos(b), z0],
        [ 0, 0, 0, 1]]
    )

class LineDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection')

        self.camera1_orientation = -rot_angle-pi/2
        self.camera2_orientation = -rot_angle

        self.rgb_camera1 = np.empty( shape=(0, 0) )
        self.rgb_camera2 = np.empty( shape=(0, 0) )
        
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.subscription_cam1info = self.create_subscription(
            CameraInfo,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/depth/camera_info',
            '/depth_camera1/camera_info',
            lambda msg: self.caminfo_callback(msg, camera_name = "camera1"),
            10)
        
        self.subscription_camera1_depth = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            '/depth_camera1/depth/image_raw',
            lambda msg: self.depth_callback(msg, camera_name = "camera1"),
            10)
        
        
        self.subscription_camera1 = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/rgb/image_rect_color',
            '/camera1/image_raw',
            lambda msg: self.image_callback(msg, camera_name = "camera1"),
            10)
        
        self.subscription_cam1info = self.create_subscription(
            CameraInfo,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/depth/camera_info',
            '/depth_short_1_camera/camera_info',
            lambda msg: self.caminfo_callback(msg, camera_name = "camera2"),
            10)
        
        self.subscription_camera1_depth = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            '/depth_short_1_camera/depth/image_raw',
            lambda msg: self.depth_callback(msg, camera_name = "camera2"),
            10)
        
        self.subscription_camera1 = self.create_subscription(
            Image,
            # '/zed/zed_node/depth/depth_registered',
            # '/zed/zed_node/rgb/image_rect_color',
            '/short_1_camera/image_raw',
            lambda msg: self.image_callback(msg, camera_name = "camera2"),
            10)

        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        self.publisher_ = self.create_publisher(
            Image,
            '/cones',
            10)
            
        self.detect_pub = self.create_publisher(Pose, 'destination_pose',10)
        self.dual_camera = True
        self.dual_camera_count = 0

        self.caminfo1 = None
        self.depth_img1 = None

        self.caminfo2 = None
        self.depth_img2 = None

        self.bridge = CvBridge()

    def odom_callback(self, msg):
        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y
        self.z_pos = msg.pose.pose.position.z
        orientation = msg.pose.pose.orientation

        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.orientation = yaw
        
    def depth_callback(self, msg, camera_name):
        if camera_name == "camera1":
            self.depth_img1 = self.bridge.imgmsg_to_cv2(msg)
        else:
            self.depth_img2 = self.bridge.imgmsg_to_cv2(msg)
        
    def caminfo_callback(self, msg, camera_name):
        if camera_name == "camera1":
            self.caminfo1 = msg
        else:
            self.caminfo2 = msg

    def image_callback(self, data, camera_name):
        # print(data)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return
        
        if camera_name == "camera1":
            if self.depth_img1 is not None and self.caminfo1 is not None:
                final = self.detect_lines(cv_image, camera_name = camera_name)

                try:
                    self.publisher_.publish(self.bridge.cv2_to_imgmsg(final))
                except CvBridgeError as e:
                    self.get_logger().error(str(e))
        else:
            if self.depth_img2 is not None and self.caminfo2 is not None:
                final = self.detect_lines(cv_image, camera_name = camera_name)

                try:
                    self.publisher_.publish(self.bridge.cv2_to_imgmsg(final))
                except CvBridgeError as e:
                    self.get_logger().error(str(e))

    def detect_lines(self, image, camera_name):
        # path='images/lines.png'
        # image = cv.imread(path)

        # dst = cv.Canny(image, 50, 200, None, 3)
        # linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        # if linesP is not None:
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]

        #         #here l contains x1,y1,x2,y2  of your line
        #         #so you can compute the orientation of the line 
        #         p1 = np.array([l[0],l[1]])
        #         p2 = np.array([l[2],l[3]])

        #         p0 = np.subtract( p1,p1 ) #not used
        #         p3 = np.subtract( p2,p1 ) #translate p2 by p1

        #         angle_radiants = math.atan2(p3[1],p3[0])
        #         angle_degree = angle_radiants * 180 / math.pi

        #         print("line degree", angle_degree)

        #         if 0 < angle_degree < 15 or 0 > angle_degree > -15 :
        #             cv.line(image,  (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

        rx = np.asarray(
            [[1, 0, 0],
            [0, np.cos(0), -np.sin(0)],
            [0, np.sin(0), np.cos(0)]]
        )
        
        ry = np.asarray(
            [[np.cos(tilt_angle), 0 ,np.sin(tilt_angle)],
            [0, 1, 0],
            [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]]
        )

        rz = np.asarray(
            [[np.cos(rot_angle), -np.sin(rot_angle), 0],
            [np.sin(rot_angle), np.cos(rot_angle), 0],
            [0, 0, 1]]
        )

        # S = np.array([[1, 0, 0],
        #       [0, 1.3, 0],
        #       [0, 1e-3, 1]])
        
        # Combined rotation matrix
        R = rx  @ ry

        # Camera intrinsic matrix (assumed)
        focal_length = 280.08366421465684  # example focal length
        K = np.array([[focal_length, 0, 400.5],
                    [0, focal_length, 400.5],
                    [0, 0, 1]])

        # Homography matrix for perspective transformation
        H = K @ R @ np.linalg.inv(K)

        rows, cols, _ = image.shape
        # Apply the perspective warp using warpPerspective
        warped_image = cv.warpPerspective(image, H, (cols, rows))
        self.dual_camera = True

        if camera_name == "camera1":
            self.rgb_camera1 = image
            self.dual_camera_count += 1
        else:
            self.rgb_camera2 = image
            self.dual_camera_count -= 1

        self.display_image()

        return image

    def display_image(self):
        if(self.dual_camera == True and self.rgb_camera1.shape[-1] == 3 and self.rgb_camera2.shape[-1] == 3 and self.dual_camera_count == 0):
            height, width, channels = self.rgb_camera1.shape

            # Define the thickness of the vertical black line
            line_thickness = 10  # Adjust thickness as needed

            # Create a black vertical line with the same height and desired thickness
            # The line is a numpy array filled with zeros (black)
            black_line = np.zeros((height, line_thickness, channels), dtype=np.uint8)

            numpy_horizontal_concat = np.concatenate((self.rgb_camera1, black_line, self.rgb_camera2), axis=1)


            # Create a blank space above the images to place the text
            text_space_height = 50  # Height of the space for text
            text_space = np.zeros((text_space_height, numpy_horizontal_concat.shape[1], channels), dtype=np.uint8)

            # Add text to the text_space
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White text
            thickness = 2

            # Calculate the position of "Image 1" and "Image 2"
            # Text will be centered horizontally in each half
            text_size_image1 = cv.getTextSize("Left Camera", font, font_scale, thickness)[0]
            text_x_image1 = (width - text_size_image1[0]) // 2  # Centering text for Image 1
            text_y = text_space_height // 2 + text_size_image1[1] // 2  # Center vertically in the text space

            text_size_image2 = cv.getTextSize("Right Camera", font, font_scale, thickness)[0]
            text_x_image2 = width + line_thickness + (width - text_size_image2[0]) // 2  # Centering text for Image 2

            # Put the text on the blank space above the images
            cv.putText(text_space, "Left Camera", (text_x_image1, text_y), font, font_scale, color, thickness)
            cv.putText(text_space, "Right Camera", (text_x_image2, text_y), font, font_scale, color, thickness)

            # Concatenate the text space on top of the images
            final_image = np.concatenate((text_space, numpy_horizontal_concat), axis=0)

            cv.imshow("IMAGE", final_image)
            cv.waitKey(1)
        
        return

def main(args=None):
    rclpy.init(args=args)
    node = LineDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
