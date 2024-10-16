# #!/usr/bin/python3

# import rclpy
# import easyocr
# from PIL import Image
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
# import cv2 as cv
# import torch
# import pathlib
# from ultralytics import YOLO
# import math
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Twist
# import time

# class ObjectDetectionNode(Node):
#     def __init__(self):
#         super().__init__('object_detection')
        
#         self.reader = easyocr.Reader(['en'], gpu = True)
#         self.model = YOLO("/home/bhanu/abhiyaan/object_detection/src/object_detection/best.pt")

#         # self.classNames = ["Barrel", "No-Turns", "One-Way", "Person", "Road-Closed", "Stop-Sign", "Tire"]
#         self.classNames = ["barrel", "people", "stop"]
#         self.flag = True
#         self.start_time = None        
#         self.object_coordinates=[]
#         self.x_stop = None
#         self.y_stop = None
#         self.z_stop = None

#         self.subscription_odom = self.create_subscription(
#             Pose,
#             '/odom',
#             self.odom_callback,
#             10)
        
#         self.subscription_rgb = self.create_subscription(
#             Image,
#             # '/zed/zed_node/depth/depth_registered',
#             # '/zed/zed_node/rgb/image_rect_color',
#             '/camera/image_raw',
#             self.image_callback,
#             10)
            
#         self.subscription_caminfo = self.create_subscription(
#             CameraInfo,
#             # '/zed/zed_node/depth/depth_registered',
#             # '/zed/zed_node/depth/camera_info',
#             '/depth_camera/camera_info',
#             self.caminfo_callback,
#             10)
            
#         self.subscription_depth = self.create_subscription(
#             Image,
#             # '/zed/zed_node/depth/depth_registered',
#             '/depth_camera/depth/image_raw',
#             self.depth_callback,
#             10)
        
#         # self.subscription_cam1info = self.create_subscription(
#         #     CameraInfo,
#         #     # '/zed/zed_node/depth/depth_registered',
#         #     # '/zed/zed_node/depth/camera_info',
#         #     '/depth_camera1/camera_info',
#         #     self.caminfo_callback,
#         #     10)
        
#         # self.subscription_camera1_depth = self.create_subscription(
#         #     Image,
#         #     # '/zed/zed_node/depth/depth_registered',
#         #     '/depth_camera1/depth/image_raw',
#         #     self.depth_callback,
#         #     10)
        
#         # self.subscription_camera1 = self.create_subscription(
#         #     Image,
#         #     # '/zed/zed_node/depth/depth_registered',
#         #     # '/zed/zed_node/rgb/image_rect_color',
#         #     '/camera1/image_raw',
#         #     lambda msg: self.image_callback(msg, camera_name = "camera1"),
#         #     10)

#         self.object_publisher = self.create_publisher(
#             Image,
#             '/objects',
#             10)
        
#         self.vel_publisher = self.create_publisher(
#             Twist,
#             '/cmd_vel',
#             10)
            
#         self.detect_pub = self.create_publisher(Pose, 'destination_pose',10)
#         self.caminfo = None
#         self.depth_img = None
#         self.bridge = CvBridge()
        
#     def odom_callback(self):
#         if ((len(self.object_coordinates) != 0)):
#             # print("-----------------IDHAR AYA------------------")
#             # print(self.object_coordinates)

#             # vel_msg = Twist()
#             # vel_msg.linear.x = 0.0
#             # vel_msg.linear.y = 0.0
#             # vel_msg.linear.z = 0.0
#             # vel_msg.angular.x = 0.0
#             # vel_msg.angular.y = 0.0
#             # vel_msg.angular.z = 0.0

#             # self.vel_publisher.publish(vel_msg)

#             for i in range(len(self.object_coordinates)):
            
#                 self.x_stop = self.object_coordinates[i][2]
#                 self.y_stop = self.object_coordinates[i][3]
#                 self.z_stop = self.object_coordinates[i][4]

#                 # if (self.x_stop**2 + self.z_stop**2 < 3):
#                 vel_msg = Twist()
#                 vel_msg.linear.x = 0.0
#                 vel_msg.linear.y = 0.0
#                 vel_msg.linear.z = 0.0
#                 vel_msg.angular.x = 0.0
#                 vel_msg.angular.y = 0.0
#                 vel_msg.angular.z = 0.0

#                 self.vel_publisher.publish(vel_msg)
        
#                 # if (math.sqrt((self.x_stop)**2 + (self.z_stop)**2) <= 3):
#                 #     if (self.start_time != None and time.time() - self.start_time > 5):
#                 #         self.start_time = None
#                 #         self.flag = True
#                 #         self.x_stop = None
#                 #         self.y_stop = None
#                 #         self.z_stop = None

#                 #     elif (self.start_time == None):
#                 #         vel_msg = Twist()
#                 #         vel_msg.linear.x = 0.0
#                 #         vel_msg.linear.y = 0.0
#                 #         vel_msg.linear.z = 0.0
#                 #         vel_msg.angular.x = 0.0
#                 #         vel_msg.angular.y = 0.0
#                 #         vel_msg.angular.z = 0.0

#                 #         self.vel_publisher.publish(vel_msg)

#                 #     elif (time.time() - self.start_time < 5):
#                 #         vel_msg = Twist()
#                 #         vel_msg.linear.x = 0.0
#                 #         vel_msg.linear.y = 0.0
#                 #         vel_msg.linear.z = 0.0
#                 #         vel_msg.angular.x = 0.0
#                 #         vel_msg.angular.y = 0.0
#                 #         vel_msg.angular.z = 0.0

#                 #         self.vel_publisher.publish(vel_msg)

#     def depth_callback(self, msg):
#         self.depth_img = self.bridge.imgmsg_to_cv2(msg)
        
#     def caminfo_callback(self, msg):
#         self.caminfo = msg
        
#     def object_detection(self, rgb):
#         # Add parameter settings or modifications as needed
#         # rgb = cv.flip(rgb, 0)
        
#         results = self.model(rgb)

#         cl_box = []
#         camera_factor = self.caminfo.k[-1]
#         camera_cx = self.caminfo.k[2]
#         camera_cy = self.caminfo.k[5]
#         camera_fx = self.caminfo.k[0]
#         camera_fy = self.caminfo.k[4]

#         self.object_coordinates=[]

#         for r in results:
#             boxes = r.boxes
    
#             for box in boxes:
#                 # bounding box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

#                 cl_box.append(((x1+x2)/2, (y1+y2)/2))

#                 # put box in cam
#                 if(self.classNames[int(box.cls[0])] == 'stop'):
#                     txt = self.reader.readtext(rgb[int(y1):int(y2), int(x1):int(x2)])
#                     if(len(txt)==0):
#                         continue
#                     elif(self.reader.readtext(rgb[int(y1):int(y2), int(x1):int(x2)])[0][-2].lower() != 'stop'):
#                         continue
                
#                 confidence = math.ceil((box.conf[0]*100))/100
                
#                 if(confidence < 0.45):
#                     continue
                    
#                 cv.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 u = int((x1+x2)/2)
#                 v = int((y1+y2)/2)
                                
#                 z = self.depth_img[v][u] / camera_factor
#                 x = (u - camera_cx) * z / camera_fx
#                 y = (v - camera_cy) * z / camera_fy

#                 # confidence
#                 confidence = math.ceil((box.conf[0]*100))/100
#                 # print("Confidence --->",confidence)
#                 # print(f'point is at:({x} , {y}, {z})')
#                 # class name
#                 cls = int(box.cls[0])
#                 # print("Class name -->", self.classNames[cls])
                
#                 if self.flag == True:
#                     self.object_coordinates.append([self.classNames[cls], confidence, x, y, z])
                    
#                     # if (self.classNames[cls] == "people" or self.classNames[cls] == "stop"):
#                     #     self.flag = False

#                     # self.odom_callback()

#                 # object details
#                 org = [x1, y1]
#                 font = cv.FONT_HERSHEY_SIMPLEX
#                 fontScale = 1
#                 color = (255, 0, 0)
#                 thickness = 2
    
#                 cv.putText(rgb, self.classNames[cls], org, font, fontScale, color, thickness)

#         for i in range(len(self.object_coordinates)):
#             if(self.object_coordinates[i][0] == 'stop' or self.object_coordinates[i][0] == 'people'):
#                 self.odom_callback()
        
#         # print(self.flag)

#         cv.imshow("IMAGE", rgb)
#         cv.waitKey(1)
            
#         return self.depth_img

#     def image_callback(self, data):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#         except CvBridgeError as e:
#             self.get_logger().error(str(e))
#             return
#         if self.depth_img is not None and self.caminfo is not None:
#             final = self.object_detection(cv_image )

#             try:
#                 self.object_publisher.publish(self.bridge.cv2_to_imgmsg(final))
#             except CvBridgeError as e:
#                 self.get_logger().error(str(e))

# def main(args=None):
#     rclpy.init(args=args)
#     node = ObjectDetectionNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
