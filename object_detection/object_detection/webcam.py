import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time

class StopBot(Node):
    def __init__(self, cam):
        super().__init__('stop_bot')

        # Read Camera and Frame from Webcam
        self.cam = cam
        self.rval, self.frame = self.cam.read()

        # Set Horizontal Line detected to False
        self.horizontal_line = False
        self.object_detected = False

        # Set Minimum value of pixel for White color
        self.thresh = 200

        # Subscribe to Horizontal Line Detection topic from ZED Camera
        self.line_subscription_flag = self.create_subscription(Bool, '/hor_flag', self.hor_callback, 10)
        self.line_subscription_flag = self.create_subscription(Bool, '/object_flag', self.object_callback, 10)

        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        self.stop_bot_func()
    
    # Publish 0 Velocity to stop
    def publish_vel(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0

        self.vel_publisher.publish(vel_msg)

    def stop_bot_func(self):
        # White Image in Frame is True
        while self.rval:
            self.rval, self.frame = self.cam.read()

            # Gray Scale --> Ranging --> Gaussian Blur
            im_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            im_bw = cv2.inRange(im_gray, 120, 255)
            im_bw = cv2.GaussianBlur(im_bw, (7,7), 0)

            # get the (largest) contour
            contours = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            # If contour detected
            if contours:
                # Get the biggest contour
                big_contour = max(contours, key=cv2.contourArea)

                # draw white filled contour on black background
                image = np.zeros_like(im_bw)

                # Draw Contour
                cv2.drawContours(image, [big_contour], 0, (255,255,255), cv2.FILLED)

                # Detect if contour is a rectange and get its BB and width
                rect = cv2.minAreaRect(big_contour)
                (x, y), (w, h), angle = rect
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # If rectangle occupies more than 80% of width stop the bot
                if w/len(image[0]) > 0.8 and self.horizontal_line == True and self.object_detected == True:
                    t_end = time.time() + 5

                    while time.time() < t_end:
                        self.publish_vel()
                        cv2.drawContours(image, [box] , 0, (255,0,0), 10)
                        cv2.imshow("Webcam", image)
                    
                    self.horizontal_line = False
                    break

                cv2.imshow("Webcam", self.frame)

                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    break
    
    def object_callback(self, msg):
        # Object Detected
        if msg.data == True:
            self.object_detected = True
    
    def hor_callback(self, msg):
        # Horizontal Line Detected
        if msg.data == True:
            self.horizontal_line = True
            self.stop_bot_func()


def main(args=None):
    rclpy.init(args=args)
    
    cv2.namedWindow("Webcam")
    cam = cv2.VideoCapture(0)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
    
    if rval:
        node = StopBot(cam)
        rclpy.spin(node)
    
    rclpy.shutdown()
    cv2.destroyWindow("Webcam")


if __name__ == '__main__':
    main()