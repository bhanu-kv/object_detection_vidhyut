#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import torch
import pathlib
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection')
        
        # path = '/home/Object_Detection/runs/detect/train/weights/best.pt'
        self.model = torch.hub.load('.', 'custom', path='best.pt, force_reload=True, source='local', device="gpu")
        # self.model = YOLO("best.pt")

        self.classNames = ["Barrel", "No-Turns", "One-Way", "Person", "Road-Closed", "Stop-Sign", "Tire"]
        
        self.subscription = self.create_subscription(
            Image,
            '/zed2i/zed_node/rgb/image_rect_color',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(
            Image,
            '/objects',
            1)
        self.bridge = CvBridge()

    def object_detection(self, rgb):
        # Add parameter settings or modifications as needed
        rgb = cv.flip(rgb, 0)

        results = self.model(rgb)
        im_rgb = cv.cvtColor(results.imgs[0], cv.COLOR_BGR2RGB)

        for r in results:
            boxes = r.boxes
    
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
    
                # put box in cam
                cv.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
    
                # class name
                cls = int(box.cls[0])
                print("Class name -->", self.classNames[cls])
    
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
    
                cv.putText(img_rgb, classNames[cls], org, font, fontScale, color, thickness)

        cv.imshow("IMAGE", img_rgb)
        cv.waitKey(1)
        
        return img_rgb

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        final = self.object_detection(cv_image)

        try:
            self.publisher_.publish(self.bridge.cv2_to_imgmsg(final))
        except CvBridgeError as e:
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
