import cv_bridge
import numpy as np
import cv2

bridge = cv_bridge.CvBridge()
image = np.zeros((480, 640, 3), dtype=np.uint8)
image_message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
print("cv_bridge test successful")
