#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image

def main():
    rospy.init_node('rtsp_stream_publisher', anonymous=True)
    rtsp_url = "rtsp://192.168.1.105:8554/test"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        rospy.logerr(f"Unable to open {rtsp_url} stream")
        return

    bridge = CvBridge()
    image_pub = rospy.Publisher('/wide_angle_camera_front/image_color_rect_resize', Image, queue_size=10)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to grab frame")
            continue

        try:
            ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
            image_pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


    cap.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception")
    pass