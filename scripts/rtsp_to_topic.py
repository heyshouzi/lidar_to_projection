#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def main():
    rospy.init_node('rtsp_stream_publisher', anonymous=True)
    rtsp_url = "rtsp://192.168.1.105:8554/test"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        rospy.logerr(f"Unable to open {rtsp_url} stream")
        return

    bridge = CvBridge()
    image_pub = rospy.Publisher('/camera_front/color/image_rect', Image, queue_size=10)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("[rtsp_to_ros]:Failed to grab image frame")
            continue

        try:
            # 将OpenCV图像转换为ROS Image消息
            ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
            # 设置Image消息的时间戳
            ros_image.header.stamp = rospy.Time.now()
            # 发布ROS Image消息
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
