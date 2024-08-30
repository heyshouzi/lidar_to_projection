import rospy
import cv2
import os
import numpy as np
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField,Image
from std_msgs.msg import Header
from livox_ros_driver2.msg import CustomMsg
from livox_ros_driver2.msg import CustomPoint 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


# 相机内参矩阵
K = np.array([[333.1000665, 0, 330.691479],
              [0, 335.804961, 255.718311],
              [0, 0, 1]])

# 相机外参矩阵
R = np.array([[0, 0,-1],
              [0, 1, 0],
              [1, 0, 0]])

# 绕z轴旋转90度旋转矩阵
Z = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0,0, 1]]) 


T = np.array([0.05,0.1,-0.46])

# 创建 CvBridge 对象
bridge = CvBridge()

def point_cloud_callback(lidar_points):
    if lidar_points is None:
        print("Point cloud message is empty.")
        return
    try:
        """
        将雷达点云从世界坐标系转换到图像坐标系
        
        :param lidar_points: 雷达点云数据 (Nx3)
        :param K: 相机内参矩阵 (3x3)
        :param R: 旋转矩阵 (3x3)
        :param T: 平移向量 (3,)
        :return: 图像坐标系中的点 (Nx2)
        """

        # 将雷达点云数据转为齐次坐标 (Nx4)
        lidar_points = ros_numpy.point_cloud2.pointcloud2_to_array(lidar_points)
        # print(f"lidar_points_array.dtype.names: {lidar_points.dtype.names}")
        # print(f"lidar_points size is {lidar_points.shape}")
        # 提取 x, y, z 点
        x = lidar_points['x']
        y = lidar_points['y']
        z = lidar_points['z']

        lidar_points_xyz = np.vstack((x, y, z)).T
        # print(f"lidar_points_xyz size is {lidar_points_xyz.shape}")
         # 将雷达点云数据转为齐次坐标 (Nx4)
        lidar_points_hom = np.hstack((lidar_points_xyz, np.ones((lidar_points_xyz.shape[0], 1))))
        # print(f"lidar_points_hom size is {lidar_points_hom.shape}")
        # 计算相机坐标系中的点 (Nx3)
        camera_points_hom = (Z @ Z @ Z @ R @ lidar_points_hom[:, :3].T).T + T

        # 将相机坐标系中的点转为齐次坐标 (Nx4)
        camera_points_hom = np.hstack((camera_points_hom, np.ones((camera_points_hom.shape[0], 1))))
        # print(f"camera_points_hom size is {camera_points_hom.shape}")
        # 使用内参矩阵将相机坐标系中的点转换到图像坐标系中 (Nx3)
        image_points_hom = (K @ camera_points_hom[:, :3].T).T
        # print(f"image_points_hom size is {image_points_hom.shape}")
        # 计算图像坐标系中的点 (Nx2)
        image_points = image_points_hom[:, :2] / image_points_hom[:, 2].reshape(-1, 1)
        # print(f"image_points size is {image_points.shape}")

        # 检查点是否在图像范围内
        image_width, image_height = 640, 480
        valid_points = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_width) & (image_points[:, 1] >= 0) & (image_points[:, 1] < image_height)

        # 筛选有效的点
        valid_image_points = image_points[valid_points]

        # 绘制点云到图像上
        image_msg = rospy.wait_for_message("/camera/image", Image)
        img = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        if img is not None:
            for point in valid_image_points:
                u, v = int(point[0]), int(point[1])
                cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
              # 发布叠加后的图像
            image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
    except Exception as e:
        print(f"An error occurred: {e}")


# 回调函数，用于处理接收到的消息
def lidar_points_callback(msg):
    filtered_points = []

    # 对每个点进行过滤
    for point in msg.points:
        if point.reflectivity > 100 and 0.5 < point.x < 3.0 and -0.4 < point.z < 1.0:
            filtered_points.append(point)

    # 如果没有过滤到任何点，直接返回
    if not filtered_points:
        return

    # 创建PointCloud2消息
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = msg.header.frame_id

    # 定义PointCloud2的字段
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('reflectivity', 12, PointField.UINT8, 1),
        PointField('tag', 13, PointField.UINT8, 1),
        PointField('line', 14, PointField.UINT8, 1),
    ]

    # 转换为PointCloud2格式
    pc2_data = [
        [point.x, point.y, point.z, point.reflectivity, point.tag, point.line]
        for point in filtered_points
    ]

    pointcloud2_msg = pc2.create_cloud(header, fields, pc2_data)

    # 发布PointCloud2消息
    pointcloud2_publisher.publish(pointcloud2_msg)


def run_rtsp_rostopic():
    rtsp_url = "rtsp://192.168.1.105:8554/test"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        rospy.logerr(f"Unable to open {rtsp_url} stream")
        return

    bridge = CvBridge()
    image_pub = rospy.Publisher('/camera/image', Image, queue_size=10)

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
    print("run_rtsp_rostopic success!")
    

if __name__ == '__main__':
    rospy.init_node('filter_lidar_to_image', anonymous=True)

    # 发布者设置
    global pointcloud2_publisher
    pointcloud2_publisher = rospy.Publisher('/filtered_lidar_points', PointCloud2, queue_size=10)
    # 订阅CustomMsg类型的/lidar_points话题
    rospy.Subscriber('/livox/lidar', CustomMsg, lidar_points_callback)
    print("run_lidar_points_filter success!")

    run_rtsp_rostopic()

    # 订阅雷达点云和相机图像话题
    rospy.Subscriber('/filtered_lidar_points', PointCloud2, point_cloud_callback)
    global image_pub
    image_pub = rospy.Publisher("/projected_image", Image, queue_size=10)
    print("run_lidar_to_image_projection success!")

