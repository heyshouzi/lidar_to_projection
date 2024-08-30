import rospy
import cv2
import numpy as np
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

class LidarImageProjector:
    def __init__(self,image_pub):
        # Initialize camera intrinsic (K) and extrinsic (R, T) parameters
        self.K = np.array([[333.1000665, 0, 330.691479],
                           [0, 335.804961, 255.718311],
                           [0, 0, 1]])
        
        self.R = np.array([[0, -1, 0],
                           [0, 0, -1],
                           [1, 0, 0]])

        self.T = np.array([0.05, 0.1, -0.46])
        
        # Create CvBridge object for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        self.image_pub = image_pub

    def project_pointcloud_to_image(self, img_msg, lidar_points):
        """ Projects LiDAR points onto an image plane. """
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
            camera_points_hom = (self.R @ lidar_points_hom[:, :3].T).T + self.T

            # 将相机坐标系中的点转为齐次坐标 (Nx4)
            camera_points_hom = np.hstack((camera_points_hom, np.ones((camera_points_hom.shape[0], 1))))
            # print(f"camera_points_hom size is {camera_points_hom.shape}")
            # 使用内参矩阵将相机坐标系中的点转换到图像坐标系中 (Nx3)
            image_points_hom = (self.K @ camera_points_hom[:, :3].T).T
            # print(f"image_points_hom size is {image_points_hom.shape}")
            # 计算图像坐标系中的点 (Nx2)
            image_points = image_points_hom[:, :2] / image_points_hom[:, 2].reshape(-1, 1)
            # print(f"image_points size is {image_points.shape}")

            # 检查点是否在图像范围内
            image_width, image_height = 640, 480
            valid_points = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_width) & (image_points[:, 1] >= 0) & (image_points[:, 1] < image_height)

            # 筛选有效的点
            valid_image_points = image_points[valid_points]

            # 绘制点云到图
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            if img is not None:
                for point in valid_image_points:
                    u, v = int(point[0]), int(point[1])
                    # print(f"u is {u},v is {v}")
                    cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
                # 发布叠加后的图像
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            print(f"An error occurred: {e}")
    # 过滤雷达点云数据
    def filter_lidar_points(self,msg):
        # 将过滤后的点云转换回 ROS 点云消息
        header = msg.header
        header.frame_id = msg.header.frame_id
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z","intensity"), skip_nans=True)))
        # print(f"points shape is {points.shape}")
        # 通过率滤波
        mask = (points[:, 0] >= 0) & (points[:, 0] <= 5.5) & (points[:, 1] >= -3.0) & (points[:, 1] <= 3.0) & (points[:,3] >= 150)
        filtered_points = points[mask]
        # print(f"filtered_points shape is {filtered_points.shape}")
        # 如果没有过滤到任何点，直接返回
        if  filtered_points is None:
            print("No points were filtered.")
            return
        filtered_points = filtered_points[:, :3]
        pointcloud2_msg = pc2.create_cloud_xyz32(header, filtered_points)
        return pointcloud2_msg
    def callback(self,image_msg, pointcloud_msg):
        #首先过滤点云
        filter_msg = self.filter_lidar_points(pointcloud_msg)
        # 然后将点云投影到像上
        if filter_msg is not None:
            self.project_pointcloud_to_image(image_msg,filter_msg)

def main():
    rospy.init_node('lidar_image_projection', anonymous=True)

    image_pub = rospy.Publisher('/projected_image', Image, queue_size=10)
    # Create the LidarImageProjector object
    projector = LidarImageProjector(image_pub)
    
    # Setup publishers and subscribers
    image_sub = Subscriber('/camera_front/color/image_rect', Image)
    lidar_sub = Subscriber('/lidar_points', PointCloud2)


    # Approximate time synchronizer for image and LiDAR data
    ats = ApproximateTimeSynchronizer([image_sub, lidar_sub], queue_size=10, slop=0.5)
    ats.registerCallback(projector.callback)
    rospy.spin()


if __name__ == "__main__":
    main()

    
