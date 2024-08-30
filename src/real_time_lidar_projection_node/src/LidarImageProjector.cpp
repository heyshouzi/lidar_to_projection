#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class LidarImageProjector {
public:
    LidarImageProjector(ros::Publisher image_pub) : image_pub_(image_pub) {
        // Initialize camera intrinsic (K) and extrinsic (R, T) parameters
        K_ << 333.1000665, 0, 330.691479,
              0, 335.804961, 255.718311,
              0, 0, 1;

        R_ << 0, -1, 0,
              0, 1, -1,
              1, 0, 0;

        T_ << 0.05, 0.1, -0.46;
    }



    void projectPointCloudToImage(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::PointCloud2ConstPtr& lidar_points_msg) {
        try {
            // Convert ROS image to OpenCV image
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
            cv::Mat image = cv_ptr->image;

            // Convert PointCloud2 message to PCL point cloud
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*lidar_points_msg, *cloud);

            for (const auto& point : cloud->points) {
                Eigen::Vector3d point3d(point.x, point.y, point.z);
                Eigen::Vector3d point_cam = R_ * point3d + T_;
                Eigen::Vector3d point_img = K_ * point_cam;

                int u = static_cast<int>(point_img[0] / point_img[2]);
                int v = static_cast<int>(point_img[1] / point_img[2]);

                if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                    cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
                }
            }      
            // Publish the modified image
            sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(img_msg->header, "bgr8", image).toImageMsg();
            image_pub_.publish(output_msg);
        } catch (const std::exception& e) {
            ROS_ERROR("An error occurred: %s", e.what());
        }
    }

    void filterLidarPoints(const sensor_msgs::PointCloud2ConstPtr& msg, sensor_msgs::PointCloud2& filtered_msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PassThrough<pcl::PointXYZI> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0, 1.5);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
        pass.filter(*cloud_filtered);

        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.5, 0.5);
        pass.filter(*cloud_filtered);

        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("intensity");
        pass.setFilterLimits(2, std::numeric_limits<float>::max());
        pass.filter(*cloud_filtered);

        pcl::toROSMsg(*cloud_filtered, filtered_msg);
    }

    void callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg) {
        sensor_msgs::PointCloud2 filtered_msg;
        filtered_msg.header =pointcloud_msg->header;
        filterLidarPoints(pointcloud_msg, filtered_msg);
        projectPointCloudToImage(image_msg, boost::make_shared<sensor_msgs::PointCloud2>(filtered_msg));
    }

private:
    ros::Publisher image_pub_;
    Eigen::Matrix3d K_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d T_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_image_projection");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/projected_image", 10);

    LidarImageProjector projector(image_pub);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera_front/color/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub(nh, "/lidar_points", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, lidar_sub);
    sync.registerCallback(boost::bind(&LidarImageProjector::callback, &projector, _1, _2));

    ros::spin();
    return 0;
}
