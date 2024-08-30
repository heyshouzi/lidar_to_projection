#include <iostream>
#include <thread>
#include <queue>
#include <future>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <real_time_lidar_projection_node/CustomMsg.h>
#include <real_time_lidar_projection_node/CustomPoint.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

class RealTimeLidarProjection {
public:
    RealTimeLidarProjection() : stop_rtsp_thread(false), stop_sync_thread(false) {
        picture_queue = std::queue<std::pair<cv::Mat, uint64_t>>();
        lidar_points_queue = std::queue<std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, uint64_t>>();
    }

    ~RealTimeLidarProjection() {
        stop_rtsp_thread = true;
        stop_sync_thread = true;

        if (sync_thread_future.valid()) {
            sync_thread_future.wait();
        }
    }
    void run() {
    ros::NodeHandle nh;
    img_pub = nh.advertise<sensor_msgs::Image>("/projected_image", 1);
    ros::Subscriber sub = nh.subscribe("/livox/lidar", 10, &RealTimeLidarProjection::filter_lidar_callback, this);
    ros::AsyncSpinner spinner(2);
     // 启动 AsyncSpinner
    spinner.start();
    sync_timestamp();
    pull_rtsp();
    }   

private:
    std::queue<std::pair<cv::Mat, uint64_t>> picture_queue;
    std::queue<std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, uint64_t>> lidar_points_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::future<void> sync_thread_future;
    ros::Publisher img_pub;

    const size_t max_queue_size = 20;
    const int64_t timestamp_tolerance = 5000000;  // 5ms
    bool stop_rtsp_thread;
    bool stop_sync_thread;

    void pull_rtsp() {
        
            std::string rtsp_url = "rtsp://192.168.1.105:8554/test";
            cv::VideoCapture cap(rtsp_url);

            if (!cap.isOpened()) {
                std::cerr << "Failed to open RTSP stream" << std::endl;
                return;
            }

            while (!stop_rtsp_thread) {
                if (!cap.isOpened()) {
                    cap.open(rtsp_url);
                    if (!cap.isOpened()) {
                        std::cerr << "Failed to re-open RTSP stream" << std::endl;
                        continue;
                    }
                }

                cv::Mat frame;
                cap >> frame;
                if (frame.empty()) {
                    std::cerr << "RTSP stream frame is empty" << std::endl;
                    continue;
                }

                uint64_t timestamp = ros::Time::now().toNSec();
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    picture_queue.push({frame, timestamp});
                    cv.notify_all(); // Notify sync_thread that new data is available

                    if (picture_queue.size() > max_queue_size) {
                        picture_queue.pop();
                    }
                }
            }
    }

    void filter_lidar_callback(const real_time_lidar_projection_node::CustomMsg::ConstPtr& msg) {
        try {
            uint64_t timestamp = msg->header.stamp.toNSec(); // 使用LiDAR消息的时间戳
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_msg(new pcl::PointCloud <pcl::PointXYZ>);
            // 过滤点云数据
            for (unsigned int i = 0; i < msg->point_num; ++i) {
                pcl::PointXYZ pointxyz;
                if(msg->points[i].reflectivity > 150  && 0 < msg->points[i].x && msg->points[i].x < 6.0 && -3.0 < msg->points[i].y && msg->points[i].y < 3.0)
                pointxyz.x = msg->points[i].x;
                pointxyz.y = msg->points[i].y;
                pointxyz.z = msg->points[i].z;
                (*filtered_msg).push_back(pointxyz);
            }
            
                std::lock_guard<std::mutex> lock(queue_mutex);
                lidar_points_queue.push({filtered_msg, timestamp});
                cv.notify_all(); // Notify sync_thread that new data is available
                if (lidar_points_queue.size() > max_queue_size) {
                    lidar_points_queue.pop();
                }
        } catch (const std::exception& e) {
            std::cerr << "Error in filter_lidar_callback: " << e.what() << std::endl;
        }
    }

    void sync_timestamp() {
        sync_thread_future = std::async(std::launch::async, [this]() {
            try {
                while (ros::ok() && !stop_sync_thread) {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cv.wait(lock, [this]() {
                        return !picture_queue.empty() || stop_sync_thread;
                    });

                    if (stop_sync_thread) {
                        break;
                    }
                    if (!picture_queue.empty() && !lidar_points_queue.empty()) {
                        while (!picture_queue.empty() && !lidar_points_queue.empty()) {
                            auto picture = picture_queue.front();
                            auto lidar = lidar_points_queue.front();

                            // 检查哪个时间戳更新
                            if (picture.second >= lidar.second) {
                                // 使用 picture_queue 的时间戳进行同步
                                while (!lidar_points_queue.empty() && picture.second - lidar.second > timestamp_tolerance) {
                                    lidar_points_queue.pop();
                                    if (!lidar_points_queue.empty()) {
                                        lidar = lidar_points_queue.front();
                                    }
                                }

                                // 如果找到匹配的 lidar 数据
                                if (!lidar_points_queue.empty() && std::abs((int64_t)(picture.second - lidar.second)) < timestamp_tolerance) {
                                    project_lidar_on_image(picture.first, lidar.first);
                                    picture_queue.pop();
                                    lidar_points_queue.pop();
                                } else {
                                    // 未找到匹配的，处理下一个 picture
                                    picture_queue.pop();
                                }
                            } else {
                                // 处理反过来的情况
                                picture_queue.pop();
                            }
                        }
                    }

                    
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in sync_timestamp: " << e.what() << std::endl;
            }
        });
    }

    void project_lidar_on_image(cv::Mat& image, const pcl::PointCloud<pcl::PointXYZ>::Ptr& lidar_data) {
        try {
            Eigen::Matrix3d K;
            K << 333.1000665, 0, 330.691479,
                0, 335.804961, 255.718311,
                0, 0, 1;

            Eigen::Matrix3d R;
            R << 0, -1, 0,
                0, 0, -1,
                1, 0, 0;

            Eigen::Vector3d T(0.05, 0.1, -0.46);

            for (const auto& point : lidar_data->points) {
                Eigen::Vector3d point3d(point.x, point.y, point.z);
                Eigen::Vector3d point_cam = R * point3d + T;
                Eigen::Vector3d point_img = K * point_cam;

                int u = static_cast<int>(point_img[0] / point_img[2]);
                int v = static_cast<int>(point_img[1] / point_img[2]);

                if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                    cv::circle(image, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
                }
            }

            publish_image(image);
        } catch (const std::exception& e) {
            std::cerr << "Error in project_lidar_on_image: " << e.what() << std::endl;
        }
    }

    void publish_image(const cv::Mat& image) {
        try {
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
            img_pub.publish(img_msg);
        } catch (const std::exception& e) {
            std::cerr << "Error in publish_image: " << e.what() << std::endl;
        }
    }


};

int main(int argc, char** argv) {
    ros::init(argc, argv, "real_time_lidar_projection");
    try {
        RealTimeLidarProjection projection;
        projection.run();
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
