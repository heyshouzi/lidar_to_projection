# lidar_to_projection
this project is to convert lidar data to projection that can be used in 3d visualization.

# for python version
first,open the folder where python is located.
```sh
cd {PROJECT_DIR}/scripts
```
secondly,run `rtsp_to_topic.py`to convert rtsp stream to ros topic and publish it.
```sh
python rtsp_to_topic.py
```
then run `LidarImageProjector.py` to convert lidar data to projection in image and publish to a new topic.
```sh
python LidarImageProjector.py
```