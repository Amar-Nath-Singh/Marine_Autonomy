from sensor_msgs.msg import Image, LaserScan
import numpy as np
import rospy
import rospkg
from cv_bridge import CvBridge


rospack = rospkg.RosPack()

package_path = rospack.get_path("object_localization")

LIDAR = "LIDAR"
CAMERA = "CAMERA"
LIDAR_HEIGHT = -0.045
CAMERA_TOPIC = "/zed2i/zed_node/left_raw/image_raw_color"
LIDAR_TOPIC = "/scan"
SENSOR_STRUCT = {
    LIDAR: {"topic": LIDAR_TOPIC, "type": LaserScan},
    CAMERA: {
        "topic": CAMERA_TOPIC,
        "type": Image,
    },
}

K = (
    np.array(
        [
            [479.51992798, 0.0, 328.18896484],
            [0.0, 479.51992798, 181.17184448],
            [0.0, 0.0, 1.0],
        ]
    )
    * 2
)

LASER_MIN_RANGE = 0.1
LASER_MAX_RANGE = 15

LASER_MIN_ANGLE = np.radians(-180)
LASER_MAX_ANGLE = np.radians(180)

CAMERA_WIDE_MIN = np.radians(-45)
CAMERA_WIDE_MAX = np.radians(45)


MIN_CLUSTER_RADIUS = 0.01
MAX_CLUSTER_RADIUS = 0.3

CLUSTER_MIN_ANGLE = np.radians(-180)
CLUSTER_MAX_ANGLE = np.radians(180)

FILTER_SCALE = 3

EPS = 0.3
MIN_SAMPLE = 3

MIN_OBSTACLE_RADIUS = 0.5
OBSTACLE_RANGE = 5
K_REP = 10

MAX_SENSOR_DELAY = 0.02

RATE = rospy.Rate(1)

SAVES = True

DATASET_FOLDER = f"{package_path}/dataset"
BRIDGE = CvBridge()

IDX = 0