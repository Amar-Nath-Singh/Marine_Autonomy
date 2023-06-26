#!/usr/bin/python3
import dis
import rospy

import cv2
import tf2_ros
from cv_bridge import CvBridge

import message_filters

from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, CameraInfo
from tf2_geometry_msgs import PointStamped

from std_msgs.msg import Float64MultiArray
import rospkg

import numpy as np

OBJ_CENTER_SHAPE = [
    np.array([0, 0.1, -0.25]),
    np.array([0, -0.1, 0.35]),
]  # x1 y1 z1, x2 y2 z2
LIDAR_HEIGHT_CAMERA = -0.045


C = 0
rospack = rospkg.RosPack()
package_path = rospack.get_path("object_localization")
FOLDER = f"{package_path}/dataset/RED"

def transform_coordinates(coords):
    return -(coords[0] + 0.05), -(coords[1] - 0.1), LIDAR_HEIGHT_CAMERA


def transform_coordinates_Temp(x, y, z, from_, to):
    point = PointStamped()
    point.header.frame_id = from_
    point.header.stamp = rospy.Time(0)
    point.point.x = x
    point.point.y = y
    point.point.z = z

    try:
        # Transform the point to the dynamic frame
        transformed_point = tf_buffer.transform(point, to, timeout=rospy.Duration(10.0))
        return [
            transformed_point.point.x,
            transformed_point.point.y,
            transformed_point.point.z,
        ]
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        rospy.logwarn("Failed to transform coordinates.")
        return None


def inFOVCam(x, y):
    theta = np.arctan2(y, x)
    # if theta < 0:
    #     theta = 2 * np.pi + theta
    # print(x,y,np.degrees(theta), -120 < np.degrees(theta) < 120)
    return np.degrees(theta) > -120 and np.degrees(theta) < 120


def TruePix(shape, pix):
    if pix[0] < shape[1] and pix[0] > 0 and pix[1] < shape[0] and pix[1] > 0:
        return True
    return False


def save_crop(center):
    global image, C
    if image is None:
        return
    
    dist = np.linalg.norm(center)
    ptx1 = center + OBJ_CENTER_SHAPE[0] * max(1,dist/3)
    ptx2 = center + OBJ_CENTER_SHAPE[1] * max(1,dist/3)

    mpx= xyz_pix(center[0],center[1],center[2])
    px1 = xyz_pix(ptx1[0], ptx1[1], ptx1[2])
    px2 = xyz_pix(ptx2[0], ptx2[1], ptx2[2])

    if px1 is None or px2 is None:
        return
    if not TruePix(image.shape, px1) or not TruePix(image.shape, px2):

        return
    v1 = min(px1[1], px2[1])
    v2 = max(px1[1], px2[1])
    u1 = min(px1[0], px2[0])
    u2 = max(px1[0], px2[0])
    
    crop = image[v1 : v2, u1 : u2]

    rospy.loginfo(f"{px1}, {px2}")


    rect = cv2.rectangle(image, (u1,v1), (u2,v2),(0,255,0),2)
    rect = cv2.putText(rect,f"D: {round(dist, 3)}", (u1,v1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1,cv2.LINE_AA)
    rect = cv2.circle(rect, mpx, 2, (0,0,255), -1)
    rospy.loginfo(f"Saving {FOLDER}/{C}.png")
    cv2.imwrite(f"{FOLDER}/{C}.png", rect)

    C +=1



def detect(img, center):
    pass

def xyz_pix(x, y, z):
    pts = np.array([-y, z, x])
    pv = np.dot(K, pts)
    return np.int32(pv[:2] / pv[2])


def point_callback(msg):
    global points
    points = np.array(msg.data)
    points = points.reshape((len(points) // 2, 2))


def img_callback(msg):
    global image
    image = bridge.imgmsg_to_cv2(msg, "bgr8")

def callback(img_data, laser):
    global points,image
    image = bridge.imgmsg_to_cv2(img_data, "bgr8")
    points = np.array(laser.data)
    points = points.reshape((len(points) // 2, 2))

if __name__ == "__main__":
    rospy.init_node("label")
    points = None
    image = None
    K = np.array([[479.51992798,   0.        , 328.18896484],
       [  0.        , 479.51992798, 181.17184448],
       [  0.        ,   0.        ,   1.        ]])
    bridge = CvBridge()
    rospy.loginfo("Initiating Detection")
    left_cam_topic = "/zed2i/zed_node/left/image_rect_color"
    cam_info_topic = "/zed2i/zed_node/left/camera_info"

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)


    # rospy.Subscriber("/cluster_scan", Float64MultiArray, point_callback)
    # rospy.Subscriber(left_cam_topic, Image, img_callback)
    # rospy.Subscriber(cam_info_topic, CameraInfo, info_callback)

    pt_sub = message_filters.Subscriber("/cluster_scan",Float64MultiArray)
    cam_sub = message_filters.Subscriber(left_cam_topic, Image)
    ts = message_filters.ApproximateTimeSynchronizer([cam_sub, pt_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.loginfo("Detection Active!")
    rate = rospy.Rate(1)
    NO_DATA_COUNTER = 0

    while not rospy.is_shutdown():
        if image is None:
            if NO_DATA_COUNTER > 10:
                rospy.logwarn("NO IMAGE DATA")
                
            else:
                NO_DATA_COUNTER += 1
            rate.sleep()
            continue

        if points is None:
            if NO_DATA_COUNTER > 10:
                rospy.logwarn("NO CLUSTERED DATA")
            else:
                NO_DATA_COUNTER += 1
            rate.sleep()
            continue
        NO_DATA_COUNTER = 0
        for center in points:
            if len(center) < 2:
                continue
            center = transform_coordinates(center)
            save_crop(center)
        image = None
        points = None
        rate.sleep()
