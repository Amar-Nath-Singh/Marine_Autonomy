#!/usr/bin/python3

import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
from ultralytics import YOLO
import rospkg
from std_msgs.msg import Float32MultiArray
import rospy
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


import message_filters
import tf2_ros
import torch
import torchvision.transforms as transforms


def map_viz(ptx):
    marker_msg = Marker()
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.header.frame_id = "odom"
    marker_msg.ns = "points"
    marker_msg.id = 0
    marker_msg.type = Marker.POINTS
    marker_msg.action = Marker.ADD
    marker_msg.pose.orientation.w = 1.0
    marker_msg.scale.x = 0.1  # Point size
    marker_msg.scale.y = 0.1  # Point size

    marker_msg.color.r = 1.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 0.0
    marker_msg.color.a = 1.0
    wc = []
    for c in ptx:
        wc.append(Point(x = -c[0], y = -c[1], z = 0))
    marker_msg.points = wc
    # colors = []
    # for lab in world_label:
    #     color = ColorRGBA()
    #     if np.argmax(lab) == RED:
    #         color.r = 1
    #     elif np.argmax(lab) == GREEN:
    #         color.g = 1
    #     else:
    #         color.b = 1
    #     color.a = 0.5
    #     colors.append(colors)
    # marker_msg.colors = colors
    color = ColorRGBA()
    color.r = 0
    color.g = 0
    color.b = 1
    color.a = 0.5
    marker_msg.colors = [color] * len(ptx)

    pub.publish(marker_msg)


def voxel_filter(points, voxel_size):
    # Create a dictionary to store voxel indices as keys and the corresponding points as values
    voxel_dict = {}

    # Iterate over each point and assign it to the appropriate voxel
    for point in points:
        voxel_index = tuple(np.floor(point / voxel_size).astype(int))
        if voxel_index in voxel_dict:
            voxel_dict[voxel_index].append(point)
        else:
            voxel_dict[voxel_index] = [point]

    # Convert the dictionary values (points) to NumPy arrays
    filtered_points = np.array(list(voxel_dict.values()))

    return filtered_points


def transform_coordinates(x, y, z):
    point = PointStamped()
    point.header.frame_id = "zed2i_base_link"
    point.header.stamp = rospy.Time(0)
    point.point.x = x
    point.point.y = y
    point.point.z = z

    try:
        # Transform the point to the dynamic frame
        transformed_point = tf_buffer.transform(
            point, "odom", timeout=rospy.Duration(10.0)
        )
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


def remove_outliers(coords, threshold=3):
    z_scores = np.abs((coords - np.mean(coords, axis=0)) / np.std(coords, axis=0))
    return coords[z_scores < threshold]


def XYZ_to_pix(coord, z):
    pts = np.array([-coord[1], z, coord[0]])
    pv = np.dot(K, pts)
    return np.int32(pv[:2] / pv[2])


def detect(img):
    results = model.predict(img, conf=0.25, verbose=False)
    # for obj in results:
    #     if debug:
    #         if len(obj.boxes.xyxy) <= 0:
    #             continue
    #         x1, y1, x2, y2 = obj.boxes.xyxy[0]
    #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #         label = f'{model.names[int(obj.boxes.cls[0])]} {obj.boxes.conf[0]:.2f}'
    #         cv2.putText(img, label, (int(x1), int(y1) - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #         cv2.imshow('Yolo Detection', img)
    #         cv2.waitKey(1)
    return results


def gen_world(real_objects):
    global world_coord, world_label
    for x, y, z, l in real_objects:
        coord = np.array(transform_coordinates(x, y, z))
        label = int(l)
        conf = l - label
        if world_coord is None and world_label is None:
            np.append(world_coord, coord)
            prob_label = np.ones_like(LABELS) * (1 - conf)
            prob_label[label] = conf
            prob_label = prob_label / np.sum(prob_label)
            world_coord = np.array([coord])
            world_label = np.array([prob_label])
            continue

        nearWorldObjIdx = np.argmin(np.linalg.norm(world_coord - coord, axis=1))
        nearWorldObjDist = np.linalg.norm(world_coord[nearWorldObjIdx] - coord)

        if nearWorldObjDist < 0.5:
            world_coord[nearWorldObjIdx] = (world_coord[nearWorldObjIdx] + coord) / 2
            world_label[nearWorldObjIdx] = world_label[nearWorldObjIdx] * (1 - conf)
            world_label[nearWorldObjIdx][label] = (
                world_label[nearWorldObjIdx][label] * conf / (1 - conf)
            )
            world_label[nearWorldObjIdx][label] = world_label[nearWorldObjIdx][
                label
            ] / np.sum(world_label[nearWorldObjIdx][label])

            world_label[nearWorldObjIdx][label] = np.round_(world_label[nearWorldObjIdx][label], decimals = 3)
        else:
            np.append(world_coord, coord)
            prob_label = np.ones_like(LABELS) * (1 - conf)
            prob_label[label] = conf
            prob_label = prob_label / np.sum(prob_label)
            np.append(world_label, prob_label)

def cam_callback(msg):
    global K
    K = np.array(msg.K).reshape((3, 3))


def img_callback(msg):
    global objects, wait, cv_image
    # objects = []
    # return
    if wait == 0:
        rospy.loginfo("DETECTION INITIATED")
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    t = rospy.Time.now().secs
    objects = detect(cv_image)
    rospy.loginfo(f"YOLO Duration : {rospy.Time.now().secs - t}")
    t = rospy.Time.now().secs
    if wait <= 10:
        rospy.loginfo("WAITING FOR DETECTION")
        wait += 1


def sensor_callback(laser):
    instamp = laser.header.stamp
    global K, objects, cv_image
    if K is None:
        rospy.loginfo("NO DATA FROM CAMERA")
        return
    if wait < 5:
        return
    if objects is None:
        return
    ranges = laser.ranges
    angle_increment = laser.angle_increment
    start_angle = laser.angle_min
    coords = []

    t = rospy.Time.now().secs
    rospy.loginfo(f"Values recieved at {t}")

    for i, range_measurement in enumerate(ranges):
        angle = start_angle + i * angle_increment
        if angle < 0:
            angle = 2 * np.pi + angle
        if not (angle > np.radians(120) and angle < np.radians(240)):
            continue
        x = range_measurement * -np.cos(angle)
        y = range_measurement * -np.sin(angle)
        coords.append([x, y])

    coords = np.array(coords)
    mask = np.isinf(coords).any(axis=1)
    filtered_coordinates = coords[~mask]

    rospy.loginfo(f"LIDAR DATA Duration : {rospy.Time.now().secs - t}")

    pixels = np.zeros((len(filtered_coordinates), 2))
    labels = np.ones(len(filtered_coordinates)) * UNK
    z = -0.045

    for i, point in enumerate(filtered_coordinates):
        pts = np.array([-point[1], z, point[0]])
        pv = np.dot(K, pts)
        pixels[i] = np.int32(pv[:2] / pv[2])

    rospy.loginfo(f"LIDAR CAM Fusion Duration: {rospy.Time.now().secs - t}")
    t = rospy.Time.now().secs

    real_objects = []
    for obj in objects:
        if len(obj.boxes.conf) > 0 and obj.boxes.conf[0] > 0.55:
            p = obj.boxes.xyxy[0].cpu().numpy()
            p1 = p[:2]
            p2 = p[-2:]
            print(p1,p2)
            mask = (pixels > p1).all(axis=1) & (pixels < p2).all(axis=1)
            l = label_struct[int(obj.boxes.cls[0])] + obj.boxes.conf[0].cpu().numpy()
            labels[mask] = l
            # coord = filtered_coordinates[mask]
            map_viz(filtered_coordinates[mask])
            # if not np.isnan(coord).any():
            #     coord = coord[len(coord) // 2]
            #     real_objects.append([coord[0], coord[1], z, l])

    objects = None

    rospy.loginfo(f"Object Estimation Duration : {rospy.Time.now().secs - t}")
    t = rospy.Time.now().secs

    gen_world(real_objects)

    rospy.loginfo(f"Map geneartion Duration : {rospy.Time.now().secs - t}")
    t = rospy.Time.now().secs



if __name__ == "__main__":
    rospy.init_node("detect")  # Initialize ROS node
    rospy.loginfo("Initiated..........")
    K = None
    cv_image = None
    objects = None
    wait = 0

    debug = True
    NUM_CLASSES = 3
    LABELS = [0, 1, 2]
    GREEN = 1
    RED = 2
    UNK = 0
    label_struct = {  # model --> real
        0: GREEN,
        2: RED,
        -1: UNK,
        1: UNK,
    }
    t = rospy.Time.now().secs
    world_coord = None
    world_label = None

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rospy.loginfo("Lodaing Model..........")
    bridge = CvBridge()
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("object_localization")
    model_path = f"{package_path}/model/RBRv8.pt"
    model = YOLO(model_path)
    model.to("cuda")
    rospy.loginfo("Testing Model..........")
    # dummy_image = np.zeros((340,640, 3), dtype = np.uint8)
    # for _ in range(5):
    #     detect(dummy_image)
    rospy.loginfo("Testing Model Finished.")
    left_cam_topic = "/zed2i/zed_node/left/image_rect_color"
    lidar_topic = "/scan"
    cam_info_topic = "/zed2i/zed_node/left/camera_info"
    rospy.loginfo(f"Subscribing to {left_cam_topic}")
    rospy.loginfo(f"Subscribing to {lidar_topic}")
    # lidar_sub = message_filters.Subscriber(lidar_topic, LaserScan)
    # cam_img_sub = message_filters.Subscriber(left_cam_topic, Image)
    # ts = message_filters.ApproximateTimeSynchronizer([lidar_sub, cam_img_sub], 10 ,1)
    # ts.registerCallback(sensor_callback)

    lidar_sub = rospy.Subscriber(lidar_topic, LaserScan, sensor_callback)
    cam_img_sub = rospy.Subscriber(left_cam_topic, Image, img_callback)
    rospy.Subscriber(cam_info_topic, CameraInfo, cam_callback)
    pub = rospy.Publisher("/per_objects", Marker,queue_size=10)

    rospy.spin()
