#!/usr/bin/python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float64MultiArray
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf.transformations import quaternion_from_euler


def lidar_callback(msg):
    global data
    data = msg


def voxel_grid_filter( data ):
    if data is None:
        return
    ranges = data.ranges
    angle_increment = data.angle_increment
    start_angle = data.angle_min
    voxel_size = 0.05
    points = []
    for i, range_measurement in enumerate(ranges):
        angle = start_angle + i * angle_increment
        t = angle
        if t < 0:
            t = 2 * np.pi + t
        if not 120 < np.degrees(t) < 240:
            continue
        x = range_measurement * np.cos(angle)
        y = range_measurement * np.sin(angle)
        points.append([x, y])

    if not len(points) > 0:
        rospy.logwarn(f"NO LIDAR DATA")
        return
    points = np.array(points)
    mask = np.isinf(points).any(axis=1)
    points = points[~mask]
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}
    for i in range(voxel_indices.shape[0]):
        if (points[i][0] ** 2 + points[i][1] ** 2) ** 0.5 > 10:
            continue
        voxel = tuple(voxel_indices[i])
        if voxel in voxel_dict:
            voxel_dict[voxel].append(points[i])
        else:
            voxel_dict[voxel] = [points[i]]
    voxel_centroids = []
    for i, (voxel, voxel_points) in enumerate(voxel_dict.items()):
        voxel_points = np.array(voxel_points)
        centroid = np.mean(voxel_points, axis=0)
        voxel_centroids.append([centroid[0], centroid[1]])
    if not len(voxel_centroids) > 0:
        rospy.logwarn("VOXEL FAILED")
        return
    dbscan = DBSCAN(eps=0.3, min_samples=3)
    labels = dbscan.fit_predict(voxel_centroids)
    voxel_centroids = np.array(voxel_centroids)
    unique_labels = np.unique(labels)
    if not len(voxel_centroids) > 0:
        print(unique_labels)
        rospy.loginfo("NO OBJETCS NEAR")
    clusters = []
    for l in unique_labels:
        if l == -1:
            continue
        clusters.append(np.mean(voxel_centroids[labels == l], axis=0))
    if not len(clusters) < 0:
        clu = Float64MultiArray()
        clu.data = list(np.array(clusters).flatten())
        cluster_pub.publish(clu)
    else:

        print(unique_labels)
        rospy.loginfo("NO OBJETCS NEAR")

if __name__ == "__main__":
    rospy.init_node("clustering")
    rospy.loginfo("Initiating Clustering")
    rospy.Subscriber("/scan", LaserScan, lidar_callback)
    cluster_pub = rospy.Publisher("/cluster_scan", Float64MultiArray, queue_size=10)
    data = None
    rate = rospy.Rate(100)
    NO_DATA_COUNTER = 0
    rospy.loginfo("Clustering Active!")
    while not rospy.is_shutdown():
        if data is None:
            if NO_DATA_COUNTER > 10:
                rospy.logwarn("NO LIDAR DATA")
            else:
                NO_DATA_COUNTER += 1
            rate.sleep()
            continue
        NO_DATA_COUNTER = 0
        voxel_grid_filter(data)
        rate.sleep()
