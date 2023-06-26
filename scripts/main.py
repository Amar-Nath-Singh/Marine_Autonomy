import rospy

rospy.init_node("CamLidar")
from utils import *
from config import *
import numpy as np
from syncsensor import Sensors
from sensor_msgs.msg import Image, LaserScan


def processLidar(data: LaserScan):
    angel_min = data.angle_min
    angel_inc = data.angle_increment
    ranges = np.array(data.ranges)
    angles = angel_min + np.arange(ranges.shape[0]) * angel_inc
    angles[angles < 0] += 2 * np.pi
    mask = (
        (ranges < LASER_MAX_RANGE)
        & (ranges > LASER_MIN_RANGE)
        & (angles > LASER_MIN_ANGLE)
        & (angles < LASER_MAX_RANGE)
    )
    angles = angles[mask]
    ranges = ranges[mask]
    points = np.vstack(
        transformLidarMulti(np.cos(angles) * ranges, np.sin(angles) * ranges)
    ).T
    cont_dist = np.linalg.norm(points[1:] - points[:-1], axis=1)
    kernal = np.ones_like(cont_dist)
    kernal[cont_dist <= EPS] = -1
    kernal = kernal[1:] * kernal[:-1]
    breaks = np.int32(np.where(kernal == -1)[0])

    breaks = np.append(breaks, len(kernal) - 1)
    clusters = []
    obstacles = None
    p = 0
    for x in breaks:
        cloud = points[p:x]
        if not cloud.shape[0] > 0:
            continue
        p = x
        x, y = np.mean(cloud, axis=0)
        r = abs(cloud[0][1] - cloud[-1][1]) / 2
        obs = np.array([[x, y]])
        if r > MIN_OBSTACLE_RADIUS:
            down_cloud = cloud[::FILTER_SCALE]
            obs = LineObstacles(down_cloud)
        if obstacles is None:
            obstacles = obs
        elif cloud.shape[0] >= MIN_SAMPLE:
            obstacles = np.append(obstacles, obs, axis=0)
            clusters.append([x, y, r])

    clusters = np.array(clusters) if len(clusters) > 0 else None

    return obstacles, clusters


def fusionLidarCamera(clusters, cam_data):
    if clusters is None:
        rospy.logwarn("NO OBJECTS")
        return
    image = BRIDGE.imgmsg_to_cv2(cam_data, "bgr8")
    radius = clusters[:, 2]
    clusters = clusters[:, :2]
    mask = (
        (radius > MIN_CLUSTER_RADIUS)
        & (radius < MAX_CLUSTER_RADIUS)
        & (inFOV(clusters))
    )
    clusters = clusters[mask]
    radius = radius[mask]

    cam_cluster = transformCam(clusters, LIDAR_HEIGHT)
    nul = np.zeros((radius.shape[0], 1))
    start_points = transformCam(
        clusters + np.hstack((nul, radius.reshape(-1, 1) + 0.05)), LIDAR_HEIGHT - 0.5
    )
    end_points = transformCam(
        clusters + np.hstack((nul, -radius.reshape(-1, 1) - 0.05)),
        LIDAR_HEIGHT + 0.5,
    )

    start_pix = np.dot(K, start_points.T).T
    start_pix = np.int32(start_pix[:, :2] / start_pix[:, 2].reshape(-1, 1))

    end_pix = np.dot(K, end_points.T).T
    end_pix = np.int32(end_pix[:, :2] / end_pix[:, 2].reshape(-1, 1))

    pixels = np.dot(K, cam_cluster.T).T
    pixels = np.int32(pixels[:, :2] / pixels[:, 2].reshape(-1, 1))

    pix_mask = (
        TruePix(start_pix, image.shape)
        & TruePix(end_pix, image.shape)
        & TruePix(pixels, image.shape)
    )

    start_pix = start_pix[pix_mask]
    end_pix = end_pix[pix_mask]
    pixels = pixels[pix_mask]

    clusters = clusters[pix_mask]
    radius = radius[pix_mask]

    dist = np.linalg.norm(clusters, axis=1)
    dist = np.uint32(dist/3)

    delta = 5 * np.vstack((dist,dist)).T
    start_pix = start_pix - delta
    end_pix = end_pix + delta
    global IDX
    vizImages(
        image=image,
        start_pix=start_pix,
        end_pix=end_pix,
        pix=pixels,
        clusters=clusters,
        radius=radius,
        ID=IDX,
    )
    IDX = IDX + 1


def main():
    rospy.loginfo("Initiating......")

    sensors: Sensors = Sensors(sensors=SENSOR_STRUCT)

    while not rospy.is_shutdown():
        laser_data: LaserScan = sensors.data[LIDAR]
        cam_data: Image = sensors.data[CAMERA]

        if laser_data is None or cam_data is None:
            rospy.logerr("NO SENSOR DATA")
            RATE.sleep()
            continue
        if abs(rospy.Time.now().secs - laser_data.header.stamp.secs) > 5:
            rospy.logwarn("NO LIDAR DETECTED")
            RATE.sleep()
            continue
        if abs(rospy.Time.now().secs - cam_data.header.stamp.secs) > 5:
            rospy.logwarn("NO CAMERA DETECTED")
            RATE.sleep()
            continue

        sensor_sync_delay = (
            abs(laser_data.header.stamp.nsecs - cam_data.header.stamp.nsecs) * 1e-9
        )
        if sensor_sync_delay > MAX_SENSOR_DELAY:
            rospy.logwarn("High Delay")
            continue
        else:
            rospy.loginfo(f"Delay {sensor_sync_delay}")

        obstacles, objects = processLidar(laser_data)
        fusionLidarCamera(objects, cam_data)
        print(repulsionForce(obstacles, k_rep=K_REP, rho0=OBSTACLE_RANGE))
        RATE.sleep()


if __name__ == "__main__":
    main()
