#!/usr/bin/python3
from filter import Filter
from syncsensor import Sensors
import dbscan
from sensor_msgs.msg import Image, LaserScan
import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
import rospkg

import time


IDX = 0


def timeTaken(fxn, *kwargs):
    b = time.time()
    v = fxn(*kwargs)
    e = time.time()

    print(str(fxn)[:-30], "->", e - b)

    return v


class Fusion:
    def __init__(self, height) -> None:
        self.height = height
        self.LIDAR = "LIDAR"
        self.CAMERA = "CAMERA"
        camera_topic = "/zed2i/zed_node/left_raw/image_raw_color"
        lidar_topic = "/scan"
        rospy.loginfo(f"Waiting for topics {camera_topic}, {lidar_topic}")
        rospy.wait_for_message(camera_topic, Image, timeout=60)
        rospy.wait_for_message(lidar_topic, LaserScan, timeout=60)
        sensor_struct = {
            self.LIDAR: {"topic": lidar_topic, "type": LaserScan},
            self.CAMERA: {
                "topic": camera_topic,
                "type": Image,
            },
        }
        rospack = rospkg.RosPack()

        package_path = rospack.get_path("object_localization")
        self.folder = f"{package_path}/dataset"
        self.sensors = Sensors(sensors=sensor_struct)
        self.filters = Filter(range=[0.1, 15], angle=[-60, 60])
        self.bridge = CvBridge()

        self.K = (
            np.array(
                [
                    [479.51992798, 0.0, 328.18896484],
                    [0.0, 479.51992798, 181.17184448],
                    [0.0, 0.0, 1.0],
                ]
            )
            * 2
        )

        self.lidar_min_range = 0.1
        self.lidar_max_range = 6

        self.cluster_min_angle = np.radians(-180)
        self.cluster_max_angle = np.radians(180)

        self.filter_radius = 0.05

        self.eps = 0.3
        self.min_sample = 3

        self.min_obstacle_radius = 0.5
        self.obstacle_range = 5

        self.min_radius = 0.1
        self.max_radius = 10

    def getData(self):

        laser_data = self.sensors.data[self.LIDAR]
        cam_data = self.sensors.data[self.CAMERA]
        if laser_data is None:
            rospy.logwarn("NO LIDAR")
            return -1
        if cam_data is None:
            rospy.logwarn("NO CAMERA")
            return -1

        delay = abs(laser_data.header.stamp.nsecs - cam_data.header.stamp.nsecs) * 1e-9
        if abs(rospy.Time.now().secs - laser_data.header.stamp.secs) > 5:
            rospy.logwarn("NO LIDAR DETECTED")
        if abs(rospy.Time.now().secs - cam_data.header.stamp.secs) > 5:
            rospy.logwarn("NO CAMERA DETECTED")
            return
        if delay > 0.04:
            rospy.logwarn("High Delay")
            return -1
        else:
            rospy.loginfo(f"Delay {delay}")

        obstacles, clusters = timeTaken(self.filterLidar, laser_data)
        image = self.bridge.imgmsg_to_cv2(cam_data, "bgr8")

        return obstacles, clusters, image

    def transformCam(self, points, h):
        trans_clusters = np.zeros((len(points), 3))
        trans_clusters[:, 2] = points[:, 0]
        trans_clusters[:, 0] = points[:, 1] * -1
        trans_clusters[:, 1] = h
        return trans_clusters

    def transformLidar(self, x, y):
        # x_diff = 0.055 if np.arctan2(-y , -x) > 0 else -0.055
        # y_diff = -0.08
        x_diff = 0.055 if np.arctan2(-y, -x) > 0 else -0.055
        y_diff = -0.08
        return np.array([-(x + x_diff), -(y + y_diff)])

    def transformLidarMulti(self, x, y):
        # x_diff = 0.055 if np.arctan2(-y , -x) > 0 else -0.055
        # y_diff = -0.08
        # x = x[x > 0] + 0.055
        # x = x[x <= 0] - 0.055

        y += -0.08
        return np.array([-(x), -(y)])

    def fit_circle(self, points):

        points = np.hstack((points, np.ones((points.shape[0], 1))))
        b = -points[:, 0] ** 2 - points[:, 1] ** 2
        center, _, _, _ = np.linalg.lstsq(points, b, rcond=None)
        cx = center[0] / -2
        cy = center[1] / -2
        r = np.sqrt(cx**2 + cy**2 - center[2])

        return cx, cy, r

    def realObstacle(self, points):
        A = points[:-1]
        B = points[1:]
        rel_pose = A - B
        mab = rel_pose[:, 1] / rel_pose[:, 0]
        mpc = -1 / mab
        x = (-A[:, 1] + mab * A[:, 0]) / (mab - mpc)
        y = A[:, 1] + mab * (x - A[:, 0])
        obstacles = np.hstack((np.expand_dims(x, 1), np.expand_dims(y, 1)))

        mask = (
            np.linalg.norm(obstacles - A, axis=1) < np.linalg.norm(A - B, axis=1)
        ) & (np.linalg.norm(obstacles - B, axis=1) < np.linalg.norm(A - B, axis=1))
        obstacles = obstacles[mask]

        return obstacles

    # def viz(self, points):

    #     self.viz_img =

    def filterLidar(self, data):
        global IDX
        angel_min = data.angle_min
        angel_inc = data.angle_increment
        ranges = np.array(data.ranges)
        angles = angel_min + np.arange(ranges.shape[0]) * angel_inc
        angles[angles < 0] += 2 * np.pi
        mask = (ranges < self.lidar_max_range) & (ranges > self.lidar_min_range)
        angles = angles[mask]
        ranges = ranges[mask]
        points = np.vstack(
            self.transformLidarMulti(np.cos(angles) * ranges, np.sin(angles) * ranges)
        ).T
        cont_dist = np.linalg.norm(points[1:] - points[:-1], axis=1)
        kernal = np.ones_like(cont_dist)
        kernal[cont_dist <= self.eps] = -1
        kernal = kernal[1:] * kernal[:-1]
        breaks = np.int32(np.where(kernal == -1)[0])
        clusters = []
        obstacles = None
        p = 0
        for x in breaks:
            cloud = points[p:x]
            p = x
            x, y = np.mean(cloud, axis=0)
            r = abs(cloud[0][1] - cloud[-1][1])/2
            theta = np.arctan2(x,y)
            obs = np.array([[x, y]])
            if r > self.min_obstacle_radius:
                down_cloud = cloud[:: self.min_sample - 1]
                obs = self.realObstacle(down_cloud)
            if obstacles is None:
                obstacles = obs
            else:
                obstacles = np.append(obstacles, obs, axis=0)
            if theta > self.cluster_min_angle and theta < self.cluster_max_angle:
                if x < 0:
                    continue
                clusters.append([x, y, 0.5])
                print([np.linalg.norm(x - y), np.degrees(theta)])
        return obstacles, np.array(clusters)

        # for i, r in enumerate(data.ranges):

        #     if r > self.lidar_max_range or r < self.lidar_min_range:
        #         continue

        #     a = angel_min + i * angel_inc
        #     if a < 0:
        #         a = 2 * np.pi + a

        #     p = np.expand_dims(self.transformLidar(r * np.cos(a), r * np.sin(a)),0)

        #     if prev_coord is None:
        #         prev_coord = p

        #     d = np.linalg.norm(prev_coord - p)
        #     prev_coord = p

        #     prev_coord = p
        #     if filterd_cloud is None:
        #         filterd_cloud = p
        #         continue
        #     if labels is None:
        #         labels = np.array([l])
        #         continue
        #     if (
        #         np.degrees(a) > self.cluster_max_angle
        #         or np.degrees(a) < self.cluster_min_angle
        #     ):
        #         continue

        #     if np.linalg.norm(filterd_cloud[-1] - p) > self.filter_radius or True:
        #         filterd_cloud = np.append(filterd_cloud, p, axis = 0)
        #         if d > self.eps:
        #             l += 1
        #         labels = np.append(labels, np.array([l]), axis = 0)
        # if filterd_cloud is None:
        #     rospy.logwarn("Filter problem")
        #     return
        # # labels = np.array(dbscan.dbscan(filterd_cloud, eps=0.3, MinPts=2))
        # unique_labels = np.unique(labels)
        # # print(unique_labels)
        # obstacles = None # Lidar Cloud
        # objects = None # buoys

        # # print(f"Cloud reudced from {len(data.ranges)} to {filterd_cloud.shape[0]}")

        # for unq_lab in unique_labels:
        #     cloud = filterd_cloud[labels == unq_lab]
        #     x,y,r = self.fit_circle(cloud)
        #     x,y = np.mean(cloud, axis = 0)
        #     cluster = np.expand_dims([x,y,r],0)
        #     obs_pose = cluster[:,:2]
        #     if r > self.min_obstacle_radius:
        #         obs_pose = self.realObstacle(cloud)
        #     if objects is None:
        #         objects = cluster
        #     else:
        #         objects = np.append(objects, cluster, axis = 0)
        #     if obstacles is None:
        #         obstacles = obs_pose
        #     else:
        #         obstacles = np.append(obstacles, obs_pose, axis = 0)
        # # print(objects)
        # return obstacles, objects

    def detect(self, img):
        return 0

    def repulsionForce(self, obstacles, k_rep, rho0):
        if obstacles is None:
            return np.zeros(2)
        rho = np.linalg.norm(obstacles, axis=1)
        mask = rho <= rho0
        obstacles = obstacles[mask]
        rho = rho[mask]
        if not obstacles.shape[0] > 0:
            return np.zeros(2)
        const = k_rep * ((1 / rho) - (1 / rho0)) * ((1 / rho**2))
        forces = (-obstacles) * const.reshape(-1, 1)
        return np.sum(forces, axis=0)

    def localize_detect(self, clusters, image):

        if clusters is None:
            return
        if image is None:
            return

        radius = clusters[:, 2]
        clusters = clusters[:, :2]
        mask = (radius > self.min_radius) & (radius < self.max_radius)
        clusters = clusters[mask]
        radius = radius[mask]

        cam_cluster = self.transformCam(clusters, self.height)
        nul = np.zeros((radius.shape[0], 1))
        start_points = self.transformCam(
            clusters + np.hstack((nul, radius.reshape(-1, 1) + 0.05)), self.height - 0.5
        )
        end_points = self.transformCam(
            clusters + np.hstack((nul, -radius.reshape(-1, 1) - 0.05)),
            self.height + 0.5,
        )

        start_pix = np.dot(self.K, start_points.T).T
        start_pix = np.int32(start_pix[:, :2] / start_pix[:, 2].reshape(-1, 1))

        end_pix = np.dot(self.K, end_points.T).T
        end_pix = np.int32(end_pix[:, :2] / end_pix[:, 2].reshape(-1, 1))

        pixels = np.dot(self.K, cam_cluster.T).T
        pixels = np.int32(pixels[:, :2] / pixels[:, 2].reshape(-1, 1))

        i = 0
        for pix, cluster, r, start, end in zip(
            pixels, clusters, radius, start_pix, end_pix
        ):

            d = np.linalg.norm(cluster)

            start = start - np.ones(2) * int(d / 3) * 5
            end = end + np.ones(2) * int(d / 3) * 5
            start = start.astype(np.int32)
            end = end.astype(np.int32)

            # if start[0] < 0 or start[0] > image.shape[1]:
            #     continue
            # if end[0] < 0 or end[0] > image.shape[1]:
            #     continue
            # if start[1] < 0 or start[1] > image.shape[0]:
            #     continue
            # if end[1] < 0 or end[1] > image.shape[0]:
            #     continue

            # img_crp = image[start[1]:end[1], start[0]:end[0]]
            # cv2.imwrite(f"{self.folder}/{c}_{round(r,2)}.png", img_crp)

            image = cv2.circle(image, pix, 3, (0, 0, 255), -1)
            image = cv2.putText(
                image,
                f"D: {round(d, 3)}",
                pix - 5,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
            )
            image = cv2.putText(
                image,
                f"R: {round(r, 3)}",
                pix + 10,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
            )

            # image = cv2.rectangle(image, start, end, (0, 255, 0), 2)
            i += 1
        cv2.imshow(f"{self.folder}.png", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("perception")
    rospy.loginfo("Initiating perception")

    f = Fusion(0.045)
    rate = rospy.Rate(5)

    rate.sleep()
    c = 0
    while not rospy.is_shutdown():
        data = timeTaken(f.getData)
        if isinstance(data, int) or data is None:
            continue
        timeTaken(f.localize_detect, data[1], data[2])
        rep_force = f.repulsionForce(data[0], 0.5, 5)
        c += 1
        rate.sleep()
