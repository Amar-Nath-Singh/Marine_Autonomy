#!/usr/bin/python3

import numpy as np
import rospy

# from sklearn.cluster import DBSCAN
import dbscan
from tf2_geometry_msgs import PointStamped
import tf2_ros


class Filter:
    def __init__(
        self,
        voxel_size=0.1,
        eps=0.3,
        min_sample=3,
        range=[0.1, 10],
        angle=[-60, 60],
        dim=np.array([0.1, 0.5]),
        thresh=np.array([0.3, 0.3]),
        frame="laser",
        parent="zed2i_base_link",
    ) -> None:
        self.voxel_size = voxel_size
        self.eps = eps
        self.angle = angle
        self.range = range
        self.min_sample = min_sample
        self.dim = dim
        self.thresh = thresh

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.frame = frame

        self.parent = parent

        del voxel_size
        del eps
        del angle
        del range
        del min_sample
        del dim

    def transform_coordinates(self, x, y):
        # x_diff = 0.055 if np.arctan2(-y , -x) > 0 else -0.055
        # y_diff = -0.08
        x_diff = 0.055 if np.arctan2(-y , -x) > 0 else -0.055
        y_diff = -0.08
        return -(x + x_diff), -(y + y_diff)
        point = PointStamped()
        point.header.frame_id = self.frame
        point.header.stamp = rospy.Time(0)
        point.point.x = x
        point.point.y = y
        point.point.z = 0

        try:
            # Transform the point to the dynamic frame
            transformed_point = self.tf_buffer.transform(
                point, self.parent, timeout=rospy.Duration(10.0)
            )
            return np.array(
                [
                    transformed_point.point.x,
                    transformed_point.point.y,
                ]
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Failed to transform coordinates.")
            return None
    def fit_circle(self, points):

        # return 0.1,0.1,0.1

        points = np.hstack((points, np.ones((points.shape[0],1))))
        
        b = -points[:,0]**2 - points[:,1]**2
        
        # Solve the linear system of equations
        center, residue, rank, singular_values = np.linalg.lstsq(points, b, rcond=None)
        
        # Extract the center coordinates and radius
        cx = center[0] / -2
        cy = center[1] / -2
        r = np.sqrt(cx**2 + cy**2 - center[2])

        return cx, cy, r

    def realObstacle(self, points):
        A = points[:-1]
        B = points[1:]
        rel_pose = A - B
        mab = rel_pose[:,1]/rel_pose[:,0]
        mpc = -1/mab
        x = (-A[:,1] + mab* A[:,0])/(mab-mpc)
        y = A[:,1] + mab*(x - A[:,0])
        obstacles = np.hstack((np.expand_dims(x,1),np.expand_dims(y,1)))

        mask = (np.linalg.norm(obstacles - A, axis = 1) < np.linalg.norm(A - B, axis = 1)) & (np.linalg.norm(obstacles - B, axis = 1) < np.linalg.norm(A - B, axis = 1))
        obstacles = obstacles[mask]
        A = A[mask]
        B = B[mask]

        print("OBS : ",obstacles, A, B)

        return obstacles
        
    def laser_filter_OP(self, data, cluster_thr, filter_thr):
        if data is None:
            return -1
        points = None
        labels = None
        current_label = 0
        prev_coord = None
        fil_coord = None
        for i, r in enumerate(data.ranges):

            a = data.angle_min + i * data.angle_increment
            
            if a < 0:
                a = 2 * np.pi + a
            
            p = np.array([[r * np.cos(a), r * np.sin(a)]])

            if (
                np.degrees(a) > self.angle[1] + 180
                or np.degrees(a) < self.angle[0] + 180
            ):
                continue

            if prev_coord is None:
                prev_coord = p
            
            if fil_coord is None:
                fil_coord = p

            d_coord = np.linalg.norm(p - prev_coord)
            d_fil = np.linalg.norm(p - fil_coord)
            prev_coord = p

            if r > self.range[1] or r < self.range[0]:
                continue

            if d_fil < filter_thr:
                continue

            fil_coord = p
            p, fil_coord = (fil_coord + p) / 2, p

            if d_coord > cluster_thr:
                current_label += 1

            if points is None:
                points = p
                labels = np.array([current_label])
            else:
                points = np.append(points, p, axis=0)
                labels = np.append(labels, np.array([current_label]), axis=0)

        if points is None:
            rospy.logwarn("NO OBJECT NEARBY")
            return -1, -1, -1, -1
        unique_labels = np.unique(labels)

        if not len(unique_labels) > 0:
            rospy.logwarn("CLUSTERING FAILED")
            return -1, -1, -1, -1
        clusters = None
        obstacles = None
        for l in unique_labels:
            cloud = points[labels == l]
            # x,y = np.mean(cloud, axis=0)
            x,y,r = self.fit_circle(cloud)
            cen = np.expand_dims([x,y,r],0)
            obs = cen[:,:2]
            if r > 0.5:
                obs = self.realObstacle(cloud)
            if obstacles is None and obs is not None:
                obstacles = obs
            elif obs is not None:
                obstacles = np.append(obstacles, obs, axis=0)

            if clusters is None:
                clusters = cen
            else:
                clusters = np.append(clusters, cen, axis=0)

        return obstacles, clusters

    def laser_filter(self, data):
        if data is None:
            return -1
        points = None
        for i, r in enumerate(data.ranges):
            if r > self.range[1] or r < self.range[0]:
                continue
            a = data.angle_min + i * data.angle_increment

            if a < 0:
                a = 2 * np.pi + a
            if (
                np.degrees(a) > self.angle[1] + 180
                or np.degrees(a) < self.angle[0] + 180
            ):
                continue

            p = np.array([[r * np.cos(a), r * np.sin(a)]])

            if points is None:
                points = p
            else:
                points = np.append(points, p, axis=0)
        if points is None:
            rospy.logwarn("NO OBJECT NEARBY")
            return -1, -1
        mask = np.isinf(points).any(axis=1)
        points = points[~mask]
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        voxel_dict = {}
        for i in range(voxel_indices.shape[0]):
            voxel = tuple(voxel_indices[i])
            if voxel in voxel_dict:
                voxel_dict[voxel].append(points[i])
            else:
                voxel_dict[voxel] = [points[i]]
        voxel_centroids = None
        for i, (voxel, voxel_points) in enumerate(voxel_dict.items()):
            voxel_points = np.array(voxel_points)
            centroid = np.mean(voxel_points, axis=0)
            centroid = self.transform_coordinates(centroid[0], centroid[1])
            if voxel_centroids is None:
                voxel_centroids = np.array([[centroid[0], centroid[1]]])
            else:
                voxel_centroids = np.append(
                    voxel_centroids, np.array([[centroid[0], centroid[1]]]), axis=0
                )
        if voxel_centroids is None:
            rospy.logwarn("VOXEL FAILED")
            return -1, -1
        # dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample)
        # labels = dbscan.fit_predict(voxel_centroids)
        labels = dbscan.dbscan(voxel_centroids, eps=self.eps, MinPts=self.min_sample)
        unique_labels = np.unique(labels)

        if len(unique_labels) - 1 < 0:
            rospy.logwarn("CLUSTERING FAILED")
            return -1, -1

        clusters = None

        for l in unique_labels:
            cloud = voxel_centroids[labels == l]
            x = cloud[:, 0]
            y = cloud[:, 1]
            dim = np.array([np.max(x) - np.min(x), np.max(y) - np.min(y)])
            if ((dim - self.dim) < self.thresh).all():
                cen = np.array([np.mean(cloud, axis=0)])
                if clusters is None:
                    clusters = cen
                else:
                    clusters = np.append(clusters, cen, axis=0)
        if clusters is None:
            return -1
            # print(clusters)
        return voxel_centroids, np.hstack((clusters, np.ones((len(clusters),1)) * 0.5))
