#!/usr/bin/python3
import torch
import numpy as np
import message_filters
import rospy
import time


def getTimeDelay(fnx, *kwargs):
    begin = time.time()
    v = fnx(*kwargs)
    end = time.time()
    print(f"Time taken in {fnx} ",end - begin)
    return v
class SyncSensors:
    def __init__(self, sensors: dict):
        self.sensors = sensors
        self.names = sensors.keys()
        self.data = {}
        pub_dict = {}
        for sensor in self.names:
            pub_dict[sensor] = message_filters.Subscriber(
                sensors[sensor]["topic"], sensors[sensor]["type"]
            )
            self.data[sensor] = None
        ts = message_filters.ApproximateTimeSynchronizer(list(pub_dict.values()), 10, 0.02)
        ts.registerCallback(self.callback)

    def callback(self, *msgs):
        for name, msg in zip(self.names, msgs):
            self.data[name] = msg

class LaserProcess:
    def __init__(self, height, min_range = 0.1, max_range=10, min_angle = 0, max_angle = 360, filter_radius = 0.05, eps = 0.1, min_samples = 3):
        self.height = height
        self.min_range = min_range
        self.max_range = max_range
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.filter_radius = filter_radius
        self.eps = eps
        self.min_samples = min_samples


class LocalObstacleAvoid:
    def __init__(self, k_rep, rho_0) -> None:
        self.k_rep = k_rep
        self.rho_0 = rho_0

    def fit_circle(self, points):

        points = np.hstack((points, np.ones((points.shape[0],1))))
        b = -points[:,0]**2 - points[:,1]**2
        center, _, _, _ = np.linalg.lstsq(points, b, rcond=None)
        cx = center[0] / -2
        cy = center[1] / -2
        r = np.sqrt(cx**2 + cy**2 - center[2])

        return cx, cy, r

    

    def getObstacle_numpy(self, cloud):
        bn = time.time()
        A = cloud[:-1]
        B = cloud[1:]
        rel_pose = A - B
        mab = rel_pose[:,1]/rel_pose[:,0]
        mpc = -1/mab
        x = (-A[:,1] + mab* A[:,0])/(mab-mpc)
        y = A[:,1] + mab*(x - A[:,0])
        obstacles = np.hstack((np.expand_dims(x,1),np.expand_dims(y,1)))
        mask = (np.linalg.norm(obstacles - A, axis = 1) < np.linalg.norm(A - B, axis = 1)) & (np.linalg.norm(obstacles - B, axis = 1) < np.linalg.norm(A - B, axis = 1))
        obstacles = obstacles[mask]
        en = time.time()
        print(en - bn)
    def getObstacle_torch(self, cloud):
        bn = time.time()
        A = cloud[:-1]
        B = cloud[1:]
        rel_pose = A - B
        mab = rel_pose[:,1]/rel_pose[:,0]
        mpc = -1/mab
        x = (-A[:,1] + mab* A[:,0])/(mab-mpc)
        y = A[:,1] + mab*(x - A[:,0])

        obstacles = torch.hstack((x.unsqueeze(-1),y.unsqueeze(-1)))
        d = torch.norm(A - B, dim = 1)
        mask = (torch.norm(obstacles - A, dim = 1) < d) & (torch.norm(obstacles - B, dim = 1) < d)
        obstacles[mask]
        en = time.time()
        print(en - bn)


if __name__ == "__main__":
    ob = LocalObstacleAvoid(0,0)
    print("Begin")
    a = torch.rand((3000,2), device = 'cuda')
    getTimeDelay(ob.fit_circle, a.cpu().numpy())
    b = torch.rand((3000,2), device = 'cuda')
    getTimeDelay(ob.fit_circle_torch, b)
