import time
import numpy as np
from config import *
import cv2

def ForceThrust():
    pass

def vizImages(image, start_pix, end_pix, pix, clusters, radius, ID = 0):
    for s, e, p, clus, r in zip(start_pix, end_pix, pix, clusters, radius):
        image = cv2.circle(image, p, 3, (0, 0, 255), -1)
        image = cv2.putText(
            image,
            f"D: {round(np.linalg.norm(clus, 3))}",
            p - 5,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
        )
        image = cv2.putText(
            image,
            f"R: {round(r, 3)}",
            p + 10,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
        )
        image = cv2.rectangle(image, s, e, (0, 255, 0), 2)
    if SAVES:
        cv2.imwrite(DATASET_FOLDER + f"/img_{ID}.png", image)
    else:
        cv2.imshow("VizImage", image)


def LineObstacles(points):
    A = points[:-1]
    B = points[1:]
    rel_pose = A - B
    mab = rel_pose[:, 1] / rel_pose[:, 0]
    mpc = -1 / mab
    x = (-A[:, 1] + mab * A[:, 0]) / (mab - mpc)
    y = A[:, 1] + mab * (x - A[:, 0])
    obstacles = np.hstack((np.expand_dims(x, 1), np.expand_dims(y, 1)))

    mask = (np.linalg.norm(obstacles - A, axis=1) < np.linalg.norm(A - B, axis=1)) & (
        np.linalg.norm(obstacles - B, axis=1) < np.linalg.norm(A - B, axis=1)
    )
    obstacles = obstacles[mask]

    return obstacles


def repulsionForce(obstacles, k_rep, rho0):
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


def transformCam(points, h):
    trans_clusters = np.zeros((len(points), 3))
    trans_clusters[:, 2] = points[:, 0]
    trans_clusters[:, 0] = points[:, 1] * -1
    trans_clusters[:, 1] = h
    return trans_clusters


def timeTaken(fxn, *kwargs):
    b = time.time()
    v = fxn(*kwargs)
    e = time.time()

    print(str(fxn.__name__), "->", e - b)

    return v


def transformLidar(x, y):
    # x_diff = 0.055 if np.arctan2(-y , -x) > 0 else -0.055
    # y_diff = -0.08
    x_diff = 0.055 if np.arctan2(-y, -x) > 0 else -0.055
    y_diff = -0.08
    return np.array([-(x + x_diff), -(y + y_diff)])


def transformLidarMulti(x, y):
    x1 = float(input("x1 : "))
    x2 = float(input("x2 : "))
    y_ = float(input("y : "))
    x[x > 0] += x1
    x[x <= 0] += x1
    y += y_
    return np.array([-(x), -(y)])


def fit_circle(points):

    points = np.hstack((points, np.ones((points.shape[0], 1))))
    b = -points[:, 0] ** 2 - points[:, 1] ** 2
    center, _, _, _ = np.linalg.lstsq(points, b, rcond=None)
    cx = center[0] / -2
    cy = center[1] / -2
    r = np.sqrt(cx**2 + cy**2 - center[2])

    return cx, cy, r


def isTruePix(pix, img_shape):
    if pix[0] < 0 or pix[0] > img_shape[0]:
        return False
    if pix[1] < 0 or pix[1] > img_shape[1]:
        return False
    return True


def inFOV(points):
    thetas = np.arctan2(points[:, 1], points[:, 0])
    return (thetas > CAMERA_WIDE_MIN) & (thetas < CAMERA_WIDE_MAX)


def TruePix(pixs, img_shape):
    return (
        (pixs[:, 0] > 0)
        & (pixs[:, 0] < img_shape[1])
        & (pixs[:, 1] > 0)
        & (pixs[:, 1] < img_shape[0])
    )


def pixFrom(cloud):
    pix = np.dot(K, cloud.T).T
    return np.int32(pix[:, :2] / pix[:, 2].reshape(-1, 1))
