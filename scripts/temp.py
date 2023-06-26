import numpy as np
import rospy
from sensor_msgs.msg import Image, LaserScan, CameraInfo
def sensor_callback(laser):
    ranges = laser.ranges
    angle_increment = laser.angle_increment
    start_angle = laser.angle_min
    coords = []

    t = rospy.Time.now().secs
    rospy.loginfo(f"Values recieved at {t}")

    for i, range_measurement in enumerate(ranges):
        angle = start_angle + i * angle_increment
        if angle < 0:
            angle = 2*np.pi + angle
        if not (angle > np.radians(240) and angle < np.radians(241)):
            continue
        x = range_measurement * np.cos(angle)
        y = range_measurement * np.sin(angle)
        coords.append([x, y])
    print(np.array(coords))
rospy.init_node("dummy") 
rospy.Subscriber('/scan', LaserScan, sensor_callback)
rospy.spin()
