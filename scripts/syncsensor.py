#!/usr/bin/python3
import message_filters
from config import *
class Sensors:
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
        ts = message_filters.ApproximateTimeSynchronizer(list(pub_dict.values()), 10, MAX_SENSOR_DELAY)
        ts.registerCallback(self.callback)

    def callback(self, *msgs):
        for name, msg in zip(self.names, msgs):
            self.data[name] = msg
            
