import math
import numpy

def xtrav(distance, angle):
    xt = math.sin(angle/180*(math.pi)) * distance
    return xt

def ytrav(distance, angle):
    yt = math.cos(angle/180*(math.pi)) * distance
    return yt

def distance(pos_a, pos_b):
    dx = pos_a[0]-pos_b[0]
    dy = pos_a[1]-pos_b[1]
    return math.sqrt(dx**2+dy**2)

def angle_between(p1, p2):
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / numpy.pi+180
    return angle

def angle_normalise(angle):
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle