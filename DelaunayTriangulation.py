import math

import dlib
import cv2
import face_recognition
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import image
from matplotlib import pyplot as plt

from numpy import array
from scipy.linalg import svd
path = "/home/fateme/Documents/archive/images/images/train/angry/186.jpg"


class delaunay:
    def __int__(self):
        pass

    def get_landmarks(self, path):
        try:
            input_image = face_recognition.load_image_file(path)
        except:
            print("An exception occurred")
            return []
        try:
            face_landmarks = face_recognition.face_landmarks(input_image)
        except:
            print("An exception occurred")
            return []
        landmark_points = []
        if face_landmarks == {} or face_landmarks == []:
            print("kk")
            return []
        for key in face_landmarks[0]:
            # print(face_landmarks[0][key])
            for point in face_landmarks[0][key]:
                landmark_points.append(point)
        return landmark_points



    def dot(self, vA, vB):
        return vA[0] * vB[0] + vA[1] * vB[1]

    def ang(self, lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
        vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
        # Get dot prod
        dot_prod = self.dot(vA, vB)
        # Get magnitudes
        magA = self.dot(vA, vA) ** 0.5
        magB = self.dot(vB, vB) ** 0.5
        # Get cosine value
        cos_ = dot_prod / magA / magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod / magB / magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle) % 360

        if ang_deg - 180 >= 0:
            # As in if statement
            return 360 - ang_deg
        else:

            return ang_deg

    def delaunay_tirangulation(self,path):
        landmark_points = self.get_landmarks(path)
        if landmark_points == []:
            return []
        tri = Delaunay(landmark_points)
        triangles_features = []
        for triangle in tri.simplices:

            p = landmark_points[triangle[0]]
            q = landmark_points[triangle[1]]
            r = landmark_points[triangle[2]]
            x = round((p[0] + q[0] + r[0]) / 3, 2)
            y = round((p[1] + q[1] + r[1]) / 3, 2)
            # triangles_features.append(p[0])
            # triangles_features.append(p[1])
            # triangles_features.append(q[0])
            # triangles_features.append(q[1])
            # triangles_features.append(r[0])
            # triangles_features.append(r[1])
            triangles_features.append(x)
            triangles_features.append(y)

            # angle
            # q_angle = self.ang([p, q], [r, q])
            # p_angle = self.ang([p, r], [p, q])
            # r_angle = 180 - q_angle - p_angle
            # triangles_features.append(q_angle)
            # triangles_features.append(p_angle)
            # triangles_features.append(r_angle)
            # triangles_features.append([int(x),int(y)])#[p, q, r, (x,y)])
        return triangles_features