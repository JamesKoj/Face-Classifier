import math
import numpy as np


class Face:

    def __init__(self, filename, emotion):
        self.filename = filename
        self.emotion = emotion
        self.detected = None
        self.landmarks = []

    def get_landmarks(self):
        return self.landmarks

    def get_emotion(self):
        return self.emotion

    def get_filename(self):
        return self.filename

    def get_detected(self):
        return self.detected

    def set_detected(self, bool_detected):
        self.detected = bool_detected

    def set_landmarks(self, shape):
        x_list = []
        y_list = []

        # Store X and Y coordinates in two lists
        for i in range(1, 68):
            x_list.append(float(shape.part(i).x))
            y_list.append(float(shape.part(i).y))
        x_central = [(x - np.mean(x_list)) for x in x_list]
        y_central = [(y - np.mean(y_list)) for y in y_list]

        for x, y, w, z in zip(x_central, y_central, x_list, y_list):
            self.landmarks.append(w)
            self.landmarks.append(z)
            distribution = np.linalg.norm(np.asarray((z, w)) - np.asarray((np.mean(y_list), np.mean(x_list))))
            self.landmarks.append(distribution)
            self.landmarks.append((math.atan2(y, x) * 360) / (2 * math.pi))