import numpy as np
import math

class NNetwork:

    current_time = 0.0
    num_of_visible = 1.0

    def __init__(self, weights, init_values, num_of_visible, T_LEN = 1000):

        self.weights = weights.copy()

        self.init_values = init_values.copy()
        self.values = self.init_values.copy()

        self.current_time = 0.0

        self.num_of_visible = num_of_visible

        self.trajectory = np.zeros((self.num_of_visible, T_LEN))

    def clear(self):
        self.current_time = 0
        self.values = self.init_values.copy()

    def setVisibleValues(self, values):
        self.values[0:self.num_of_visible] = self.values[:]

    def setVisibleInitValues(self, values):
        self.init_values[0:self.num_of_visible] = self.values[:]
        self.clear()

    def calculate(self, dt):

        predictor = self.values.copy()

        signal_p = 0
        signal_c = 0

        for i in range(0, self.values.shape[0], 1):
            signal_p = np.sum(self.weights[i, :]*self.values[:])
            predictor[i]+=math.atan(signal_p)*dt


        for i in range(0, self.values.shape[0], 1):
            signal_p = np.sum(self.weights[i, :] *   predictor[:])
            signal_c = np.sum(self.weights[i, :] * self.values[:])
            self.values[i]+=math.atan((signal_p + signal_c)*0.5)*dt

        self.current_time += dt

        return self.current_time

    def getState(self):
        return self.values[0:self.num_of_visible]