from numpy import  linalg as LA
import numpy as np
import math

class Pendulum:

    g = 9.8

    alpha = 0
    beta  = 0

    ddt_alpha = 0
    ddt_beta  = 0

    p_alpha = 0
    p_beta  = 0

    mass = 0
    len = 1
    time = 0.0
    last_out_time = 0.0
    curr_index = 0

    out_period = 0.0

    def __init__(self, len1, len2, mass1, mass2, num_of_steps):
        self.len = len1
        self.mass = mass1

        self.history = np.zeros((3,num_of_steps))

    def InitCondisions(self, alpha, beta, ddt_alpha =0, ddt_beta = 0, start_time = 0):
        self.time  = start_time
        self.alpha = alpha
        self.beta  = beta
        self.ddt_alpha = ddt_alpha
        self.ddt_beta  = ddt_beta

        self.p_alpha = 0.0#1.0 / 6.0 * self.mass * pow(self.len, 2) * (8.0 * ddt_alpha + 3 * ddt_beta  * math.cos(self.alpha - self.beta))
        self.p_beta  = 0.0#1.0 / 6.0 * self.mass * pow(self.len, 2) * (2.0 * ddt_beta  + 3 * ddt_alpha * math.cos(self.alpha - self.beta))

        self.curr_index = 0



    def Calc(self, dt):
     #   ddt_a = 6.0 / self.mass / pow(self.len, 2) * (
     #               2.0 * self.p_alpha - 3.0 * math.cos(self.alpha - self.beta) * self.p_beta) / (
     #                                16.0 - 9.0 * pow(math.cos(self.alpha - self.beta), 2))

      #  ddt_b = 6.0 / self.mass / pow(self.len, 2) * (
      #              8.0 * self.p_beta - 3.0 * math.cos(self.alpha - self.beta) * self.p_alpha) / (
      #                               16.0 - 9.0 * pow(math.cos(self.alpha - self.beta), 2))

        ddt_a, ddt_b = self.calcDdtAB(self.alpha, self.beta, self.p_alpha, self.p_beta)

        a  = self.alpha +  ddt_a*dt
        b  = self.beta  +  ddt_b*dt

        ddt_p_alpha, ddt_p_beta = self.calcDdtP((a+self.alpha)/2.0, (b+self.beta)/2.0, ddt_a, ddt_b)

        p_a  = self.p_alpha + ddt_p_alpha* dt
        p_b  = self.p_beta  + ddt_p_beta * dt

       # self.p_alpha = p_a
       # self.p_beta  = p_b

        ddt_a, ddt_b = self.calcDdtAB((a+self.alpha)/2.0, (b+self.beta)/2.0, p_a, p_b)

        ddt_p_alpha, ddt_p_beta = self.calcDdtP((a + self.alpha) / 2.0, (b + self.beta) / 2.0, ddt_a, ddt_b)

        self.p_alpha = self.p_alpha + ddt_p_alpha * dt
        self.p_beta  = self.p_beta  + ddt_p_beta  * dt

        ddt_a, ddt_b = self.calcDdtAB((a + self.alpha) / 2.0, (b + self.beta) / 2.0, self.p_alpha, self.p_beta)

        self.alpha   = self.alpha +  ddt_a*dt
        self.beta    = self.beta  +  ddt_b*dt

        self.time += dt

        if((self.time - self.last_out_time) < self.out_period):
            self.last_out_time = self.time
            self.SaveState()

        return self.time

    def calcDdtP(self, alpha, beta, ddt_alpha, ddt_beta):
        ddt_p_alpha = -1.0 / 2.0 * self.mass * pow(self.len, 2) * (ddt_alpha * ddt_beta * math.sin(
            alpha - beta) + 3 * self.g / self.len * math.sin(alpha))

        ddt_p_beta = -1.0 / 2.0 * self.mass * pow(self.len, 2) * (- ddt_alpha * ddt_beta * math.sin(
            alpha - beta) + self.g / self.len * math.sin(beta))

        return ddt_p_alpha, ddt_p_beta

    def calcDdtAB(self, alpha, beta, p_alpha, p_beta):
        ddt_a = 6.0 / self.mass / pow(self.len, 2) * (
            2.0 * p_alpha - 3.0 * math.cos(alpha - beta) * p_beta) / (
                    16.0 - 9.0 * pow(math.cos(alpha - beta), 2))

        ddt_b = 6.0 / self.mass / pow(self.len, 2) * (
            8.0 * p_beta - 3.0 * math.cos(alpha - beta) * p_alpha) / (
                    16.0 - 9.0 * pow(math.cos(alpha - beta), 2))

        return ddt_a, ddt_b

    def getState(self):
        return (self.alpha, self.beta)

    def setOutputTimestep(self, out_period):
        self.out_period = out_period

    def SaveState(self):
        if(self.curr_index<self.history.shape[1]):
            self.history[0, self.curr_index] = self.time
            self.history[1, self.curr_index] = self.alpha
            self.history[2, self.curr_index] = self.beta
            self.curr_index+=1
        else:
            tmp_array = np.zeros((3,self.history.shape[1]))
            self.history = np.hstack((self.history, tmp_array))

    def getHistory(self):
        return self.history[:,0:self.curr_index]

