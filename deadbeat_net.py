import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeadbeatNet(nn.Module):
    def __init__(self):
        super(DeadbeatNet, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(3, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1))

    def forward(self, x):
        return self.layers(x)

class DeadbeatStepController:
    def __init__(self, constants, deadbeat_net, ks, device):
        self.constants = constants
        self.deadbeat_net = deadbeat_net
        self.ks = ks
        self.device = device

    # Unused
    def getVelDes(self, Lstep, xdot, Ts, Tf):
        xfh = (xdot * Ts)/2
        xdot_des = (Lstep - 0.25 * self.ks * Ts * xdot - xfh)/(Tf + 0.25 * self.ks * Ts)
        return xdot_des

    # The extra parameters are for drop-in swappability with StepController
    # Need to refactor--StepController is never used 
    def calcAngle(self, Lstep, xdot, Ts, Tf, upper_lim = 2.0, lower_lim = 1.0, y = 1.0):
        point = torch.from_numpy(np.array([xdot, y, Lstep])).float().to(self.device)
        leg_angle = self.deadbeat_net(point).detach().cpu().numpy()[0]
        # some saturation
        if leg_angle > upper_lim:
            leg_angle = upper_lim
        if leg_angle < lower_lim:
            leg_angle = lower_lim
        return leg_angle

# Use this to add some noise (if desired) to the StepController
class PerturbedStepController:
    def __init__(self, base_controller, noise_std):
        self.std = noise_std
        self.controller = base_controller

    def calcAngle(self, Lstep, xdot, Ts, Tf, upper_lim = 2.0, lower_lim = 1.0, y = 1.0):
        noise = np.random.randn() * self.std
        return self.controller.calcAngle(Lstep, xdot, Ts, Tf, upper_lim, lower_lim, y) + noise


### 2D case ###
class DeadbeatStepController2D:
    def __init__(self, constants, pitch_net, roll_net, device):
        self.constants = constants
        self.pitch_net = pitch_net
        self.roll_net = roll_net
        self.device = device

    # TODO: copy this to colab notebook
    def calcAngle(self, x_Lstep, xdot, y_Lstep, ydot, z):
        x_point = torch.from_numpy(np.array([xdot, x_Lstep])).float().to(self.device)
        y_point = torch.from_numpy(np.array([ydot, y_Lstep])).float().to(self.device)
        pitch = self.pitch_net(x_point).detach().cpu().numpy()[0]
        roll = self.roll_net(y_point).detach().cpu().numpy()[0]
        return [pitch, roll]
        
