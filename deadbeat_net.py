import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeadbeatNet(nn.Module):
    def __init__(self):
        super(DeadbeatNet, self).__init__()
        '''
        self.layers = nn.Sequential(
            torch.nn.Linear(3, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1))
        '''
        self.layers = nn.Sequential(
            torch.nn.Linear(2, 10),
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

    def getVelDes(self, Lstep, xdot, Ts, Tf):
        xfh = (xdot * Ts)/2
        xdot_des = (Lstep - 0.25 * self.ks * Ts * xdot - xfh)/(Tf + 0.25 * self.ks * Ts)
        return xdot_des

    # The extra parameters are for drop-in swappability with StepController
    def calcAngle(self, Lstep, xdot, Ts, Tf, upper_lim = 2.0, lower_lim = 1.0, y = 1.0):
        # xdot_des = self.getVelDes(Lstep, xdot, Ts, Tf)
        # THIS Y IS NECESSARY FOR ACCURACY!
        # point = torch.from_numpy(np.array([xdot, y, Lstep])).float().to(self.device)
        point = torch.from_numpy(np.array([xdot, Lstep])).float().to(self.device)
        leg_angle = self.deadbeat_net(point).detach().cpu().numpy()[0]
        # some saturation
        if leg_angle > upper_lim:
            leg_angle = upper_lim
        if leg_angle < lower_lim:
            leg_angle = lower_lim
        return leg_angle
