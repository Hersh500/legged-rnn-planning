# How to estimate Tf, Ts?
class RaibertPController:
    def __init__(self, kp):
        self.kp = kp
        self.kd = kd

    def calcAngle(self, vel_des, cur_vel):
        err = vel_des - cur_vel
        # kp * (err)
        return angle


class RaibertPDController:
    def __init__(self, kp, kd):
        return 

    def calcAngle(self, vel_des, cur_vel, err_prev):
        return angle, err
