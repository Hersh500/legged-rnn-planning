import numpy as np
import matplotlib.pyplot as plt
import hopper

class RaibertPController:
    def __init__(self, kp, constants):
        self.kp = kp
        self.constants = constants

    def calcAngle(self, vel_des, cur_vel, tst):
        err = vel_des - cur_vel
        # kp * (err)
        # err is positive --> need to go faster --> foot placed further back
        # err is negative --> neeg to go slower --> foot placed further forward
        foot_pos = (cur_vel * tst)/2 - self.kp * err
        angle = np.pi/2 + np.arcsin(foot_pos/self.constants.L)
        return angle


class RaibertPDController:
    def __init__(self, kp, kd, constants):
        self.kp = kp
        self.kd = kd
        self.constants = constants
        return 

    def calcAngle(self, vel_des, cur_vel, err_prev):
        return angle, err


# test the controller by spawning a robot and providing a desired velocity
def main():
    # initialize some stuff
    constants = hopper.Constants()
    robot = hopper.Hopper(constants)
    terrain_normal_func = lambda x: np.pi/2
    terrain_func = lambda x: 0
    friction = 0.8

    kp = 0.02
    controller = RaibertPController(kp, constants)

    vel_des = 0.5
    done = False
    initial_apex = [0, 1.2, 0, 0, 0, np.pi/2]
    state = initial_apex
    tst = 0.17
    num_steps = 0
    while not done and num_steps < 10:
        cur_vel = state[2]
        angle_des = controller.calcAngle(vel_des, cur_vel, tst) 
        print("Angle des:", angle_des)
        first_apex, second_apex, last_flight = hopper.getNextState2(robot,
                                                        state,
                                                        angle_des,
                                                        terrain_func,
                                                        terrain_normal_func,
                                                        friction,
                                                        at_apex = True) 
        if first_apex is None:
            print("Failed!")
            done = True
        else:
            print("Next State:", second_apex)
            print("Step Location:", last_flight[0])
            state = second_apex
            num_steps += 1
    return

if __name__ == "__main__":
    main()
