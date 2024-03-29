import numpy as np
import math
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from hopper import sim_codes, sim_codes_rev
from heightmap import HeightMap
import hopper
import csv

# State space of 1D hopper (simplified):
# [x, y, xvel, yvel, pitch (unused), pitch_vel, leg_input_angle]
# assumption: you can set your leg angle to whatever regardless of the pitch of the robot

# Flight State space of 2D hopper:
# [x, y, z, xvel, yvel, zvel, pitch, roll, pitch_vel, roll_vel, leg_pitch_angle, leg_roll_angle]
# In similar way to previous, ignoring pitch + roll of the body this time.
# container class
class FlightState2D:
    def __init__(self, array = None):
        if array is None:
            array = [0 for i in range(13)]
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
        self.xdot = array[3]
        self.ydot = array[4]
        self.zdot = array[5]

        self.xf = array[10]
        self.yf = array[11]
        self.zf = array[12]

        # Unused for now--eventually include the full dynamics
        self.pitch = array[6]
        self.roll = array[7]
        self.pitch_vel = array[8]
        self.roll_vel = array[9]
        

    def getDerivatives(self, constants):
        deriv = [0 for i in range(13)]
        deriv[0] = self.xdot
        deriv[1] = self.ydot
        deriv[2] = self.zdot
        
        deriv[3] = 0
        deriv[4] = 0
        deriv[5] = constants.g
        
        deriv[10] = self.xdot
        deriv[11] = self.ydot
        deriv[12] = self.zdot

        # body orientation is unused
        deriv[6] = 0
        deriv[7] = 0
        deriv[8] = 0
        deriv[9] = 0
        
        return deriv

    def getArray(self):
        return [self.x, self.y, self.z, self.xdot, self.ydot, self.zdot, self.pitch,
                self.roll, self.pitch_vel, self.roll_vel, self.xf, self.yf, self.zf]
        
    

# Stance state space of 2d hopper; switches to centroidal dynamics
# [x - xf, y - yf, z - zf, (x - xf)dot, (y - yf)dot, (z - zf)dot, xf, yf, zf]
# Dynamics from "Lateral stability of the spring-mass hopper suggests a two-step control strategy for running"
class StanceState2D:
    def __init__(self, array = None):
        if array is None:
            array = [0 for i in range(9)]

        self.xhat = array[0]
        self.yhat = array[1]
        self.zhat = array[2]

        self.xhat_vel = array[3]
        self.yhat_vel = array[4]
        self.zhat_vel = array[5]

        self.xf = array[6]
        self.yf = array[7]
        self.zf = array[8]

    def getDerivatives(self, constants):
        derivs = [0 for i in range(9)]
        derivs[0] = self.xhat_vel
        derivs[1] = self.yhat_vel
        derivs[2] = self.zhat_vel

        tot_len = np.sqrt(self.xhat**2 + self.yhat**2 + self.zhat**2)
        term = constants.k * np.abs(tot_len - constants.Lf - constants.Lk0)/(constants.m * tot_len)
        derivs[3] = term * self.xhat
        derivs[4] = term * self.yhat
        derivs[5] = constants.g + term * self.zhat

        derivs[6] = 0
        derivs[7] = 0
        derivs[8] = 0
        return derivs


class Hopper2D:
    def __init__(self, constants):
        self.constants = constants


    def getFootPosFromAngles(self, state, u):
        # foot_x = state.x + self.constants.L * np.sin(u[0]) * np.cos(u[1])
        # foot_y = state.y + self.constants.L * np.sin(u[0]) * np.sin(u[1])
        foot_x = state.x + self.constants.L * np.sin(u[0])
        foot_y = state.y + self.constants.L * np.sin(u[1])
        if self.constants.L**2 - (state.x - foot_x)**2 - (state.y - foot_y)**2 < 0:
            print("less than zero error!")
        foot_z = state.z - np.sqrt(self.constants.L**2 - (state.x - foot_x)**2 - (state.y - foot_y)**2)  # Hack, since L, x, and y should determine z
        return foot_x, foot_y, foot_z
        

    # takes in flight state and converts to stance state when foot touches down
    # conversion assumes that it happens at the instant that the foot touches the ground
    def flightToStance(self, x_flight):
        state = FlightState2D(x_flight)
        # foot_x, foot_y, foot_z = self.getFootPos(state)
        stance_state = [0 for i in range(9)]

        stance_state[0] = state.x - state.xf
        stance_state[1] = state.y - state.yf
        stance_state[2] = state.z - state.zf

        stance_state[3] = state.xdot
        stance_state[4] = state.ydot
        stance_state[5] = state.zdot

        stance_state[6] = state.xf
        stance_state[7] = state.yf
        stance_state[8] = state.zf

        return stance_state


    # takes in stance state and does change of coordinates to convert to flight when 
    # (x**2 + y**2 + z**2) >= lf**2
    # conversion assumes that it happens at the instant that that sum is equal to lf**2
    def stanceToFlight(self, x_stance):
        state = StanceState2D(x_stance)
        flight_state = [0 for i in range(13)]
        flight_state[0] = state.xhat + state.xf
        flight_state[1] = state.yhat + state.yf
        flight_state[2] = state.zhat + state.zf
        flight_state[3] = state.xhat_vel
        flight_state[4] = state.yhat_vel
        flight_state[5] = state.zhat_vel

        flight_state[6] = 0
        flight_state[7] = 0
        flight_state[8] = 0
        flight_state[9] = 0

        flight_state[10] = state.xf
        flight_state[11] = state.yf
        flight_state[12] = state.zf
        return flight_state
   
    def flightDynamics(self, t, x):
        state = FlightState2D(x)
        return state.getDerivatives(self.constants)

    def checkFlightCollision(self, state, hmap):
        if state.z < hmap.at(state.x, state.y):
            return sim_codes["BODY_CRASH"]
        if state.zf < hmap.at(state.xf, state.yf) - 0.03:
            return sim_codes["FOOT_CRASH"]
        return sim_codes["SUCCESS"]

    def checkStanceCollision(self, state, hmap):
        # check friction cone violation
        # TODO: in the future, need to actually use terrain_normal_func, currently assuming it's np.pi/2
        hyp = np.sqrt(state.xhat**2 + state.yhat**2)
        angle = np.arctan(hyp/state.zhat)
        r_lim = np.arctan(hmap.info["friction"])
        if angle > r_lim:
            return sim_codes["FRICTION_CONE"]
    
        # check body crash
        if state.zhat + state.zf < hmap.at(state.xhat + state.xf, state.yhat + state.yf):
            return sim_codes["BODY_CRASH"]

        tot_length = np.sqrt(state.xhat**2 + state.yhat**2 + state.zhat**2)
        # check spring bottoming out
        if tot_length < self.constants.Lf - 0.02:
            return sim_codes["SPRING_BOTTOM_OUT"]

        return sim_codes["SUCCESS"]

    def stanceDynamics(self, t, x):
        state = StanceState2D(x)
        return state.getDerivatives(self.constants)


    def simulateOneFlightPhase(self, x_init, u, hmap, till_apex = False, hit_apex = False, init_from_stance = False, tstep = 0.01):
        if init_from_stance:
            x0_flight = self.stanceToFlight(x_init)
        else:
            x0_flight = x_init

        flight_states = [x0_flight]
        state = FlightState2D(x0_flight)
        integrator = ode(self.flightDynamics)
        integrator.set_initial_value(x0_flight, 0)
        ret_val = sim_codes["SUCCESS"]

        while integrator.successful():
            x = integrator.integrate(integrator.t + tstep)
            if x is None:
                print("None state!")
            else:
                for i in range(0, len(x)):
                    if np.isnan(x[i]) or x[i] is None:
                        print("none element!")
            prev_state = state
            state = FlightState2D(integrator.y)
            if state.zdot < 0 and prev_state.zdot >= 0:
                hit_apex = True
                if till_apex:
                  flight_states.append(integrator.y)
                  return sim_codes["SUCCESS"], flight_states, integrator.t

            if hit_apex:
                foot_x, foot_y, foot_z = self.getFootPosFromAngles(state, u)
                state.xf = foot_x
                state.yf = foot_y
                if np.isnan(foot_z) or np.isnan(foot_x) or np.isnan(foot_y):
                    print("Got nan!")
                    ret_val = sim_codes["FOOT_CRASH"]
                    break
                state.zf = foot_z

            flight_states.append(state.getArray())

            cur_ter_loc = hmap.at(state.xf, state.yf)
            flat_check = (hmap.at(state.xf + 0.05, state.yf + 0.05) == cur_ter_loc and
                    hmap.at(state.xf - 0.05, state.yf - 0.05) == cur_ter_loc and
                    hmap.at(state.xf - 0.05, state.yf) == cur_ter_loc and
                    hmap.at(state.xf + 0.05, state.yf) == cur_ter_loc and
                    hmap.at(state.xf, state.yf - 0.05) == cur_ter_loc and
                    hmap.at(state.xf, state.yf + 0.05) == cur_ter_loc)

            # This condition checks if the robot landed
            if hit_apex and state.zf <= hmap.at(state.xf, state.yf) and flat_check:
                break

            # ensures we've actually taken off before checking for collision
            ret_val = self.checkFlightCollision(state, hmap)
            if ret_val < 0:
                break
        if not integrator.successful():
            print("unsuccessful integration!")
            ret_val = sim_codes["BODY_CRASH"]

        return ret_val, flight_states, integrator.t


    def simulateOneStancePhase(self, last_flight, hmap, tstep = 0.01):
        x0_stance = self.flightToStance(last_flight)
        integrator = ode(self.stanceDynamics)
        integrator.set_initial_value(x0_stance, 0)
        code = sim_codes["SUCCESS"]
        stance_states = [x0_stance]
        while integrator.successful():
            integrator.integrate(integrator.t + tstep)
            if integrator.y is None:
                print("None state!")
            else:
                for i in range(0, len(integrator.y)):
                    if np.isnan(integrator.y[i]) or integrator.y[i] is None:
                        print("none element!")
            stance_states.append(integrator.y)
            state = StanceState2D(integrator.y)
            length = state.xhat**2 + state.yhat**2 + state.zhat**2
            if length >= self.constants.L**2:
              break
            code = self.checkStanceCollision(state, hmap)
            if code < 0:
                break
        return code, stance_states, integrator.t


# Returns the success, next apex, touchdown location (last flight state)
def getNextApex2D(robot, x_flight, angles, hmap, at_apex = True, return_count = False):
    code, flight_states1, t = robot.simulateOneFlightPhase(x_flight, angles, hmap, till_apex = False, hit_apex = at_apex, init_from_stance = False, tstep = 0.01)
    if code < 0:
        if return_count:
            return code, [], [], len(flight_states1)
        else:
            return code, [], []

    code, stance_states, t_stance = robot.simulateOneStancePhase(flight_states1[-1], hmap, tstep = 0.01)
    if code < 0:
        if return_count:
            return code, [], [], len(flight_states1) + len(stance_states)
        else:
            return code, [], []

    code, flight_states2, t_flight = robot.simulateOneFlightPhase(stance_states[-1], [0, 0], hmap, till_apex = True, hit_apex = False, init_from_stance = True, tstep = 0.01)
    if code < 0:
        if return_count:
            return code, [], [], len(flight_states1) + len(stance_states) + len(flight_states2)
        else:
            return code, [], []

    code, flight_states3, t_flight = robot.simulateOneFlightPhase(flight_states2[-1], [0, 0], hmap, till_apex = False, hit_apex = True, init_from_stance = False, tstep = 0.01)
    if code < 0:
        if return_count:
            return code, [], [], len(flight_states1) + len(stance_states) + len(flight_states2) + len(flight_states3)
        else:
            return code, [], []

    if return_count:
        return code, flight_states3[-1], flight_states2[-1], len(flight_states1) + len(stance_states) + len(flight_states2) + len(flight_states3)
    else:
        return code, flight_states3[-1], flight_states2[-1]
    

## TODO: convert all below to use heightmaps, and to convert 2D A* to use heightmaps
## and to convert RNN evaluation code to use heightmaps
## and to save datasets as pickles of the class
## --think about how to save this stuff--is the pickling overhead actually that bad?

# ditch_info is a 2D array:
# [[ditch_x, ditch_y, x_width, y_width]]
def generateTerrain2D(params, feature_info):
    disc = params["disc"]
    terrain_array = np.zeros((int(params["max_y"]/disc), int(params["max_x"]/disc)))
    for feat in feature_info:
        x_idx = int(feat[0]/disc)
        y_idx = int(feat[1]/disc)
        x_end = int((feat[0] + feat[2])/disc)
        y_end = int((feat[1] + feat[3])/disc)
        terrain_array[y_idx:y_end, x_idx:x_end] = feat[4]
    return HeightMap(terrain_array, params)


# disc is the length of one side of each square (discretized)
def generateRandomStepTerrain2D(params, num_ditches):
    # max width in any single dimension
    max_width = 2
    min_width = 1

    max_height = 0.6
    min_height = 0.1

    disc = params["disc"]
    x_min = params["corner_val_m"][0]
    y_min = params["corner_val_m"][1]

    terrain_array = np.zeros((int(params["max_y"]/disc), int(params["max_x"]/disc)))
    hmap = HeightMap(terrain_array, params)
    prev_ditch_end_x = params["corner_val_m"][0]
    prev_ditch_end_y = params["corner_val_m"][1]
    for _ in range(num_ditches):
        cur_ditch_x = np.random.uniform(x_min + 1, params["max_x"])
        cur_ditch_y = np.random.uniform(y_min, params["max_y"])
        cur_ditch_x_width = np.random.uniform(min_width, max_width)
        cur_ditch_y_width = np.random.uniform(min_width, max_width)
        
        prev_ditch_end_x = cur_ditch_x + cur_ditch_x_width
        prev_ditch_end_y = cur_ditch_y + cur_ditch_y_width
        start_row, start_col = hmap.m_to_idx((cur_ditch_x, cur_ditch_y))
        end_row, end_col = hmap.m_to_idx((prev_ditch_end_x, prev_ditch_end_y))
        hmap.terrain_array[start_row:end_row, start_col:end_col] = np.random.uniform(min_height, max_height)
    return hmap


def generateGaps2D(max_x, max_y, disc, num_ditches, gap_lims = (0.1, 0.3)):
    min_width = gap_lims[0]
    max_width = gap_lims[1]
    return terrain_array, terrain_func


def generateRandomTerrain2D(params, num_ditches):
    max_width = 2
    min_width = 0.6
    disc = params["disc"]
    x_min = params["corner_val_m"][0]
    y_min = params["corner_val_m"][1]

    terrain_array = np.zeros((int((params["max_y"] - y_min)/disc), int((params["max_x"] - x_min)/disc)))
    hmap = HeightMap(terrain_array, params)
    for _ in range(num_ditches):
        cur_ditch_x = np.random.uniform(x_min + 1, params["max_x"])
        cur_ditch_y = np.random.uniform(y_min, params["max_y"])
        cur_ditch_x_width = np.random.uniform(min_width, max_width)
        cur_ditch_y_width = np.random.uniform(min_width, max_width)
        
        prev_ditch_end_x = cur_ditch_x + cur_ditch_x_width
        prev_ditch_end_y = cur_ditch_y + cur_ditch_y_width
        start_row, start_col = hmap.m_to_idx((cur_ditch_x, cur_ditch_y))
        end_row, end_col = hmap.m_to_idx((prev_ditch_end_x, prev_ditch_end_y))
        hmap.terrain_array[start_row:end_row, start_col:end_col] = -1  # TODO: vary this depth
    return hmap


# Need to save actual heightmap, upper left corner, and discretization.
def saveTerrainAsCSV(path, terrain_array, corner_value_m, disc_m, friction):
    with open(path, 'w', newline="\n") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for row in range(terrain_array.shape[0]):
            csvwriter.writerow(terrain_array[row])
        # delimiter and metadata
        csvwriter.writerow(['D'])
        csvwriter.writerow([disc_m, friction, corner_value_m[0], corner_value_m[1]])
    return


def main():
    robot = Hopper2D(hopper.Constants())
    state = FlightState2D()
    state.x = 0
    state.y = 0
    state.z = 1.1
    state.zf = state.z - robot.constants.L
    state.xdot = 0
    friction = 0.8

    x0_flight = state.getArray()
    terrain_func = lambda x,y: 0
    terrain_normal_func = lambda x,y: np.pi/2
    u = [0, np.pi/20]
    print(u)
    code, flight_states, t_flight = robot.simulateOneFlightPhase(x0_flight,
                                                          u,
                                                          terrain_func,
                                                          till_apex = False,
                                                          hit_apex = True,
                                                          init_from_stance = False)
    print(sim_codes_rev[code])
    last_flight_state = flight_states[-1]
    code, stance_states, t_stance = robot.simulateOneStancePhase(last_flight_state,
                                                          terrain_func,
                                                          terrain_normal_func, friction)
    print(sim_codes_rev[code])
    
    last_stance_state = stance_states[-1]
    code, flight_states2, t_flight2 = robot.simulateOneFlightPhase(last_stance_state,
                                                          u,
                                                          terrain_func,
                                                          till_apex = False,
                                                          hit_apex = False,
                                                          init_from_stance = True)
    print(sim_codes_rev[code])
    '''
    print("----Flight Phase----")
    for state in flight_states:
        print(state)

    print("----Stance Phase----")
    for state in stance_states:
        print(state)
    '''

    fig, axs = plt.subplots(3, 1)
    zs = np.concatenate((np.array(flight_states)[:,2],
                         np.array(stance_states)[:,2],
                         np.array(flight_states2)[:,2]))
    xs = np.concatenate((np.array(flight_states)[:,0],
                         np.array(stance_states)[:,0] + np.array(stance_states)[:,-3],
                         np.array(flight_states2)[:,0]))
    ys = np.concatenate((np.array(flight_states)[:,1],
                         np.array(stance_states)[:,1] + np.array(stance_states)[:,-2],
                         np.array(flight_states2)[:,1]))

    f_t = np.arange(0, t_flight + t_stance + t_flight2 + 0.01, 0.01)
    axs[0].scatter(f_t, xs[:f_t.shape[0]], color = "blue")
    axs[1].scatter(f_t, ys[:f_t.shape[0]], color = "blue")
    axs[2].scatter(f_t, zs[:f_t.shape[0]], color = "blue")
    plt.show()

def main2():
    terrain_array, terrain_func = generateRandomTerrain2D(8, 4, 0.25, 5)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotTerrain2D(ax, terrain_array, 0.25)
    plt.show()

if __name__ == "__main__":
    terrain_array, terrain_func = generateTerrain2D(8, 4, 0.2, [[3, 0, 0.4, 4]])
    saveTerrainAsCSV("test_terrain_save.csv", terrain_array, (0, 0), 0.2, 0.8)
