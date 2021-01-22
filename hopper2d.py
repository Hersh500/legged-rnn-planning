import numpy as np
import math
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt

from hopper import sim_codes
import hopper

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

        # Unused for now
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
        foot_x = state.x + self.constants.L * np.sin(u[0]) * np.cos(u[1])
        foot_y = state.y + self.constants.L * np.sin(u[0]) * np.sin(u[1])
        foot_z = state.z - np.sqrt(self.constants.L**2 - (state.x - foot_x)**2 - (state.y - foot_y)**2)  # Hack, since Lf, x, and y should determine z
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

    def checkFlightCollision(self, state, terrain_func):
        # TODO: implement this.
        return sim_codes["SUCCESS"]

    def checkStanceCollision(self, state, terrain_func, terrain_normal_func, friction):
        # TODO: implement this.
        return sim_codes["SUCCESS"]

    def stanceDynamics(self, t, x):
        state = StanceState2D(x)
        return state.getDerivatives(self.constants)


    def simulateOneFlightPhase(self, x_init, u, terrain_func, till_apex = False, hit_apex = False, init_from_stance = False, tstep = 0.01):
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
                state.zf = foot_z

            flight_states.append(state.getArray())

            flat_check = (terrain_func(state.xf + 0.02, state.yf + 0.02) == terrain_func(state.xf, state.yf) and
                    terrain_func(state.xf - 0.02, state.yf - 0.02) == terrain_func(state.xf, state.yf))

            # This condition checks if the robot landed
            if hit_apex and state.zf <= terrain_func(state.xf, state.yf) and flat_check:
                break

            # ensures we've actually taken off before checking for collision
            ret_val = self.checkFlightCollision(state, terrain_func)
            if ret_val < 0:
                break
            if state.z < 0:
                print("crossed ground!")
                break

        return ret_val, flight_states, integrator.t


    def simulateOneStancePhase(self, last_flight, terrain_func, terrain_normal_func, friction, tstep = 0.01):
        x0_stance = self.flightToStance(last_flight)
        integrator = ode(self.stanceDynamics)
        integrator.set_initial_value(x0_stance, 0)
        code = sim_codes["SUCCESS"]
        stance_states = [x0_stance]
        while integrator.successful():
            integrator.integrate(integrator.t + tstep)
            stance_states.append(integrator.y)
            state = StanceState2D(integrator.y)
            length = state.xhat**2 + state.yhat**2 + state.zhat**2
            if length >= self.constants.L**2:
              break
            code = self.checkStanceCollision(state,
                                          terrain_func,
                                          terrain_normal_func = terrain_normal_func,
                                          friction = friction)
            if code < 0:
                break
        return code, stance_states, integrator.t


# How to do this: generate ditches, or islands?
def generateRandomTerrain2d():
    return


def main():
    robot = Hopper2D(hopper.Constants())
    state = FlightState2D()
    state.x = 0
    state.y = 0
    state.z = 1.5
    state.zf = 1.0
    state.xdot = 1

    x0_flight = state.getArray()
    terrain_func = lambda x,y: 0
    terrain_normal_func = lambda x,y: np.pi/2
    u = [np.pi/10, 0]
    code, flight_states, t_flight = robot.simulateOneFlightPhase(x0_flight,
                                                          u,
                                                          terrain_func,
                                                          till_apex = False,
                                                          hit_apex = True,
                                                          init_from_stance = False)
    last_flight_state = flight_states[-1]
    code, stance_states, t_stance = robot.simulateOneStancePhase(last_flight_state,
                                                          terrain_func,
                                                          terrain_normal_func, 0.8)
    
    last_stance_state = stance_states[-1]
    code, flight_states2, t_flight2 = robot.simulateOneFlightPhase(last_stance_state,
                                                          u,
                                                          terrain_func,
                                                          till_apex = False,
                                                          hit_apex = False,
                                                          init_from_stance = True)
    '''
    print("----Flight Phase----")
    for state in flight_states:
        print(state)

    print("----Stance Phase----")
    for state in stance_states:
        print(state)
    '''

    fig, axs = plt.subplots(2, 1)
    zs = np.concatenate((np.array(flight_states)[:,2], np.array(stance_states)[:,2], np.array(flight_states2)[:,2]))
    xs = np.concatenate((np.array(flight_states)[:,0], np.array(stance_states)[:,0], np.array(flight_states2)[:,0]))

    f_t = np.arange(0, t_flight + t_stance + t_flight2 + 0.01, 0.01)
    axs[0].scatter(f_t, xs[:f_t.shape[0]], color = "blue")
    axs[1].scatter(f_t, zs[:f_t.shape[0]], color = "blue")
    plt.show()

if __name__ == "__main__":
    main()
