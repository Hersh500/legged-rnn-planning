# State space of 1D hopper (simplified):
# [x, y, xvel, yvel, pitch (unused), pitch_vel, leg_input_angle]
# assumption: you can set your leg angle to whatever regardless of the pitch of the robot

# Flight State space of 2D hopper:
# [x, y, z, xvel, yvel, zvel, pitch, roll, pitch_vel, roll_vel, leg_pitch_angle, leg_roll_angle]
# In similar way to previous, ignoring pitch + roll of the body this time.
# container class
class FlightState2D:
    def __init__(self, array):
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
        self.xdot = array[3]
        self.ydot = array[4]
        self.zdot = array[5]

        self.leg_pitch = array[10]
        self.leg_roll = array[11]

        # Unused for now
        self.pitch = array[6]
        self.roll = array[7]
        self.pitch_vel = array[8]
        self.roll_vel = array[9]

    def getDerivatives(self, constants):
        deriv = [0 for i in range(12)]
        deriv[0] = self.xdot
        deriv[1] = self.ydot
        deriv[2] = self.zdot
        
        deriv[3] = 0
        deriv[4] = 0
        deriv[5] = constants.g   # TODO: define this, or pass it in in some way
        
        deriv[10] = 0
        deriv[11] = 0

        # body orientation is unused
        deriv[6] = 0
        deriv[7] = 0
        deriv[8] = 0
        deriv[9] = 0
        
        return deriv
        
    

# Stance state space of 2d hopper; switches to centroidal dynamics
# [pitch, pitch_vel, roll, roll_vel, L, Ldot]
class StanceState2D:
    def __init__(self, array):
        self.pitch = array[0]
        self.pitch_vel = array[1]
        self.roll = array[2]
        self.roll_vel = array[3]
        self.L = array[4]
        self.Ldot = array[5]

    def getDerivatives(self, constants):
        derivs = [0 for i in range(6)]
        derivs[0] = self.pitch_vel
        derivs[1] = (-2 * self.Ldot * self.pitch_vel - np.abs(constants.g) * np.cos(self.pitch))/self.L

        derivs[2] = self.roll_vel
        derivs[3] = (-2 * self.Ldot * self.roll_vel - np.abs(constants.g) * np.cos(self.roll))/self.L

        derivs[4] = self.Ldot
        
        # TODO: how does this work in 2d => vertical component of pitch, roll?
        # derivs[5] = -constants.k/constants.m * (self.L - constants.Lf - constants.Lk0) - np.abs(constants.g) * np.sin()...
         


class Hopper2D:
    def __init__(self, constants):
        self.constants = constants

    def flightToStance():
        return

    def stanceToFlight():
        return
   
    def flightDynamics(self, x):
        state = FlightState2D(x)
        return state.getDerivatives(self.constants)

    def stanceDynamics(self, x):
        state = StanceState2D(x)
        return state.getDerivatives(self.constants)

    def simulateOneFlightPhase(self, x_init, foot_pos, u, tstep, terrain_func, till_apex = False, hit_apex = False, init_from_stance = False):
        if init_from_stance:
            x0_flight = self.stanceToFlight(x_init)
        else:
            x0_flight = x_init

        return

    def simulateOnestancePhase(self):
        return


# How to do this: generate ditches, or islands?
def generateRandomTerrain2d():
