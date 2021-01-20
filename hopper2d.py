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
    

# Stance state space of 2d hopper; switches to centroidal dynamics
# [L, Ldot, pitch, pitch_vel, roll, roll_vel]
class StanceState2D:
    def __init__(self, array):
        self.L = array[0]
        self.Ldot = array[1]
        self.pitch = array[2]
        self.pitch_vel = array[3]
        self.roll = array[4]
        self.roll_vel = array[5]
         


class Hopper2D:
    def __init__(self, constants):
        self.constants = constants

    def flightToStance():
        return

    def stanceToFlight():
        return
   
    def flightDynamics():
        return

    def stanceDynamics():
        return

    # Ballistic
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
