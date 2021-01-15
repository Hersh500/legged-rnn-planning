import numpy as np
import random
import queue
import math
from scipy.integrate import ode, odeint


### LOW LEVEL ROBOT FUNCTIONS ###
class Constants:
    def __init__(self):
        # self.k = 2800  # Spring Constant, N/m
        self.L = 0.5  # length of leg in flight
        self.Lf = 0.3 # length of unsprung part of leg
        self.g = -9.8
        self.m = 7  # kg
        self.k = 3200
        self.u = 1
        self.Lk0 = 0.2 # length of uncompressed spring
        self.Cd = 0.5
        self.eps = 1e-2


def sample_terrain_func(x):
    terrain_disc = {0:0, 0.3:0, 0.6:0.1, 0.9:0.2, 1.2:-0.5, 1.5:0.3, 1.8:0.3, 2.1:0.2, 2.4:0.3, 2.7: 0, 3.0: 0.2}
    tmp = round(x, 2) * 100
    tmp = (tmp + (30 - tmp % 30))/100
# discretized to 30 cm steps"
    if tmp < 0 or tmp >= 3.3:
        return 0
    return terrain_disc[tmp]

def default_terrain_func(x):
    return 0

'''
Makes sure we are within friction cone bounds and spring hasn't bottomed-out.
'''
def checkStanceFailure(x_stance, foot_pos,
                       terrain_func, terrain_normal_func=None,
                       u=None, debug = True):
  if x_stance[2] < constants.Lf - 0.05:
    if debug:
      print("Spring bottom-ed out:", x_stance[2])
    return sim_codes["SPRING_BOTTOM_OUT"]
  
  if terrain_normal_func is not None and u is not None:
    normal = terrain_normal_func(foot_pos[0])
    forward_limit = normal - np.arctan(u)
    rear_limit = normal + np.arctan(u)

    if x_stance[0] < forward_limit:
      if debug:
        print("Forward Friction cone violation!: leg_angle", x_stance[0], "limit = ", forward_limit)
      return sim_codes["FRICTION_CONE"]
    if x_stance[0] > rear_limit:
      if debug:
        print("Rear Friction cone violation!: leg_angle", x_stance[0], "limit = ", rear_limit)
      return sim_codes["FRICTION_CONE"]

  body_pos = getBodyXYInStance(x_stance, foot_pos)
  
  # This indicates a collision with a wall
  if terrain_func(body_pos[0]) > body_pos[1]:
    if debug:
      print("Collision with wall in stance! Body y", body_pos[1], " < terrain y", terrain_func(body_pos[0]))
    return sim_codes["BODY_CRASH"]
  return sim_codes["SUCCESS"]


'''
Checks if the body or foot collides with any terrain features
- terrain_func is a function that maps x coord to y coord
'''
def checkFlightCollision(x_flight, terrain_func, debug = False):
  body_x = x_flight[0]
  body_y = x_flight[1]
  foot_pos = getFootXYInFlight(x_flight)
  if terrain_func(body_x) > body_y:
    if debug:
      print("body collision in flight phase!")
      print("For body y = ", body_y, "terrain y", terrain_func(body_x))
    return sim_codes["BODY_CRASH"]
  elif terrain_func(foot_pos[0]) > foot_pos[1] + 0.05:
    if debug:
      print("foot collision in flight phase! x = ", foot_pos[0], "y = ", foot_pos[1])
    return sim_codes["FOOT_CRASH"]
  return sim_codes["SUCCESS"]


def getFootXYInFlight(x_flight):
    a = x_flight[5]
    body_x = x_flight[0]
    body_y = x_flight[1]
    foot_x = body_x - np.cos(a) * constants.L
    foot_y = body_y - np.sin(a) * constants.L
    return [foot_x, foot_y]


def getBodyXYInStance(x_stance, foot_pos):
    a = x_stance[0]
    Lb = x_stance[2]

    x_b = foot_pos[0] + np.cos(a) * Lb
    y_b = foot_pos[1] + np.sin(a) * Lb
    return [x_b, y_b]


'''
flight state vector:
 x[0]: CoM x (m)
 x[1]: CoM y (m)
 x[2]: CoM x velocity (m/s)
 x[3]: CoM y velocity (m/s)
 x[4]: angular velocity (rad/s)
 x[5]: leg angle wrt negative x-axis
'''
def flightStep(x, u, tstep):
    com_x_next = x[0] + tstep * x[2]
    com_y_next = x[1] + tstep * x[3] + 0.5*constants.g*(tstep**2)
    x_vel_next = x[2]  # x velocity does not change in flight
    y_vel_next = x[3] + constants.g * tstep  # y velocity decreases by acceleration, which is g
    w_next = x[4]
    foot_angle_next = u
    x_next = [com_x_next, com_y_next, x_vel_next, y_vel_next, w_next, foot_angle_next]
    return x_next


def flightDynamics(t, x):
    derivs = [x[2], x[3], 0, 0, 0, 0]
    derivs[2] = 0
    derivs[3] = constants.g
    return derivs

# calculates leg angle as a function of the current flight state and desired CoM velocity
def legAnglePControl(x_flight, x_vel_des, k, Kp, tst = 0.18):
    x_vel_cur = x_flight[2]
    xs_dot = k*(x_vel_cur + x_vel_des)/2
    # xs_dot = (x_vel_cur + x_vel_des)/2
    leg_fd = - (xs_dot * tst)/2 + Kp * (x_vel_des - x_vel_cur)
    leg_angle = np.arccos(leg_fd/constants.L)

    return leg_angle


'''
stance state vector:
    a: the angle between the leg and the positive x axis
    a_d: the angular velocity of the leg
    Lb: the length of the leg
    Lb_d: the derivative of the length of the leg
'''
# Currently assuming flat ground. Needs to be reworked for mixed terrain.
# Returns actual second derivatives, not the state at the second timestep.
# This is because a closed form solution for the dynamics doesn't exist.
def stanceDynamics(t, x):
    a = x[0]
    a_d = x[1]
    Lb = x[2]
    Lb_d = x[3]

    derivs = [a_d, 0, Lb_d, 0]
    a_dd = (-2 * Lb_d * a_d - np.abs(constants.g) * np.cos(a))/Lb
    Lb_dd = -constants.k/constants.m * (Lb - constants.Lf - constants.Lk0) - np.abs(constants.g) * np.sin(a) + Lb*(a_d**2)
    derivs[1] = a_dd
    derivs[3] = Lb_dd
    return derivs


# converts final flight state into initial stance state upon touchdown
# Also need to save the foot position, since it is assumed stationary during stance phase
def flightToStance(x):
    Lb_init = constants.L
    # x_diff = x[0] - foot_pos[0]
    # y_diff = x[1] - foot_pos[1]
    # Lb_init = np.sqrt(x_diff**2 + y_diff**2)
    a_init = x[5]  
    a_d_init = (-np.sin(a_init) * x[2] + np.cos(a_init) * x[3])/Lb_init
    Lb_d_init = np.cos(a_init) * x[2] + np.sin(a_init) * x[3]
    stance_vec = [a_init, a_d_init, Lb_init, Lb_d_init]
    # print("converted stance energy", stancePhaseEnergy(stance_vec))
    return stance_vec


# converts final stance state to initial flight state on takeoff
def stanceToFlight(x_stance, foot_pos):
    # print("input stance energy", stancePhaseEnergy(x_stance))
    xb = foot_pos[0] + np.cos(x_stance[0]) * x_stance[2]
    yb = foot_pos[1] + np.sin(x_stance[0]) * x_stance[2]
    xb_d = np.cos(x_stance[0]) * x_stance[3] - np.sin(x_stance[0]) * x_stance[1] * x_stance[2]
    yb_d = np.sin(x_stance[0]) * x_stance[3] + np.cos(x_stance[0]) * x_stance[1] * x_stance[2]
    flight_vec = [xb, yb, xb_d, yb_d, 0, x_stance[0]]
    # print("converted flight energy:", flightPhaseEnergy(flight_vec))
    return flight_vec


def simulateOneFlightPhaseODE(x_stance, 
                              foot_pos = None, 
                              x_init = None, 
                              debug = False, 
                              u = np.pi/2, 
                              tstep = 0.01, 
                              terrain_func = default_terrain_func,
                              till_apex = False,
                              print_fails = True,
                              hit_apex = False):
    if debug:
        x0_flight = x_init
    else:
        x0_flight = stanceToFlight(x_stance, foot_pos)

    foot_angle_next = u

    flight_states = [x0_flight]
    integrator = ode(flightDynamics)
    integrator.set_initial_value(x0_flight, 0)
    ret_val = sim_codes["SUCCESS"]
    cond = False
    success = True
    while integrator.successful():
        x = integrator.integrate(integrator.t + tstep)
        state = integrator.y
        # ke = 0.5 * constants.m * (state[2]**2 + state[3]**2)
        # pe = constants.m * np.abs(constants.g) * state[1]
        # print("flight energy:", ke + pe)
        if integrator.y[3] <= 0 and flight_states[len(flight_states)-1][3] >= 0:
            hit_apex = True
            if till_apex:
              flight_states.append(integrator.y)
              return sim_codes["SUCCESS"], flight_states, integrator.t

        cur_state = integrator.y
        if hit_apex:
            cur_state[5] = foot_angle_next
            cond = True

        flight_states.append(cur_state)
        foot_pos = getFootXYInFlight(cur_state)
        if cond and foot_pos[1] <= terrain_func(foot_pos[0]):
            if terrain_func(foot_pos[0] + 0.02) == terrain_func(foot_pos[0] - 0.02):
                break
        if np.abs(cur_state[1] - x0_flight[1]) > 0.05:
            ret_val = checkFlightCollision(cur_state, terrain_func, debug = print_fails)
        if ret_val < 0:  
            break

    return ret_val, flight_states, integrator.t


def getStanceEnergy(x_stance):
  stance_ke = (0.5 * x_stance[1]**2 * x_stance[2]**2 * constants.m + 
              0.5 * x_stance[3]**2 * constants.m)
  stance_pe = (0.5 * constants.k * (x_stance[2] - constants.Lf - constants.Lk0)**2 +
               np.abs(constants.g) * np.sin(x_stance[0]) * (x_stance[2] * constants.m))
  return stance_ke + stance_pe
      
  
def simulateOneStancePhase(x_flight, tstep = 0.01, terrain_func = default_terrain_func, 
                           print_fails = True, terrain_normal_func = None, friction = None):
    x0_stance = flightToStance(x_flight)
    # print("initial stance state:", x0_stance)
    # print("initial stance energy:", getStanceEnergy(x0_stance))
    foot_pos = getFootXYInFlight(x_flight)
    integrator = ode(stanceDynamics)
    integrator.set_initial_value(x0_stance, 0)
    # backend = "dopri5"
    # integrator.set_integrator(backend)
    code = sim_codes["SUCCESS"]
    stance_states = [x0_stance]
    success = True
    while integrator.successful():
        integrator.integrate(integrator.t + tstep)
        # print("stance energy:", getStanceEnergy(integrator.y))
        stance_states.append(integrator.y)
        if integrator.y[2] >= constants.L:
          break
        code = checkStanceFailure(integrator.y,
                                  foot_pos,
                                  terrain_func,
                                  debug = print_fails,
                                  terrain_normal_func = terrain_normal_func,
                                  u = friction)
        if code < 0:
            break
    return code, stance_states, integrator.t




### SIMULATION UTILITIES ###
sim_codes = {"SUCCESS":0, "SPRING_BOTTOM_OUT":-1, "FRICTION_CONE":-2, "A*_NO_PATH":-3, "BODY_CRASH":-4, "FOOT_CRASH":-5, "TOO_MANY_STEPS":-6}
sim_codes_rev = dict(map(reversed, sim_codes.items()))
constants = Constants()

# Wrapper class for low level functions 
class Hopper:
    def __init__(self, constants):
        self.constants = constants
    
    def flightToStance(self, x):
        Lb_init = self.constants.L
        # x_diff = x[0] - foot_pos[0]
        # y_diff = x[1] - foot_pos[1]
        # Lb_init = np.sqrt(x_diff**2 + y_diff**2)
        a_init = x[5]  
        a_d_init = (-np.sin(a_init) * x[2] + np.cos(a_init) * x[3])/Lb_init
        Lb_d_init = np.cos(a_init) * x[2] + np.sin(a_init) * x[3]
        stance_vec = [a_init, a_d_init, Lb_init, Lb_d_init]
        # print("converted stance energy", stancePhaseEnergy(stance_vec))
        return stance_vec


    # converts final stance state to initial flight state on takeoff
    def stanceToFlight(self, x_stance, foot_pos):
        # print("input stance energy", stancePhaseEnergy(x_stance))
        xb = foot_pos[0] + np.cos(x_stance[0]) * x_stance[2]
        yb = foot_pos[1] + np.sin(x_stance[0]) * x_stance[2]
        xb_d = np.cos(x_stance[0]) * x_stance[3] - np.sin(x_stance[0]) * x_stance[1] * x_stance[2]
        yb_d = np.sin(x_stance[0]) * x_stance[3] + np.cos(x_stance[0]) * x_stance[1] * x_stance[2]
        flight_vec = [xb, yb, xb_d, yb_d, 0, x_stance[0]]
        # print("converted flight energy:", flightPhaseEnergy(flight_vec))
        return flight_vec

    '''
    Makes sure we are within friction cone bounds and spring hasn't bottomed-out.
    '''
    def checkStanceFailure(self, x_stance, foot_pos,
                           terrain_func, terrain_normal_func=None,
                           u = None, debug = True):
      if x_stance[2] < self.constants.Lf - 0.05:
        if debug:
          print("Spring bottom-ed out:", x_stance[2])
        return sim_codes["SPRING_BOTTOM_OUT"]
      
      if terrain_normal_func is not None and u is not None:
        normal = terrain_normal_func(foot_pos[0])
        forward_limit = normal - np.arctan(u)
        rear_limit = normal + np.arctan(u)

        if x_stance[0] < forward_limit:
          if debug:
            print("Forward Friction cone violation!: leg_angle", x_stance[0], "limit = ", forward_limit)
          return sim_codes["FRICTION_CONE"]
        if x_stance[0] > rear_limit:
          if debug:
            print("Rear Friction cone violation!: leg_angle", x_stance[0], "limit = ", rear_limit)
          return sim_codes["FRICTION_CONE"]

      body_pos = self.getBodyXYInStance(x_stance, foot_pos)
      
      # This indicates a collision with a wall
      if terrain_func(body_pos[0]) > body_pos[1]:
        if debug:
          print("Collision with wall in stance! Body y", body_pos[1], " < terrain y", terrain_func(body_pos[0]))
        return sim_codes["BODY_CRASH"]
      return sim_codes["SUCCESS"]


    '''
    Checks if the body or foot collides with any terrain features
    - terrain_func is a function that maps x coord to y coord
    '''
    def checkFlightCollision(self, x_flight, terrain_func, debug = False):
      body_x = x_flight[0]
      body_y = x_flight[1]
      foot_pos = self.getFootXYInFlight(x_flight)
      if terrain_func(body_x) > body_y:
        if debug:
          print("body collision in flight phase!")
          print("For body y = ", body_y, "terrain y", terrain_func(body_x))
        return sim_codes["BODY_CRASH"]
      elif terrain_func(foot_pos[0]) > foot_pos[1] + 0.05:
        if debug:
          print("foot collision in flight phase! x = ", foot_pos[0], "y = ", foot_pos[1])
        return sim_codes["FOOT_CRASH"]
      return sim_codes["SUCCESS"]


    def getFootXYInFlight(self, x_flight):
        a = x_flight[5]
        body_x = x_flight[0]
        body_y = x_flight[1]
        foot_x = body_x - np.cos(a) * self.constants.L
        foot_y = body_y - np.sin(a) * self.constants.L
        return [foot_x, foot_y]


    def getBodyXYInStance(self, x_stance, foot_pos):
        a = x_stance[0]
        Lb = x_stance[2]

        x_b = foot_pos[0] + np.cos(a) * Lb
        y_b = foot_pos[1] + np.sin(a) * Lb
        return [x_b, y_b]

    def simulateOneStancePhase(self, x_flight, tstep = 0.01, terrain_func = default_terrain_func, 
                               print_fails = True, terrain_normal_func = None, friction = None):
        x0_stance = self.flightToStance(x_flight)
        foot_pos = self.getFootXYInFlight(x_flight)
        integrator = ode(stanceDynamics)
        integrator.set_initial_value(x0_stance, 0)
        code = sim_codes["SUCCESS"]
        stance_states = [x0_stance]
        success = True
        while integrator.successful():
            integrator.integrate(integrator.t + tstep)
            # print("stance energy:", getStanceEnergy(integrator.y))
            stance_states.append(integrator.y)
            if integrator.y[2] >= self.constants.L:
              break
            code = self.checkStanceFailure(integrator.y,
                                      foot_pos,
                                      terrain_func,
                                      debug = print_fails,
                                      terrain_normal_func = terrain_normal_func,
                                      u = friction)
            if code < 0:
                break
        return code, stance_states, integrator.t

    def simulateOneFlightPhaseODE(self, 
                                  x_stance, 
                                  foot_pos = None, 
                                  x_init = None, 
                                  debug = False, 
                                  u = np.pi/2, 
                                  tstep = 0.01, 
                                  terrain_func = default_terrain_func,
                                  till_apex = False,
                                  print_fails = True,
                                  hit_apex = False):
        if debug:
            x0_flight = x_init
        else:
            x0_flight = self.stanceToFlight(x_stance, foot_pos)

        foot_angle_next = u

        flight_states = [x0_flight]
        integrator = ode(flightDynamics)
        integrator.set_initial_value(x0_flight, 0)
        ret_val = sim_codes["SUCCESS"]
        cond = False
        success = True
        while integrator.successful():
            x = integrator.integrate(integrator.t + tstep)
            state = integrator.y
            # ke = 0.5 * constants.m * (state[2]**2 + state[3]**2)
            # pe = constants.m * np.abs(constants.g) * state[1]
            # print("flight energy:", ke + pe)
            if integrator.y[3] <= 0 and flight_states[len(flight_states)-1][3] >= 0:
                hit_apex = True
                if till_apex:
                  flight_states.append(integrator.y)
                  return sim_codes["SUCCESS"], flight_states, integrator.t

            cur_state = integrator.y
            if hit_apex:
                cur_state[5] = foot_angle_next
                cond = True

            flight_states.append(cur_state)
            foot_pos = self.getFootXYInFlight(cur_state)
            if cond and foot_pos[1] <= terrain_func(foot_pos[0]):
                if terrain_func(foot_pos[0] + 0.02) == terrain_func(foot_pos[0] - 0.02):
                    break
            if np.abs(cur_state[1] - x0_flight[1]) > 0.05:
                ret_val = self.checkFlightCollision(cur_state, terrain_func, debug = print_fails)
            if ret_val < 0:  
                break

        return ret_val, flight_states, integrator.t

# Higher level wrapper functions
def getNextState(hopper, x_flight, u, terrain_func):
    success, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                          x_init = x_flight,
                                                          debug = True,
                                                          u = u,
                                                          terrain_func = terrain_func,
                                                          print_fails = False)
    if not success:
        return False, -1, -1, []
    last_flight_state = flight_states[len(flight_states)-1]
    foot_pos = hopper.getFootXYInFlight(last_flight_state)
    success, stance_states, _ = hopper.simulateOneStancePhase(last_flight_state, terrain_func = terrain_func,
                                                              print_fails = False)
    if not success:
        return False, -1, -1, []
    last_stance_state = stance_states[len(stance_states) - 1]
    success, flight_states, _ = hopper.simulateOneFlightPhaseODE(last_stance_state, 
                                                                 foot_pos, 
                                                                 terrain_func = terrain_func,
                                                                 print_fails = False)
    if not success:
        return False, -1, -1, []
    apex_state = None
    for i in range (len(flight_states)-1):
        if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
            apex_state = flight_states[i]
            break
    return True, foot_pos[0], flight_states[len(flight_states)-1], apex_state


def getNextState2Count(hopper, x_flight, input_angle, terrain_func, terrain_normal_func = None, friction = None,
                  at_apex = False):
  count = 0
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                        x_init = x_flight,
                                                        debug = True,
                                                        u= input_angle,
                                                        terrain_func = terrain_func,
                                                        print_fails = False,
                                                        hit_apex = at_apex)
  count += len(flight_states)
  if code < 0:
    return None, None, None, count

  if at_apex:
    first_apex = x_flight
  else:
    for i in range (len(flight_states)-1):
        if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
            apex_state = flight_states[i]
            first_apex = apex_state
            break
  
  last_flight_state = flight_states[len(flight_states)-1]
  foot_pos = hopper.getFootXYInFlight(last_flight_state)
  first_step_loc = foot_pos[0]
  code, stance_states, _ = hopper.simulateOneStancePhase(last_flight_state,
                                                      terrain_func = terrain_func,
                                                      print_fails = False,
                                                      terrain_normal_func = terrain_normal_func,
                                                      friction = friction)
  count += len(stance_states)
  if code < 0:
    return None, None, None, count
  last_stance_state = stance_states[len(stance_states) - 1]
  x_flight = hopper.stanceToFlight(last_stance_state, foot_pos)
  
  # simulate one more flight phase to check for collision (for robustness)
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                        x_init = x_flight,
                                                        debug = True,
                                                        u= np.pi/2,
                                                        terrain_func = terrain_func,
                                                        print_fails = False)
  second_step = flight_states[-1][0]
  count += len(flight_states)
  if code < 0:
    return None, None, None, count
  for i in range (len(flight_states)-1):
    if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
      apex_state = flight_states[i]
      second_apex = apex_state
      break
  return first_apex, second_apex, flight_states[-1], count

def getNextState2(hopper, x_flight, input_angle, terrain_func, terrain_normal_func = None, friction = None,
                  at_apex = False):
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                        x_init = x_flight,
                                                        debug = True,
                                                        u= input_angle,
                                                        terrain_func = terrain_func,
                                                        print_fails = False,
                                                        hit_apex = at_apex)
  if code < 0:
    return None, None, None

  if at_apex:
    first_apex = x_flight
  else:
    for i in range (len(flight_states)-1):
        if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
            apex_state = flight_states[i]
            first_apex = apex_state
            break
  
  last_flight_state = flight_states[len(flight_states)-1]
  foot_pos = hopper.getFootXYInFlight(last_flight_state)
  first_step_loc = foot_pos[0]
  code, stance_states, _ = hopper.simulateOneStancePhase(last_flight_state,
                                                         terrain_func = terrain_func,
                                                         print_fails = False,
                                                         terrain_normal_func = terrain_normal_func,
                                                         friction = friction)
  if code < 0:
    return None, None, None
  last_stance_state = stance_states[len(stance_states) - 1]
  x_flight = hopper.stanceToFlight(last_stance_state, foot_pos)
  
  # simulate one more flight phase to check for collision (for robustness)
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                        x_init = x_flight,
                                                        debug = True,
                                                        u= np.pi/2,
                                                        terrain_func = terrain_func,
                                                        print_fails = False)
  second_step = flight_states[-1][0]
  if code < 0:
    return None, None, None
  for i in range (len(flight_states)-1):
    if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
      apex_state = flight_states[i]
      second_apex = apex_state
      break
  return first_apex, second_apex, flight_states[-1]

def getNextState3(hopper, x_flight, input_angle, terrain_func, terrain_normal_func = None, friction = None):
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                      x_init = x_flight,
                                                      debug = True,
                                                      u= input_angle,
                                                      terrain_func = terrain_func,
                                                      print_fails = False)
  if code < 0:
    return None, None

  for i in range (len(flight_states)-1):
      if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
          apex_state = flight_states[i]
          first_apex = apex_state
          break
  
  last_flight_state = flight_states[len(flight_states)-1]
  foot_pos = hopper.getFootXYInFlight(last_flight_state)
  first_step_loc = foot_pos[0]
  code, stance_states, _ = hopper.simulateOneStancePhase(last_flight_state,
                                                  terrain_func = terrain_func,
                                                  print_fails = False,
                                                  terrain_normal_func = terrain_normal_func,
                                                  friction = friction)
  if code < 0:
    return None, None
  last_stance_state = stance_states[len(stance_states) - 1]
  x_flight = hopper.stanceToFlight(last_stance_state, foot_pos)
  
  # simulate one more flight phase to check for collision (for robustness)
  code, flight_states, _ = hopper.simulateOneFlightPhaseODE(None, 
                                                    x_init = x_flight,
                                                    debug = True,
                                                    u= np.pi/2,
                                                    terrain_func = terrain_func,
                                                    print_fails = False)
  
  second_step = flight_states[-1][0]
  if code < 0:
    return None, None

  return first_step_loc, second_step


def simulateNSteps(robot, n, initial_apex, input_angles, terrain_func, return_apexes = False,
                   terrain_normal_func = None, friction = None, print_fails = False):
  steps = 0
  step_locs = []
  if len(input_angles) != n:
    return None, None
  x_flight = initial_apex
  apexes = []
  while steps < n:
    code, flight_states, _ = robot.simulateOneFlightPhaseODE(None, 
                                                      x_init = x_flight,
                                                      debug = True,
                                                      u= input_angles[steps],
                                                      terrain_func = terrain_func,
                                                      print_fails = print_fails)
    if code < 0:
      return None, None

    for i in range (len(flight_states)-1):
        if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
            apex_state = flight_states[i]
            apexes.append(apex_state)
            break
    
    last_flight_state = flight_states[len(flight_states)-1]
    foot_pos = hopper.sgetFootXYInFlight(last_flight_state)
    step_locs.append(foot_pos[0])
    code, stance_states, _ = robot.simulateOneStancePhase(last_flight_state,
                                                       terrain_func = terrain_func,
                                                       print_fails = print_fails,
                                                       terrain_normal_func = terrain_normal_func,
                                                       friction = friction)
    steps += 1
    if code < 0:
      return None, None
    last_stance_state = stance_states[len(stance_states) - 1]
    x_flight = hopper.sstanceToFlight(last_stance_state, foot_pos)
  
  # simulate one more flight phase to check for collision (for robustness)
  code, flight_states, _ = robot.simulateOneFlightPhaseODE(None, 
                                                        x_init = x_flight,
                                                        debug = True,
                                                        u= np.pi/2,
                                                        terrain_func = terrain_func,
                                                        print_fails = print_fails)
  
  if code < 0:
    return None, None
  for i in range (len(flight_states)-1):
    if flight_states[i+1][3] <= 0 and flight_states[i][3] >= 0:
      apex_state = flight_states[i]
      apexes.append(apex_state)
      break
  if return_apexes:
    return step_locs, apexes
  else:
    return step_locs
