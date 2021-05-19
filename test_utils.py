import terrain_utils
import numpy as np
import hopper2d
import hopper


### TESTING CLASSES ###
class TestMetrics:
  def __init__(self, percent_success = 0,
               mean_steps = 0,
               mean_distance = 0,
               success_matrix = [],
               failure_cases = {},
               num_odes = 0, 
               avg_time = 0,
               info = None):
    self.percent_success = percent_success
    self.mean_steps = mean_steps
    self.mean_distance = mean_distance
    self.failure_cases = failure_cases
    self.success_matrix = success_matrix
    self.num_odes = num_odes
    self.avg_time = avg_time
    self.info = info

  def printMetrics(self):
    print("*** RESULTS ***")
    print("Success percentage:")
    print(self.percent_success)
    print("Mean steps:")
    print(self.mean_steps)
    print("Mean distance:")
    print(self.mean_distance)
    print("Failure Cases:")
    print(self.failure_cases)
    print("Average number of ODE calls per success:")
    if self.percent_success > 0:
        print(self.num_odes / (self.percent_success * len(self.success_matrix) * len(self.success_matrix[0])))
    print("Average time to generate plan:")
    print(self.avg_time)
    print("Success Matrix:")
    print(self.success_matrix)


class TestMatrix:
  def __init__(self, arrays, apexes, profile = None,
               disc = None, max_x = None, max_y = None):
    self.arrays = arrays
    self.apexes = apexes
    self.terrain_profile = profile
    self.disc = disc
    self.max_x = max_x
    self.max_y = max_y
  
  def getFunctions(self):
    funcs = []
    for t_arr in self.arrays:
      funcs.append(terrain_utils.getTerrainFunc(t_arr, disc = 0.1, min_lim = 0))
    return funcs
    
  # Use in the 2D case.
  def getFunctions2D(self):
    funcs = []
    if self.disc is None or self.max_x is None or self.max_y is None:
        print("error getting the functions!")
        return []
    for t_arr in self.arrays:
      funcs.append(terrain_utils.getTerrainFunc2D(t_arr, disc = self.disc, max_x = self.max_x, max_y = self.max_y))
    return funcs

def generateTestMatrix(ditch_profile, num_apexes, max_vel = 2.5):
  # first generate a bunch of initial apexes with increasing initial velocity
  # then generate a bunch of island terrains with an increasing number of islands
  max_y = 1.5
  min_y = 1.0

  terrain_arrays = []
  terrain_functions = []
  initial_apexes = []
  vels = np.linspace(0, max_vel, num_apexes)
  for i in range(num_apexes):
    initial_apexes.append(np.array([0, np.random.rand() * (max_y - min_y) + min_y,
                                vels[i], 0, 0, np.pi/2]))
  
  for num_ditches in ditch_profile:
    ta, tf = terrain_utils.generateIslandsByDitches(num_ditches, until = 8)
    terrain_arrays.append(ta)
    terrain_functions.append(tf)
  
  matrix = TestMatrix(terrain_arrays, initial_apexes, ditch_profile)
  return matrix

def generateStepTestMatrix(step_profile, num_apexes, max_vel = 2.5):
  max_y = 1.5
  min_y = 1.0

  terrain_arrays = []
  terrain_functions = []
  initial_apexes = []
  vels = np.linspace(0, max_vel, num_apexes)
  for i in range(num_apexes):
    initial_apexes.append(np.array([0, np.random.rand() * (max_y - min_y) + min_y,
                                vels[i], 0, 0, np.pi/2]))
  
  for num_steps in step_profile:
    steps_dict = terrain_utils.generateStairsParams(num_steps, until = 8)
    ta, tf = terrain_utils.generateTerrain2(ditch_locs = {}, step_locs = steps_dict, until = 8)
    terrain_arrays.append(ta)
    terrain_functions.append(tf)
  
  step_matrix = TestMatrix(terrain_arrays, initial_apexes, step_profile)
  return step_matrix


##################
# Parallels for 2D 
##################
def generateDitchTestMatrix2D(ditch_profile, num_apexes, max_x_vel, max_y_vel, array_params):
    initial_apexes = []
    arrays = []
    max_x = array_params[0]
    max_y = array_params[1]
    disc = array_params[2]
    x_vels = np.linspace(0, max_x_vel, num_apexes)
    y_vels = np.random.rand(num_apexes) * (max_y_vel)
    for i in range(num_apexes):
        height = np.random.rand() * (1.5 - 0.9) + 0.9 
        initial_apex = hopper2d.FlightState2D()
        initial_apex.xdot = x_vels[i]
        initial_apex.ydot = y_vels[i]
        initial_apex.y = 2
        initial_apex.z = height
        initial_apex.zf = height - hopper.Constants().L
        initial_apexes.append(initial_apex.getArray()) 
    
    for num_ditches in ditch_profile:
        array, func = hopper2d.generateRandomTerrain2D(max_x, max_y, disc, num_ditches)
        arrays.append(array)
    # need the extra params in the 2D case.
    test_matrix = TestMatrix(arrays, initial_apexes, ditch_profile, disc = disc, max_x = max_x, max_y = max_y)
    return test_matrix


def generateStepTestMatrix2D(ditch_profile, num_apexes, max_x_vel, max_y_vel, array_params):
    initial_apexes = []
    arrays = []
    max_x = array_params[0]
    max_y = array_params[1]
    disc = array_params[2]
    x_vels = np.linspace(0, max_x_vel, num_apexes)
    y_vels = np.random.rand(num_apexes) * (max_y_vel)
    for i in range(num_apexes):
        height = np.random.rand() * (1.5 - 0.9) + 0.9 
        initial_apex = hopper2d.FlightState2D()
        initial_apex.xdot = x_vels[i]
        initial_apex.ydot = y_vels[i]
        initial_apex.y = 2
        initial_apex.z = height
        initial_apex.zf = height - hopper.Constants().L
        initial_apexes.append(initial_apex.getArray()) 
    
    for num_ditches in ditch_profile:
        array, func = hopper2d.generateRandomStepTerrain2D(max_x, max_y, disc, num_ditches)
        arrays.append(array)

    # need the extra params in the 2D case.
    test_matrix = TestMatrix(arrays, initial_apexes, ditch_profile, disc = disc, max_x = max_x, max_y = max_y)
    return test_matrix

