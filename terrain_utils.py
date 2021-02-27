import numpy as np
import matplotlib.pyplot as plt

## Terrain Generation Functions
def truncate(x, n):
    factor = 10**n
    tmp = float(int(x * factor))
    return tmp/factor
    

# generate piecewise-flat terrain 
# generates both the discretized vector and the terrain function (for generating the labels)
def generateTerrain(until = 3, feature_height = 0.5,
                    make_ditch = False, make_step = False, 
                    feature_width = 0.5, feature_loc = None, return_loc = False):
    # generates only a ditch
    if make_ditch:
        feature_height = -feature_height
    elif make_step:
        feature_height = feature_height
    else:
        feature_height = 0
    
    if feature_loc is None:
      feature_loc = truncate(np.random.rand() * until/2, 1)
      
    locs = np.arange(0, until, 0.1)
    array = []  # this is for passing into the network
    height_dict = {}  # This is for generating the function
    for pos in locs:
        if pos >= feature_loc and pos <= feature_loc + feature_width:
            array.append(feature_height)
            height_dict[int(pos * 10)] = feature_height
        else:
            array.append(0)
            height_dict[int(pos * 10)] = 0
            
    def terrain_func(x):
        if x > until:
            return 0
        if x < -0.5:
            return 2
        if x < 0:
            return 0
        return height_dict[int(truncate(x, 1)*10)]
    if not return_loc:
      return array, terrain_func
    else:
      return array, terrain_func, feature_loc


# ditch_locs = {loc: (height, width)}
# step_locs = {loc: (height, width)}
def generateTerrain2(ditch_locs = {}, step_locs = {}, until = 3, height_range = (-1, 1), step = 0.5):
          
    locs = np.arange(0, until, 0.1)
    array = np.zeros(locs.shape)  # this is for passing into the network
    height_dict = {}  # This is for generating the function
    # if ditch_locs == {} and step_locs == {}:
    #   # generate random terrains (TODO: some structured way of doing this?)
    #   # the terrain changes every 0.5 m 
    #   spacing = int(10 * step)
    #   poss = np.arange(0, until * 10, spacing)
    #   prev = 0
    #   for p in poss:
    #     val = 0.75 * (np.random.rand() * (height_range[1]- height_range[0]) + height_range[0]) + 0.25 * prev
    #     array[p:p+spacing] = val
    #     prev = val
    # else:
    for loc in ditch_locs.keys():
      height, width = ditch_locs[loc]
      array[int(loc/0.1):int((loc + width)/0.1)] = height
      
    for loc in step_locs.keys():
      height, width = step_locs[loc]
      array[int(loc/0.1):int((loc + width)/0.1)] = height

    for pos in locs:
      height_dict[int(pos * 10)] = array[int(pos/0.1)]
            
    def terrain_func(x):
        if x > until:
            return 0
        if x < -0.5:
            return 2
        if x < 0:
            return 0
        return height_dict[int(truncate(x, 1)*10)]

    return array, terrain_func


def plot_terrain(ax, x_minus, x_plus, terrain_func):
    segments = np.arange(start = x_minus, stop = x_plus + 0.1, step = 0.1)
    segments[segments.size - 1] = min(segments[segments.size - 1], x_plus)
    for i in range(0, segments.size - 1):
        ax.plot([segments[i], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i])], color = "black")
        ax.plot([segments[i+1], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i+1])], color = "black")
    return


# Generate multiple island scenario + stairs scenarios for training
def generateIslandsParams(num_islands = -1, until = 8, disc = 0.1):
  # generate a huge ditch in the middle, then randomly generate islands.
  ditch_loc = np.random.rand() * (until//2 - 0.5) + 0.5
  ditch_width = np.random.rand() * 4 + 2
  ditch_dict = {ditch_loc:(-1, ditch_width)}
  if num_islands == -1:
    num_islands = int(np.random.rand() * 5) # will give us between 0 and 4 islands

  islands_dict = {}
  island_widths = [0] * num_islands
  island_height = 0

  prev_island_end = ditch_loc
  for i in range(num_islands):
    loc = np.random.rand() * 1.5 + prev_island_end
    width = np.random.rand() + 1
    prev_island_end = loc + width
    islands_dict[loc] = (0, width)
  return ditch_dict, islands_dict


def generateIslandsByDitches(num_ditches, until = 8, disc = 0.1):
  ditch_dict = {}
  prev_ditch_end = 0.5
  solo_arrays = []
  solo_funcs = []
  for i in range(num_ditches):
    loc = np.random.rand() + prev_ditch_end + 0.5
    width = np.random.rand() + 0.3
    prev_ditch_end = loc + width
    ditch_dict[loc] = (-1 * (np.random.rand() + 0.5), width)
    # arr, func = generateTerrain2(ditch_locs = {loc:(-1, width)}, step_locs = {}, until = until)

  comp_arr, comp_func = generateTerrain2(ditch_locs = ditch_dict, step_locs = {}, until = until)
  return comp_arr, comp_func


def generateIslandTerrains(num_terrains, until = 8, disc = 0.1):
  # want to get a good spread on the number of islands, ranging from 1-4
  terrain_arrays = []
  terrain_functions = []
  num_ditches = 1
  for i in range(num_terrains):    
    if i > num_ditches * (num_terrains/4):
      num_ditches += 1
    # ditch_dict, islands_dict = generateIslandsParams(num_islands)
    # arr, func = generateTerrain2(ditch_dict, islands_dict, until = until)
    arr, func = generateIslandsByDitches(num_ditches, until = until, disc = disc)
    terrain_arrays.append(arr)
    terrain_functions.append(func)
  return terrain_arrays, terrain_functions


def generateStairsParams(num_steps = -1, until = 8, dict = 0.1):
  steps_dict = {}
  prev_step_end = np.random.rand() * 2 + 0.5
  prev_step_height = 0
  if num_steps == -1:
    num_steps = int(np.random.rand() * 5)

  for i in range(num_steps):
    loc = np.random.rand() * 0.3 + prev_step_end  # allow small gaps..?
    width = np.random.rand() * 1.5 + 0.5
    height = np.random.rand() * 0.4
    prev_step_end = loc + width
    steps_dict[loc] = (height, width)

  return steps_dict


def generateStairTerrains(num_terrains, until = 8, disc = 0.1):
  terrain_arrays = []
  terrain_functions = []
  num_steps = 1
  for i in range(num_terrains):    
    if i > num_steps * (num_terrains/4):
      num_steps += 1
    steps_dict = generateStairsParams(num_steps)
    arr, func = generateTerrain2(ditch_locs = {}, step_locs = steps_dict, until = until)
    terrain_arrays.append(arr)
    terrain_functions.append(func)
  return terrain_arrays, terrain_functions

def generateSingleFeatureTerrains(num_terrains, until = 8):
    max_dd = 1.0
    min_dd = 0.4
    
    terrain_arrays = []
    terrain_functions = []
    terrain_locs = []
    terrain_widths = []
    num_ditch = max(1, num_terrains//2)
    num_step = num_terrains//2
    
    num_individual_terrains = num_ditch + num_step
    heights = np.linspace(min_dd, max_dd, int(np.sqrt(num_ditch)))
    locs = np.linspace(0.5, until-2, int(np.sqrt(num_ditch)))
    widths = np.random.rand(num_ditch) + 0.7
    c = 0
    for height in heights:
        for loc in locs:
            arr, func, loc = generateTerrain(until = until, 
                                        make_ditch = True,
                                        feature_height = height,
                                        feature_width = widths[c], 
                                        feature_loc = loc,
                                        return_loc = True)
            terrain_arrays.append(arr)
            terrain_functions.append(func)
            terrain_locs.append(loc)
            terrain_widths.append(widths[c])
            c += 1

    max_height = 1.0
    min_height = 0.05

    heights = np.linspace(min_height, max_height, int(np.sqrt(num_step)))
    locs = np.linspace(0.5, until/2, int(np.sqrt(num_step)))
    widths = np.random.rand(num_step) + 1
    c = 0
    for height in heights:
        for loc in locs:
            arr, func, loc = generateTerrain(until = until,
                                        make_step = True, 
                                        feature_height = height,
                                        feature_width = widths[c],
                                        feature_loc = loc,
                                        return_loc = True)
            terrain_arrays.append(arr)
            terrain_functions.append(func)
            terrain_locs.append(loc)
            terrain_widths.append(widths[c])
            c += 1

    return terrain_arrays, terrain_functions, terrain_locs, terrain_widths


'''
    Generate terrain function from array
'''
def getTerrainFunc(t_array, disc = 0.1, min_lim = 0):
    def terrain_func(x):
        x_disc = int(x/disc + min_lim)
        if x_disc < 0:
            return 0
        if x_disc >= len(t_array):
            return 0
        return t_array[x_disc]
    return terrain_func
