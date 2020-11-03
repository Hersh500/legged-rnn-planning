import numpy as np

import hopper
import astar_tree_search
from astar_tree_search import aStarHelper
import terrain_utils


# Idea: randomly sampling for long sequences takes a long ass time. 
# Instead compose a bunch of shorter sequences together to form a longer sequence.
# Also could just use A*, since this could get stuck in local minima. 
def generateNFeasibleSteps(n,
                           initial_apex,
                           terrain_func,
                           terrain_normal_func,
                           friction,
                           max_tries = 100):
  steps_list = []
  angles_list = []
  tries = 0
  x0 = initial_apex

  min_angle = np.pi/2 - np.arctan(friction/1.5)
  max_angle = np.pi/2 + np.arctan(friction/1.5)

  while len(steps_list) < n and tries < max_tries:
    angles = np.random.rand(2) * (max_angle - min_angle) + min_angle
    steps, apexes = hopper.simulateNSteps(2, x0, angles, terrain_func, return_apexes = True, 
                                   terrain_normal_func = terrain_normal_func, friction = friction)
    if steps is not None:
      if len(steps_list) == 0:
        steps_list += steps
        angles_list.append(angles[0])
      else:
        steps_list.append(steps[1])
        angles_list.append(angles[0])
      tries = 0
      x0 = apexes[1]
    else:
      tries += 1
  if len(steps_list) == n:
    return steps_list, angles_list
  else:
    return None, None


def generateRandomSequences(num_terrains, 
                            num_apexes,
                            num_astar_sequences,
                            max_backup_steps,
                            friction,
                            cost_fn = astar_tree_search.stateCost,
                            full_tree = False,
                            progress = 0.5,
                            only_sf = False,
                            seed = 42):
  
  np.random.seed(seed)
  # While we haven't generated $num_sequences sequences, do a random rollout
  # of leg angles and try it.
  initial_states = []  # terrains + apexes + the "inevitable step"?
  sequences = [] # list of lists of feasible steps
  final_apexes = []
  until = 8
  min_step = -3   # The farthest back we will consider.
  min_x = 0
  max_x = 2
  min_y = 1.5
  max_y = 0.8
  min_x_dot = -1
  max_x_dot = 3


  pos_encode = False
  min_angle = np.pi/2 - np.arctan(friction)
  max_angle = np.pi/2 + np.arctan(friction)
  if only_sf:
    sf_arrs, sf_funcs, terrain_locs, terrain_widths = terrain_utils.generateSingleFeatureTerrains(num_terrains, until = until)
    terrain_arrays = sf_arrs
    terrain_functions = sf_funcs
  else:
    # sf_arrs, sf_funcs, terrain_locs, terrain_widths = generateSingleFeatureTerrains(num_terrains//2, until = until)
    island_arrs, island_fns = terrain_utils.generateIslandTerrains(num_terrains//2, until = until)
    stair_arrs, stair_fns = terrain_utils.generateStairTerrains(num_terrains//2, until = until)
    terrain_arrays = island_arrs + stair_arrs
    terrain_functions = island_fns + stair_fns


  random_initial_apexes = np.zeros((max(50, num_apexes), 6))
  for a in range(random_initial_apexes.shape[0]):
    random_initial_apexes[a] = np.array([0, np.random.rand() * (max_y - min_y) + min_y,
                                np.random.rand() * (max_x_dot - min_x_dot) + min_x_dot,
                                0, 0, np.pi/2])

  max_tries = 4  # try $max_tries sequences before giving up.
  pos_array = np.arange(0, 8.0, 0.1)
  num_augmented = 0
  for i in range(len(terrain_arrays)):
    if not pos_encode:
      this_arr = list(terrain_arrays[i])
    else:
      this_arr = list(np.array(terrain_arrays[i]) + pos_array)
    random_indices = (np.random.rand(num_apexes) * random_initial_apexes.shape[0]).astype(int)
    apexes = random_initial_apexes[random_indices]
    for initial_apex in apexes:
      success_count = 0
      num_tries = 0
      while success_count < num_astar_sequences and num_tries < max_tries:
        step_sequences, angle_sequences = aStarHelper(initial_apex,
                                                     [10, 0],
                                                     num_astar_sequences,
                                                     terrain_functions[i],
                                                     lambda x: np.pi/2,
                                                     friction,
                                                     num_angle_samples = 18,
                                                     timeout = 500,
                                                     max_speed = 3,
                                                     get_full_tree = full_tree,
                                                     neutral_angle = False,
                                                     cost_fn = cost_fn)
        for l, ss in enumerate(step_sequences):
          cond = len(ss) > max_backup_steps and ss[-1] >= ss[0]
          for s in range(1, len(ss)):
            cond = cond and (ss[s] > ss[s-1] - progress)
          if cond:
            success_count += 1
            initial_condition = this_arr + list(initial_apex[:3])
            sequences.append(ss)
            initial_states.append(initial_condition)
        if success_count < num_astar_sequences:
          print("A* backup: trying with more samples..")
          step_sequences, angle_sequences = aStarHelper(initial_apex,
                                              [10, 0],
                                              num_astar_sequences,
                                              terrain_functions[i],
                                              lambda x: np.pi/2,
                                              friction,
                                              num_angle_samples = 30,
                                              timeout = 500,
                                              max_speed = 2.5,
                                              get_full_tree = full_tree,
                                              neutral_angle = False,
                                              cost_fn = cost_fn)
          for l, ss in enumerate(step_sequences):
            cond = len(ss) > max_backup_steps and ss[-1] >= ss[0]
            for s in range(1, len(ss)):
              cond = cond and (ss[s] > ss[s-1] - progress)
            if cond:
              success_count += 1
              initial_condition = this_arr + list(initial_apex[:3])
              sequences.append(ss)
              initial_states.append(initial_condition)
          num_tries += 1
        # if A* doesn't find enough seqs just try to generate some feasible plan of steps, even if it doesn't reach goal.
        '''
        if success_count < num_astar_sequences:
          print("A* backup: Trying random sequences...")
          for _ in range(5):
            num_steps = int(np.random.rand() * (max_backup_steps-3) + 3)
            steps, angles = generateNFeasibleSteps(num_steps, initial_apex, terrain_functions[i], lambda x: np.pi/2, friction)
            cond = steps is not None and steps[-1] > steps[0]
            if steps is None:
              cond = False
            else:
              for s, step in enumerate(steps[1:]):
                cond = cond and (steps[s] > steps[s-1] - progress)
            if cond:
              # print("found random sequence!")
              success_count += 1
              initial_condition = this_arr + list(initial_apex[:3])
              sequences.append(steps)
              initial_states.append(initial_condition)
          num_tries += 1
        '''
        print("added", success_count, "sequences")

    print("finished terrain", i)
  return initial_states, sequences


def generateHeuristicDataset(num_terrains, num_apexes, friction):
    # Generate training sequences according to the heuristic planner, in receding horizon?
    # Requires passing in a controller.
    # If the planner fails, then use A*?
    return

# Generates sequences from A* that is planned in the footstep space
def generateRandomSequences2():
    return
