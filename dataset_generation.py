import numpy as np

import hopper
import hopper2d
import astar_tree_search
from astar_tree_search import aStarHelper
import terrain_utils


def costFn(x_flight, neighbors, goal, p):
    x_pos = x_flight[0]
    y_pos = x_flight[1]

    x_vel = x_flight[3]
    y_vel = x_flight[4]

    # Don't use spread for now.
    spread = 0
    for n in range(len(neighbors)):
        x_dist = (neighbors[n][0] - x_pos)**2
        y_dist = (neighbors[n][1] - y_pos)**2
        spread += np.sqrt(x_dist + y_dist)/len(neighbors)

    return (np.sqrt(1 * np.abs(x_pos - goal[0])**2 + 0.5 * np.abs(y_pos - goal[1])**2) +
            0.5 * np.abs(x_vel - goal[2]) + 0.5 * np.abs(y_vel - goal[3])) + 0 * spread


def processSeq(seq, buf, mov_amount, terrain_func):
    mod_seq = []
    for step in seq:
        if terrain_func(step - buf) < 0:
            mod_seq.append(step + mov_amount)
        elif terrain_func(step + buf) < 0:
            mod_seq.append(step - mov_amount)
        else:
            mod_seq.append(step)
    return mod_seq


def generateNFeasibleSteps(robot, n,
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
    steps, apexes = hopper.simulateNSteps(robot, 2, x0, angles, terrain_func, return_apexes = True, 
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

def main():
    robot = hopper2d.Hopper2D(hopper.Constants())
    initial_states, initial_terrains, sequences = generateRandomSequences2D(robot,
                                                                      num_terrains = 5, 
                                                                      num_apexes = 1,
                                                                      num_astar_sequences = 1,
                                                                      min_steps = 4,
                                                                      friction = 0.6,
                                                                      cost_fn = astar_tree_search.cost_fn2d,
                                                                      full_tree = False,
                                                                      progress = 0.1,
                                                                      seed = 42)
    print(sequences)

    
def generateTerrainDataset2D(num_terrains, until_x, until_y, disc):
    min_num_ditches = 5
    max_num_ditches = 9

    min_num_steps = 5
    max_num_steps = 9
    all_terrains = np.zeros((num_terrains, int(until_y / disc), int(until_x / disc)))
    for i in range(num_terrains//2):
        num_ditches = np.random.randint(low = min_num_ditches, high = max_num_ditches)
        terrain_array, _ = hopper2d.generateRandomTerrain2D(until_x,
                                                            until_y,
                                                            disc,
                                                            num_ditches)
        all_terrains[i] = terrain_array

    for j in range(num_terrains//2):
        num_steps = np.random.randint(low = min_num_steps, high = max_num_steps)
        terrain_array, _ = hopper2d.generateRandomStepTerrain2D(until_x,
                                                            until_y,
                                                            disc,
                                                            num_steps)
        all_terrains[i+j] = terrain_array

    return all_terrains


def generateTerrainStepDataset2D(num_terrains, until_x, until_y, disc):
    min_num_ditches = 5
    max_num_ditches = 12

    min_num_steps = 5
    max_num_steps = 12
    all_terrains = np.zeros((num_terrains, 2, int(until_y / disc), int(until_x / disc)))
    for i in range(num_terrains//2):
        num_ditches = np.random.randint(low = min_num_ditches, high = max_num_ditches)
        terrain_array, _ = hopper2d.generateRandomTerrain2D(until_x,
                                                            until_y,
                                                            disc,
                                                            num_ditches)
        all_terrains[i][0] = terrain_array
        x, y = np.random.randint(0, until_x//disc), np.random.randint(0, until_y//disc)
        all_terrains[i][1][y][x] = 1

    for j in range(num_terrains//2):
        num_steps = np.random.randint(low = min_num_steps, high = max_num_steps)
        terrain_array, _ = hopper2d.generateRandomStepTerrain2D(until_x,
                                                            until_y,
                                                            disc,
                                                            num_steps)
        all_terrains[i+j][0] = terrain_array
        x, y = np.random.randint(0, until_x//disc), np.random.randint(0, until_y//disc)
        all_terrains[i+j][1][y][x] = 1

    return all_terrains


# terrain_params = [friction, max_x, max_y, disc]
def generateCurricularDataset2D(robot,
                                max_num_feats,
                                terrain_schedule,
                                apex_schedule,
                                terrain_params,
                                cost_fn,
                                seed = 42):
    
    if len(terrain_schedule) != max_num_feats+1 or len(apex_schedule) != max_num_feats+1:
        print("error! need terrain and apex schedule to have max_num_feats elements")
        return [], [], []

    np.random.seed(seed)
    # these are 3d arrays: max_num_feats x sequences generated for each num feat
    sequences = []
    terrains = []
    initial_states = []
    
    friction = terrain_params[0]
    until_x = terrain_params[1]
    until_y = terrain_params[2]
    disc = terrain_params[3]

    max_x_dot = 3
    min_x_dot = -1
    max_y_dot = 2
    min_y_dot = -1
    max_z = 1.5
    min_z = 0.8

    # generate the initial_apexes 
    random_initial_apexes = np.zeros((max(50, max(apex_schedule)), 13))
    for a in range(random_initial_apexes.shape[0]):
        initial_state = hopper2d.FlightState2D()
        initial_state.xdot = np.random.rand() * (max_x_dot - min_x_dot) + min_x_dot
        initial_state.ydot = np.random.rand() * (max_y_dot - min_y_dot) + min_y_dot
        initial_state.z = np.random.rand() * (max_z - min_z) + min_z
        initial_state.y = until_y//2
        initial_state.zf = initial_state.z - robot.constants.L

        state_array = initial_state.getArray()
        random_initial_apexes[a] = np.array(state_array)

    for i in range(max_num_feats+1):
        # generate the terrains for this # features
        terrain_arrays = []
        terrain_funcs = []
        for _ in range(terrain_schedule[i]//2):
          terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(until_x, until_y, disc, i)
          terrain_arrays.append(terrain_array)
          terrain_funcs.append(terrain_func)
        for _ in range(terrain_schedule[i]//2):
          terrain_array, terrain_func = hopper2d.generateRandomStepTerrain2D(until_x, until_y, disc, i)
          terrain_arrays.append(terrain_array)
          terrain_funcs.append(terrain_func)

        feat_seqs = []
        feat_terrains = []
        feat_is = []
        for j in range(terrain_schedule[i]):
            terrain_func = terrain_funcs[j]
            terrain_array = terrain_arrays[j]
            for k in range(apex_schedule[i]):
                index = np.random.choice(random_initial_apexes.shape[0])
                initial_apex = random_initial_apexes[index]
                s, t, i_s = generateSequencesForScenario(robot, terrain_func,
                                                         terrain_array, initial_apex,
                                                         friction, [until_x, until_y//2, 0, 0], cost_fn,
                                                         1, False,
                                                         0.0)
                feat_seqs += s
                feat_terrains += t
                feat_is += i_s

        sequences.append(feat_seqs) 
        terrains.append(feat_terrains)
        initial_states.append(feat_is)
        print("finished",i,"feature terrains")
    return sequences, terrains, initial_states


# breaking this out for use in generating the curricular dataset
def generateSequencesForScenario(robot,
                                terrain_func,
                                terrain_array,
                                initial_apex,
                                friction,
                                goal,
                                cost_fn,
                                num_astar_sequences,
                                full_tree,
                                progress,
                                min_steps = 5):

  sequences = []
  initial_terrains = []
  initial_states = [] 
  success_count = 0
  num_tries = 0
  max_tries = 2
  while success_count < num_astar_sequences and num_tries < max_tries:
    step_sequences, angles, count = astar_tree_search.angleAstar2Dof(robot,
                                                                  initial_apex,
                                                                  goal,
                                                                  15,
                                                                  num_astar_sequences,
                                                                  cost_fn,
                                                                  terrain_func,
                                                                  lambda x,y:np.pi/2,
                                                                  friction,
                                                                  get_full_tree = full_tree)
    for l, ss in enumerate(step_sequences):
      cond = len(ss) > min_steps and ss[-1][0] >= ss[0][0] and ss[-1][1] >= ss[0][1]
      for s in range(1, len(ss)):
        cond = cond and (ss[s][0] > ss[s-1][0] - progress) and (ss[s][1] > ss[s-1][1] - progress)
      if cond:
        success_count += 1
        # xdot, ydot, z height
        initial_condition = [initial_apex[3], initial_apex[4], initial_apex[2]]
        initial_terrains.append(terrain_array)
        sequences.append(ss)
        initial_states.append(initial_condition)
    if success_count < num_astar_sequences:
      step_sequences, angles, count = astar_tree_search.angleAstar2Dof(robot,
                                             initial_apex,
                                             goal,
                                             15,
                                             num_astar_sequences,
                                             cost_fn,
                                             terrain_func,
                                             lambda x,y:np.pi/2,
                                             friction,
                                             get_full_tree = full_tree)
      for l, ss in enumerate(step_sequences):
        cond = len(ss) > min_steps and ss[-1][0] >= ss[0][0]
        for s in range(1, len(ss)):
          cond = cond and (ss[s][0] > ss[s-1][0] - progress)
        if cond:
          success_count += 1
          # xdot, ydot, z height
          initial_condition = [initial_apex[3], initial_apex[4], initial_apex[2]]
          initial_terrains.append(terrain_array)
          sequences.append(ss)
          initial_states.append(initial_condition)
      num_tries += 1
  return sequences, initial_terrains, initial_states


def generateRandomSequences2D(robot,
                            num_terrains, 
                            num_apexes,
                            num_astar_sequences,
                            min_steps,
                            friction,
                            cost_fn = astar_tree_search.stateCost,
                            full_tree = False,
                            only_ditch = False,
                            progress = 0.5,
                            until_x = 5,
                            until_y = 5,
                            disc = 0.1,
                            seed = 42,
                            path = './tmp/'):
  
  # print("using seed,", seed)
  np.random.seed(seed)
  initial_states = []
  initial_terrains = []
  sequences = [] # list of lists of feasible steps
  final_apexes = []

  max_z = 1.5
  min_z = 0.8

  min_x_dot = -0.5
  max_x_dot = 2
  min_y_dot = -0.5
  max_y_dot = 2

  terrain_arrays = []
  terrain_functions = []
  # rework this: the number of features is too small, resulting in uninteresting trajs.
  if only_ditch:
    max_num_ditches = min(6, num_terrains+1)
  else:
    max_num_ditches = min(6, num_terrains//2+1)
    max_num_steps = min(6, num_terrains//2+1)


  offset = 5
  # Generate the training terrains
  for num_ditches in range(offset, offset + max_num_ditches):
    if not only_ditch:
      for _ in range(num_terrains//(2 * (max_num_ditches - 1))):
        terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(until_x, until_y, disc, num_ditches)
        terrain_arrays.append(terrain_array)
        terrain_functions.append(terrain_func)
    else:
      for _ in range(num_terrains//((max_num_ditches - 1))):
        terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(until_x, until_y, disc, num_ditches)
        terrain_arrays.append(terrain_array)
        terrain_functions.append(terrain_func)

  if not only_ditch:
    for num_steps in range(offset, offset + max_num_steps):
      for _ in range(num_terrains//(2 * (max_num_steps - 1))):
        terrain_array, terrain_func = hopper2d.generateRandomStepTerrain2D(until_x, until_y, disc, num_steps)
        terrain_arrays.append(terrain_array)
        terrain_functions.append(terrain_func)

  # Generate the initial apexes
  random_initial_apexes = np.zeros((max(50, num_apexes), 13))
  for a in range(random_initial_apexes.shape[0]):
    initial_state = hopper2d.FlightState2D()
    initial_state.xdot = np.random.rand() * (max_x_dot - min_x_dot) + min_x_dot
    initial_state.ydot = np.random.rand() * (max_y_dot - min_y_dot) + min_y_dot
    initial_state.z = np.random.rand() * (max_z - min_z) + min_z
    initial_state.y = until_y//2  # RNN always starts at the left end of the array, in the middle y
    initial_state.zf = initial_state.z - robot.constants.L

    state_array = initial_state.getArray()
    random_initial_apexes[a] = np.array(state_array)

  # Actually generate the training sequenecs
  max_tries = 2
  for i in range(len(terrain_arrays)):
    this_arr = terrain_arrays[i].tolist()
    random_indices = (np.random.rand(num_apexes) * random_initial_apexes.shape[0]).astype(int)
    apexes = random_initial_apexes[random_indices]
    for initial_apex in apexes:
      s, t, i_s = generateSequencesForScenario(robot,
                                            terrain_func,
                                            terrain_array,
                                            initial_apex,
                                            friction,
                                            [until_x, until_y//2,0,0],
                                            cost_fn,
                                            num_astar_sequences,
                                            full_tree,
                                            progress)
      sequences += s
      initial_terrains += t
      initial_states += i_s
    print("finished terrain", i)

  np.save(path + "terrains_" + str(seed), initial_terrains)
  np.save(path + "sequences_" + str(seed), sequences)
  np.save(path + "states_" + str(seed), initial_states)
  print("saved files")
  return initial_states, initial_terrains, sequences


def generateRandomSequences(robot,
                            num_terrains, 
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
  initial_states = [] # apexes + terrains
  sequences = [] # list of lists of feasible steps
  final_apexes = []
  until = 8
  min_step = -3   # The farthest back we will consider.
  min_x = 0
  max_x = 2
  min_y = 0.8 
  max_y = 1.5
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
    island_arrs, island_fns = terrain_utils.generateIslandTerrains(num_terrains//2, until = until)
    stair_arrs, stair_fns = terrain_utils.generateStairTerrains(num_terrains//2, until = until)
    terrain_arrays = island_arrs + stair_arrs
    terrain_functions = island_fns + stair_fns


  # ensure even coverage:
  num_all_apexes = 50
  heights = np.linspace(min_y, max_y, int(np.sqrt(num_all_apexes)))
  vels = np.linspace(min_x_dot, max_x_dot, int(np.sqrt(num_all_apexes)))
  random_initial_apexes = np.zeros((max(int(np.sqrt(num_all_apexes))**2, num_apexes), 6))
  h, v = np.meshgrid(heights, vels)
  h, v = h.flatten(), v.flatten()
  for a in range(len(v)):
    # random_initial_apexes[a] = np.array([0, np.random.rand() * (max_y - min_y) + min_y,
    #                            np.random.rand() * (max_x_dot - min_x_dot) + min_x_dot,
    #                            0, 0, np.pi/2])
    random_initial_apexes[a] = np.array([0, h[a], v[a], 0, 0, np.pi/2])

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
        step_sequences, angle_sequences = aStarHelper(robot,
                                                      initial_apex,
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
            sequences.append(processSeq(ss, 0.2, 0.2, terrain_functions[i]))
            initial_states.append(initial_condition)
        if success_count < num_astar_sequences:
          # print("A* backup: trying with more samples..")
          step_sequences, angle_sequences = aStarHelper(robot,
                                                      initial_apex,
                                                      [10, 0],
                                                      num_astar_sequences,
                                                      terrain_functions[i],
                                                      lambda x: np.pi/2,
                                                      friction,
                                                      num_angle_samples = 30,
                                                      timeout = 500,
                                                      max_speed = 4,
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
              sequences.append(processSeq(ss, 0.2, 0.2, terrain_functions[i]))
              initial_states.append(initial_condition)
          num_tries += 1
    print("finished terrain", i)
  return initial_states, sequences


# Generates sequences with the step-space planner (currently unused) 
def generateRandomSequences2(robot,
                            num_terrains,
                            num_apexes,
                            num_astar_sequences,
                            friction,
                            step_controller,
                            spacing = 0.2,
                            horizon = 3,
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
  max_x = 3 
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
        step_sequences, angle_sequences = footSpaceAStar(initial_apex, [9, 0], num_astar_sequences,
                                                         step_controller, terrain_functions[i],
                                                         friction, horizon, spacing,
                                                         cost_fn, get_full_tree = full_tree)
        for l, ss in enumerate(step_sequences):
          cond = len(ss) > 3 and ss[-1] >= ss[0]
          for s in range(1, len(ss)):
            cond = cond and (ss[s] > ss[s-1] - progress)
          if cond:
            success_count += 1
            initial_condition = this_arr + list(initial_apex[:3])
            sequences.append(ss)
            initial_states.append(initial_condition)
        if success_count < num_astar_sequences:
          print("A* backup: trying with more samples..")
          step_sequences, angle_sequences = footSpaceAStar(initial_apex, [9, 0], num_astar_sequences,
                                                           step_controller, terrain_functions[i],
                                                           friction, 4, 0.1,
                                                           cost_fn, get_full_tree = full_tree)
          for l, ss in enumerate(step_sequences):
            cond = len(ss) > 3 and ss[-1] >= ss[0]
            for s in range(1, len(ss)):
              cond = cond and (ss[s] > ss[s-1] - progress)
            if cond:
              success_count += 1
              initial_condition = this_arr + list(initial_apex[:3])
              sequences.append(ss)
              initial_states.append(initial_condition)
          num_tries += 1
        print("added", success_count, "sequences")

    print("finished terrain", i)
  return initial_states, sequences


if __name__ == "__main__":
    main()
