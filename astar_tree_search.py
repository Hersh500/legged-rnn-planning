import numpy as np
import hopper
import hopper2d
import queue
from scipy.spatial import KDTree
import heightmap

# All of the astar utilities
class GraphNode:
  # loc is the x position of the stance state
  # apex is the apex for which this stance state is inevitable
  # angle is the leg angle input applied during that apex to get to this stance.
  # step is which step of the planner this node gets hit.
  def __init__(self, x_loc, angle, apex, step, cost, parent, children):
    self.x_loc = x_loc
    self.apex = apex
    self.prev_angle = angle
    self.step = step
    self.value = cost
    self.parent = parent
    self.children = children

  def updateValue(self, new_cost):
    self.value = new_cost

  # states are considered equal if velocity and pos are within 5 cm of each other
  def areEqual(self, other_node, tol = 5e-2):
    return (np.abs(self.x_loc - other_node.x_loc) < tol and np.abs(self.apex[2] - other_node.apex[2]) < tol)

  # this is different from self.value because self.value is intended to be the cost of this node
  # plus the cost of its parent
  def nodeCost(self, goal):
    return stateCost(self.apex, goal, self.step)


# penalize high velocities and distance to the goal.
def stateCost(x_flight, neighbors, goal, p):
    x_pos = x_flight[0]
    x_vel = x_flight[2]
    return np.abs(x_pos - goal[0]) + np.abs(x_vel)


def stateCost_neighbors(x_flight, neighbors, goal, p):
    x_pos = x_flight[0]
    x_vel = x_flight[2]

    # gets the mean spread of the two neighbors
    spread = 0
    for i in range(len(neighbors)):
        spread += np.abs(neighbors[i][0] - x_flight[0])
    spread = spread / 2

    return np.abs(x_pos - goal[0]) +  3 * np.abs(x_vel) + spread


def sampleLegAngles(robot, x_apex, goal, step, terrain_func, terrain_normal_func, friction,
                    num_samples, cost_fn, cov_factor = 4, neutral_angle_normal = True):
  # this is the limit of how far forward the leg can be placed in front of body
  # based on the friction cone
  # roughly pi/4
  min_limit = np.pi/2 - np.arctan(friction)

  # this is the limit of how far backward the leg can be placed behind body
  max_limit = np.pi/2 + np.arctan(friction)

  angles = []
  costs = []
  apexes = []
  locs = []
  last_flights = []
  total_count = 0

  # Hardcoded stance time, not necessarily accurate
  mean = np.arccos(x_apex[2] * 0.18 / 2)
  cov = (max_limit - min_limit)/cov_factor

  if neutral_angle_normal:
    # samples from a normal distr centered around the mean
    pos = np.random.randn(num_samples) * cov + mean
  else:
    # deterministic gridding
    pos = np.linspace(min_limit, max_limit, num_samples)
    pos = sorted(pos)

  for i in range(len(pos)):
    apex1, apex2, last_flight, count = hopper.getNextState2Count(robot, 
                                                    x_apex,
                                                    pos[i],
                                                    terrain_func,
                                                    terrain_normal_func,
                                                    friction,
                                                    at_apex = True)
    total_count += count
    if apex1 is not None:
      last_flights.append(last_flight)
      # cost = stateCost_neighbors(last_flight, [], goal, step + 1)
      angles.append(pos[i])
      # costs.append(cost)
      apexes.append(apex2)
      locs.append(last_flight[0])

  for i in range(len(angles)):
    if i > 0 and i < len(angles) - 1:
      neighbors = [last_flights[i-1], last_flights[i+1]]
    elif len(angles) == 1:
      neighbors = []
    elif i == 0:
      neighbors = [last_flights[i+1]]
    else:
      neighbors = [last_flights[i-1]]
    # cost = stateCost_neighbors(last_flights[i], neighbors, goal, step + 1)
    cost = cost_fn(last_flights[i], neighbors, goal, step + 1) + np.random.rand() * 1e-4  # add small random quantity to prevent errors due to ties
    costs.append(cost)

  return angles, costs, apexes, locs, total_count


def inOrderHelper(root_node, goal):
  all_paths = []
  all_angles = []
  if len(root_node.children) == 0:
    if root_node.x_loc >= goal:
      all_paths = [[root_node.x_loc]]
      all_angles = [[root_node.prev_angle]]
    else:
      all_paths = [[]]
      all_angles = [[]]
  for child in root_node.children:
    paths, angles = inOrderHelper(child, goal)
    for i, p in enumerate(paths):
      all_paths.append([root_node.x_loc] + paths[i])
      all_angles.append([root_node.prev_angle] + angles[i])
  return all_paths, all_angles


'''
Performs an A* search from the initial node to the goal node.
Returns a list of sequences that have reached the goal state.
'''
def aStarHelper(robot, x0_apex, goal, num_goal_nodes,
                terrain_func, terrain_normal_func, friction,
                num_angle_samples, timeout = 1000, get_full_tree = False,
                cov_factor = 4, neutral_angle = True, max_speed = 2,
                cost_fn = stateCost, count_odes = False):

  time_till_ground = 2 * (x0_apex[1] - robot.constants.L)/(-robot.constants.g)
  xstep_pred = x0_apex[0] + x0_apex[2] * time_till_ground

  cur_node = GraphNode(x0_apex[0], 0, x0_apex, 0, 0, None, [])
  cur_node.x_loc = xstep_pred
  root_node = cur_node
  deepest_node = cur_node

  node_queue = queue.PriorityQueue(maxsize = 10000)  # size limit to avoid things getting out of control
  min_heap = []
  cur_apex = x0_apex
  step = 0
  goal_nodes = []
  iters = 0
  total_odes = 0

  while len(goal_nodes) < num_goal_nodes and iters < timeout:
    if cur_node.x_loc > deepest_node.x_loc:
      deepest_node = cur_node
    if cur_node.x_loc >= goal[0]:
      goal_nodes.append(cur_node)
      if node_queue.qsize() > 0:
          cur_node = node_queue.get(False)[1]
          cur_apex = cur_node.apex
          step = cur_node.step + 1
      else:
          break
    else:
      angles, costs, apexes, locs, count = sampleLegAngles(robot, cur_apex, goal, step, terrain_func, 
                                                    terrain_normal_func, friction, num_angle_samples,
                                                    cov_factor = cov_factor, neutral_angle_normal = neutral_angle,
                                                    cost_fn = cost_fn)
      total_odes += count
      for i in range(0, len(angles)):
        # angles was the angle used from the previous node to get to this node.
        if np.abs(apexes[i][2]) <= max_speed:
          node = GraphNode(locs[i], angles[i], apexes[i], step, costs[i] + cur_node.value, cur_node, [])
          cur_node.children.append(node)
          if node_queue.qsize() < 10000:
            node_queue.put((costs[i], node))

      if node_queue.qsize() <= 0:
        break
      if node_queue.qsize() >= 10000:
        break

      cur_node = node_queue.get(False)[1]
      cur_apex = cur_node.apex
      step = cur_node.step + 1
      iters += 1

  # now that the while loop has terminated, we can reconstruct the trajectory
  if len(goal_nodes) == 0:
    goal_nodes.append(deepest_node)

  sequences = []
  inputs = []
  # traverse the whole tree
  if get_full_tree:
    all_paths, all_angles = inOrderHelper(root_node, goal[0])
    if count_odes:
      return all_paths, all_angles, total_odes
    else:
      return all_paths, all_angles
  else:
    for cur_node in goal_nodes:
      traj = [cur_node.x_loc]
      angles = [cur_node.prev_angle]
      while cur_node.parent is not None:
        cur_node = cur_node.parent
        traj.append(cur_node.x_loc)
        angles.append(cur_node.prev_angle)
      traj = list(reversed(traj))
      angles = list(reversed(angles))
      sequences.append(traj)
      inputs.append(angles)
    if count_odes:
      return sequences, inputs, total_odes
    else:
      return sequences, inputs


# Step-space search
def footSpaceAStar(robot, 
                   x0_apex,
                   goal,
                   num_goals,
                   step_controller,
                   terrain_func,
                   friction,
                   horizon,
                   spacing,
                   cost_fn,
                   get_full_tree = True,
                   count_odes = False,
                   timeout = 1000,
                   debug = False):
  # initialize all the shits
  pq = queue.PriorityQueue()
  nodes = []
  goal_nodes = []
  iters = 0
  pos = np.arange(0, 8.0, 0.1)
  terrain_normal_func = lambda x: np.pi/2
  Ts = 0.17
  Tf = 0.60
  step = 0
  total_odes = 0

  cur_node = GraphNode((x0_apex[0], 0), 0, x0_apex, 0, 0, None, [])
  root_node = cur_node
  deepest_node = cur_node
  cur_apex = x0_apex
  time_till_ground = 2 * (x0_apex[1] - robot.constants.L)/(-robot.constants.g)
  xstep_pred = x0_apex[0] + x0_apex[2] * time_till_ground

  cur_node.x_loc = (xstep_pred, xstep_pred)
  hit_goal = False
  while len(goal_nodes) < num_goals and iters < timeout:
    if debug:
      print("----Iter", iters, "----")
      print("cur node = ", cur_node.x_loc[0])
    # print("cur node xvel = ", cur_apex[2])
    # print("cur node height = ", cur_apex[1])
    # print("-------------------------------")
    if cur_node.x_loc[0] > deepest_node.x_loc[0]:
      deepest_node = cur_node

    # used to be 3
    orig_samples = np.arange(cur_node.x_loc[0]-2, cur_node.x_loc[0] + horizon, spacing)
    next_samples = []

    # remove samples if they land in a ditch
    for j in range(len(orig_samples)):
      if terrain_func(orig_samples[j]) >= 0:
        next_samples.append(orig_samples[j])

    next_apexes = []
    last_flights = []
    samples = []
    added = []
    for sample in next_samples:
      input = step_controller.calcAngle(sample-cur_apex[0], cur_apex[2], 0, 0, upper_lim = 3.0, lower_lim = 0.5, y = cur_apex[1])
      apex1, apex2, last_flight, count = hopper.getNextState2Count(robot, cur_apex,
                                                      input,
                                                      terrain_func,
                                                      terrain_normal_func,
                                                      friction,
                                                      at_apex = True)
      foot_input = sample  # save the control input that got us here.
      total_odes += count
      if apex1 is not None:
        next_apexes.append(apex2)
        last_flights.append(last_flight)
        samples.append(foot_input)
        # print("last flight vs sample", last_flight[0], sample, "with calc angle = ", input)
    for i in range(len(next_apexes)):
      # calculate the cost at last flight (right before landing)
      if i == 0:
        neighbors = [last_flights[min(1, len(next_apexes) - 1)]]   
      elif i == len(next_apexes) - 1:
        neighbors = [last_flights[i-1]]
      else:
        neighbors = [last_flights[i-1], last_flights[i+1]]
      cost = cost_fn(last_flights[i], neighbors, goal, step) + np.random.rand() * 1e-4
      if debug:
        print("sample next step loc:", last_flights[i][0])
        print("sample controller input:", samples[i])
        print("sample next step cost:", cost)
        print("-")

      # instead of noting the actual x loc, save the input to the controller
      # was previously last_flights[i][0]
      # print("last flight x:", last_flights[i][0], "; sample x:", samples[i])
      new_node = GraphNode((last_flights[i][0], samples[i]),
                           input, next_apexes[i],
                           step, cost + cur_node.value, 
                           cur_node, [])
      added.append(last_flights[i][0])
      if last_flights[i][0] > goal[0]:
        hit_goal = True
      # tup = (new_node.value, new_node)
      tup = (cost, new_node)
      cur_node.children.append(new_node)
      pq.put(tup)
    # print("added", added)
    if pq.qsize() <= 0:
      break
    cur_node = pq.get()[1]
    cur_apex = cur_node.apex
    step = cur_node.step + 1
    iters += 1
    if cur_node.x_loc[0] > goal[0]:
      goal_nodes.append(cur_node)

  if iters >= timeout and len(goal_nodes) == 0:
    print("timeout! deepest node = ", deepest_node.x_loc[0])
  # print("Foot Space A* found", len(goal_nodes), "goal nodes")
  if len(goal_nodes) == 0:
    goal_nodes.append(deepest_node)
  if get_full_tree:
    all_paths, all_angles = inOrderHelper(root_node, goal[0])
    if count_odes:
      return all_paths, all_angles, total_odes
    else:
      return all_paths, all_angles
  else:
    sequences = []
    actual_locs = []
    inputs = []
    for cur_node in goal_nodes:
      traj = [cur_node.x_loc[1]]
      loc_seq = [cur_node.x_loc[0]]
      angles = [cur_node.prev_angle]
      while cur_node.parent is not None:
        cur_node = cur_node.parent
        traj.append(cur_node.x_loc[1])
        loc_seq.append(cur_node.x_loc[0])
        angles.append(cur_node.prev_angle)
      traj = list(reversed(traj))
      angles = list(reversed(angles))
      loc_seq = list(reversed(loc_seq))
      sequences.append(traj)
      actual_locs.append(loc_seq)
      inputs.append(angles)
    if count_odes:
      return sequences, inputs, total_odes, actual_locs
    else:
      return sequences, inputs, actual_locs


### RNN Guided A* Utilities ###
def path_from_parent(node):
  steps = []
  angles = []
  while node is not None:
    steps.append(node.x_loc)
    angles.append(node.prev_angle)
    node = node.parent
  return steps[::-1], angles[::-1]

def path_from_parent2(node, use_first = False):
  steps = []
  while node is not None:
    if use_first:
        steps.append(node.x_loc[0])
    else:
        steps.append(node.x_loc[1])
    node = node.parent
  return steps[::-1]

# sample from a 1D distribution
def discrete_sampling(distribution, num_samples,
                   min, max):
  disc = (max - min)/distribution.shape[0]
  domain = np.arange(min, max, disc)
  cdf = np.cumsum(distribution)
  uniforms = np.random.rand(num_samples)
  output_samples = []
  for u in uniforms:
    for i in range(len(cdf)):
      if u <= cdf[i]:
        output_samples.append(domain[i])
        break
  return output_samples


# Currently using controller inputs instead of true location inputs
# TODO: return/print some debugging information, or something like that.
def RNNGuidedAstar(robot, 
                   x0_apex,
                   goal,
                   rnn_planner,
                   step_controller,
                   terrain_func,
                   friction,
                   num_samples,
                   cost_fn,
                   debug = False):
  # initialize all the shits
  pq = queue.PriorityQueue()
  nodes = []
  num_goal_nodes = 0
  iters = 0
  timeout = 1000
  pos = np.arange(0, 8.0, 0.1)
  terrain_normal_func = lambda x: np.pi/2
  Ts = 0.17
  Tf = 0.60
  step = 0
  total_odes = 0

  cur_node = GraphNode(x0_apex[0], 0, x0_apex, 0, 0, None, [])
  deepest_node = cur_node
  cur_apex = x0_apex
  time_till_ground = 2 * (x0_apex[1] - robot.constants.L)/(-robot.constants.g)
  xstep_pred = x0_apex[0] + x0_apex[2] * time_till_ground

  cur_node.x_loc = (xstep_pred, xstep_pred)
  if goal[0] >= x0_apex[0] + 8.0:
    goal[0] = x0_apex[0] + 7.9

  while num_goal_nodes == 0 and iters < timeout:
    if debug:
      print("----ITER", iters, "----")
      print("on Node", cur_node.x_loc[0])
    if cur_node.x_loc[0] > deepest_node.x_loc[0]:
      deepest_node = cur_node

    # get a terrain array along this time horizon

    # currently this doesn't use the full step history.
    # prev_step = cur_node.x_loc - cur_apex[0]
    # planning_apex = [0, cur_apex[1], cur_apex[2]]
    # t_array = []
    # for p in pos:
    #   t_array.append(terrain_func(cur_node.x_loc + p))

    prev_steps = path_from_parent2(cur_node)
    # planning_apex = x0_apex[0:3]
    planning_apex = [0, x0_apex[1], x0_apex[2]]
    t_array = []
    for p in pos:
      t_array.append(terrain_func(x0_apex[0] + p))
    _, softmaxes = rnn_planner.predict(1, planning_apex, t_array, prev_steps)

    distribution = softmaxes[-1].cpu().detach().numpy()[0][0]
    next_samples = discrete_sampling(distribution, num_samples, min = -3, max = 8)
    for sample in next_samples:
      input = step_controller.calcAngle(sample - cur_apex[0] + x0_apex[0], cur_apex[2], 0, 0, y = cur_apex[1])
      apex1, apex2, last_flight, count = hopper.getNextState2Count(robot,
                                                      cur_apex,
                                                      input,
                                                      terrain_func,
                                                      terrain_normal_func,
                                                      friction,
                                                      at_apex = True)
      total_odes += count 
      if apex1 is not None:
        # calculate the cost at last flight (right before landing)
        cost = cost_fn(last_flight, [], goal, step) + np.random.rand() * 1e-4
        new_node = GraphNode((last_flight[0], x0_apex[0] + sample), 
                             input, apex2, 
                             step, cost + cur_node.value, 
                             cur_node, [])
        tup = (cost, new_node)
        cur_node.children.append(new_node)
        if debug:
          print("Sample step loc:", last_flight[0])
          print("Lstep to hit that sample:", sample + x0_apex[0] - cur_apex[0])
          print("Sample cost:", cost)
        pq.put(tup)
    if pq.qsize() <= 0:
      break
    cur_node = pq.get()[1]
    cur_apex = cur_node.apex
    step = cur_node.step + 1
    iters += 1
    if cur_node.x_loc[0] > goal[0]:
      num_goal_nodes += 1
      goal_node = cur_node

  if num_goal_nodes == 0:
    locs = path_from_parent2(deepest_node, use_first = True)
    return path_from_parent2(deepest_node), locs, total_odes

  # calculate the final path
  steps = path_from_parent2(goal_node)
  locs = path_from_parent2(goal_node, use_first = True)
  return steps, locs, total_odes


##### 2D Hopper Planning #####
def sampleAnglesInCircle(num_samples_sqrt, friction):
    thetas = []
    phis = []
    fric_range = np.linspace(0, friction, num_samples_sqrt)
    for f in fric_range:
        theta_range = np.linspace(-np.arctan(f), np.arctan(f), num_samples_sqrt)
        phi_range = np.arcsin(np.sqrt(f - np.sin(theta_range)**2))
        thetas += list(theta_range)
        phis += list(phi_range)
    thetas, phis = [], []
    while len(thetas) < num_samples_sqrt**2:
        theta = np.random.rand() * (2 * np.arctan(friction)) - np.arctan(friction)
        phi = np.random.rand() * (2 * np.arctan(friction)) - np.arctan(friction)
        if np.arcsin(np.sin(theta)**2 + np.sin(phi)**2) < np.arctan(friction):
            thetas.append(theta)
            phis.append(phi)
    return thetas, phis


def sampleAngles2D(robot, num_samples_sqrt, cur_apex, hmap):
    # pitch_angles = np.linspace(-np.arctan(friction), np.arctan(friction), num_samples_sqrt)
    # roll_angles = np.linspace(-np.arctan(friction), np.arctan(friction), num_samples_sqrt)
    # all_pitches, all_rolls = np.meshgrid(pitch_angles, roll_angles) 
    # all_pitches, all_rolls = all_pitches.flatten(), all_rolls.flatten()
    all_pitches, all_rolls = sampleAnglesInCircle(num_samples_sqrt, hmap.info["friction"])
    apexes, last_flights, angles = [], [], []
    total_count = 0
    for i in range(len(all_pitches)):
        u = [all_pitches[i], all_rolls[i]]
        # print("trying angles", u)
        code, last_flight, next_apex, count = hopper2d.getNextApex2D(robot, cur_apex, u, hmap, at_apex = True, return_count = True)
        # print("next apex:", next_apex)
        if code == 0:
            apexes.append(next_apex)
            last_flights.append(last_flight)
            angles.append(u)
        total_count += count
    return apexes, last_flights, angles, total_count 


def atGoal(node, goal_state, two_d = False, use_y = True):
    loc = node.x_loc
    if two_d:
        loc = node.x_loc[0]
    if use_y:
        return loc[0] >= goal_state[0] and loc[1] >= goal_state[1]
    else:
        return loc[0] >= goal_state[0]


def inOrderHelper2D(root_node, goal):
  all_paths = []
  all_angles = []
  if len(root_node.children) == 0:
    if atGoal(root_node, goal):
      all_paths = [[root_node.x_loc]]
      all_angles = [[root_node.prev_angle]]
    else:
      all_paths = [[]]
      all_angles = [[]]
  for child in root_node.children:
    paths, angles = inOrderHelper(child, goal)
    for i, p in enumerate(paths):
      all_paths.append([root_node.x_loc] + paths[i])
      all_angles.append([root_node.prev_angle] + angles[i])
  return all_paths, all_angles


def angleAstar2Dof(robot, x0_apex, goal_state, num_samples_sqrt, 
                   num_goal_des, cost_fn, hmap, get_full_tree = False):
    pq = queue.PriorityQueue()
    goal_nodes = []
    num_goal_nodes = 0
    iters = 0
    timeout = 500
    step = 0
    total_odes = 0

    cur_node = GraphNode([x0_apex[0], x0_apex[1]], 0, x0_apex, 0, 0, None, [])
    root_node = cur_node
    deepest_node = cur_node
    cur_apex = x0_apex
    time_till_ground = 2 * (x0_apex[2] - robot.constants.L)/(-robot.constants.g)
    xstep_pred = x0_apex[0] + x0_apex[3] * time_till_ground
    ystep_pred = x0_apex[1] + x0_apex[4] * time_till_ground

    cur_node.x_loc = [xstep_pred, ystep_pred]
    # Get landing location of first node, and add it to the queue
    # in a while loop: 
        # sample all inputs (discretized) and get the costs
        # add to the pq
        # dequeue top node
    while num_goal_nodes < num_goal_des and iters < timeout:
        # print("on node with apex", cur_node.apex[0:2], "step loc:", cur_node.x_loc)
        next_apexes, last_flights, angles, total_count = sampleAngles2D(robot, num_samples_sqrt, cur_apex, hmap)
        if len(last_flights) > 0:
            tree = KDTree(np.array([[p[0], p[1]] for p in last_flights]))
        total_odes += total_count
        for i in range(len(next_apexes)):
            dd, ii = tree.query([[last_flights[i][0], last_flights[i][1]]], k = 4)
            neighbors = []
            for neigh_idx in ii[0]:
                try:
                    neighbors.append([last_flights[neigh_idx][0], last_flights[neigh_idx][1]]) 
                except IndexError:
                    # print("Error; length of last_flights is", len(last_flights), "but neigh_idx = ", neigh_idx)
                    pass

            cost = cost_fn(last_flights[i], neighbors, cur_node.apex, goal_state, step) + np.random.randn() * 1e-4
            # print("considering node with loc", [last_flights[i][0], last_flights[i][1]])
            node = GraphNode([last_flights[i][0], last_flights[i][1]], angles[i], next_apexes[i], step,
                             cost + cur_node.value, cur_node, [])
            cur_node.children.append(node)
            pq.put((cost, node))

        if pq.qsize() == 0:
            break
        if pq.qsize() >= 10000:
            break

        cur_node = pq.get()[1]
        if atGoal(cur_node, goal_state, use_y = False):
            goal_nodes.append(cur_node) 
            num_goal_nodes += 1
            if pq.qsize() > 0:
                cur_node = pq.get()[1]
            else:
                break
        cur_apex = cur_node.apex
        step = cur_node.step + 1
        iters += 1

    if num_goal_nodes == 0 and not get_full_tree:
        # print("Couldn't find full path!")
        return [], [], total_odes
    
    if get_full_tree:
        all_paths, all_angles = inOrderHelper2D(root_node, goal_state)
        return all_paths, all_angles, total_odes
    else:
        all_locs, all_angles = [], []
        for goal_node in goal_nodes:
            locs, angles = path_from_parent(goal_node)
            all_locs.append(locs)
            all_angles.append(angles)
    return all_locs, all_angles, total_odes


def footSpaceAStar2D(robot, step_controller, x0_apex, goal_state, horizon, num_samples_sqrt,
                     num_goal_des, cost_fn, hmap, get_full_tree = False):
    pq = queue.PriorityQueue()
    goal_nodes = []
    num_goal_nodes = 0
    iters = 0
    timeout = 3000
    step = 0
    total_odes = 0

    cur_node = GraphNode([x0_apex[0], x0_apex[1]], 0, x0_apex, 0, 0, None, [])
    root_node = cur_node
    deepest_node = cur_node
    cur_apex = x0_apex
    time_till_ground = 2 * (x0_apex[2] - robot.constants.L)/(-robot.constants.g)
    xstep_pred = x0_apex[0] + x0_apex[3] * time_till_ground
    ystep_pred = x0_apex[1] + x0_apex[4] * time_till_ground
    cur_node.x_loc = [[xstep_pred, ystep_pred], [xstep_pred, ystep_pred]]

    while num_goal_nodes < num_goal_des and iters < timeout:
        next_xs = np.linspace(cur_node.x_loc[0][0] - 1, cur_node.x_loc[0][0] + horizon, num_samples_sqrt)
        next_ys = np.linspace(cur_node.x_loc[0][1] - 1, cur_node.x_loc[0][1] + horizon, num_samples_sqrt)
        xx, yy = np.meshgrid(next_xs, next_ys) 
        xx, yy = xx.flatten(), yy.flatten()
        total_count = 0
        next_apexes = []
        last_flights = []
        angles = []
        ctrlr_inputs = []
        for i in range(len(xx)):
            x_Lstep = xx[i] - cur_apex[0]
            y_Lstep = yy[i] - cur_apex[1]
            u = step_controller.calcAngle(x_Lstep, cur_apex[3], y_Lstep, cur_apex[4], cur_apex[2])
            code, last_flight, next_apex, count = hopper2d.getNextApex2D(robot, cur_apex, u, hmap, at_apex = True, return_count = True)
            total_count += count
            if code == 0:
                last_flights.append(last_flight)
                next_apexes.append(next_apex)
                angles.append(u)
                ctrlr_inputs.append([xx[i], yy[i]])

        for i in range(len(next_apexes)):
            if i > 0 and i < len(angles) - 1:
              neighbors = [last_flights[i-1], last_flights[i+1]]
            elif len(angles) == 1:
              neighbors = []
            elif i == 0:
              neighbors = [last_flights[i+1]]
            else:
              neighbors = [last_flights[i-1]]
            
            cost = cost_fn(last_flights[i], neighbors, goal_state, step) + np.random.randn() * 1e-4
            node = GraphNode([[last_flights[i][0], last_flights[i][1]], ctrlr_inputs[i]], angles[i], next_apexes[i], step,
                             cost + cur_node.value, cur_node, [])
            cur_node.children.append(node)
            pq.put((cost, node))

        if pq.qsize() <= 0:
            break
        if pq.qsize() >= 10000:
            break

        cur_node = pq.get()[1]
        if atGoal(cur_node, goal_state, two_d = True, use_y = False):
            goal_nodes.append(cur_node) 
            num_goal_nodes += 1
            cur_node = pq.get()[1]
        cur_apex = cur_node.apex
        step = cur_node.step + 1
        iters += 1

    if num_goal_nodes == 0 and not get_full_tree:
        # print("Couldn't find full path!")
        return [], [], total_odes
    
    if get_full_tree:
        all_paths, all_angles = inOrderHelper2D(root_node, goal_state)
        for i in range(0, len(all_paths)):
            all_paths[i] = [p[1] for p in all_paths[i]]  # select the controller input
        return all_paths, all_angles, total_odes
    else:
        all_locs, all_angles = [], []
        for goal_node in goal_nodes:
            locs, angles = path_from_parent(goal_node)
            locs = [l[1] for l in locs]  # select the controller input
            all_locs.append(locs)
            all_angles.append(angles)
    return all_locs, all_angles, total_odes


def discrete_sampling2D(distr, num_samples, disc):
    choices = np.prod(distr.shape)
    indices = np.random.choice(choices, size = num_samples, p = distr.ravel())
    i_2d = np.column_stack(np.unravel_index(indices, shape = distr.shape))
    # remember, y's are rows, x's are columns
    i_2d[:,0] = i_2d[:,0]/disc
    i_2d[:,1] = i_2d[:,1]/disc
    return i_2d


def RNNGuidedAStar2D(robot, step_controller, conv_rnn_planner, x0_apex, goal_state, num_samples,
                     num_goal_des, cost_fn, hmap, get_full_tree = False, sampling_method = "distribution"):
    pq = queue.PriorityQueue()
    goal_nodes = []
    num_goal_nodes = 0
    iters = 0
    timeout = 3000
    step = 0
    total_odes = 0

    cur_node = GraphNode([x0_apex[0], x0_apex[1]], 0, x0_apex, 0, 0, None, [])
    root_node = cur_node
    deepest_node = cur_node
    cur_apex = x0_apex
    time_till_ground = 2 * (x0_apex[2] - robot.constants.L)/(-robot.constants.g)
    xstep_pred = x0_apex[0] + x0_apex[3] * time_till_ground
    ystep_pred = x0_apex[1] + x0_apex[4] * time_till_ground
    cur_node.x_loc = [[xstep_pred, ystep_pred], [xstep_pred, ystep_pred]]

    # Currently assumes x0_apex is 0

    x_pos = np.arange(0, conv_rnn_planner.max_x, conv_rnn_planner.disc)
    y_pos = np.arange(-conv_rnn_planner.max_y//2, conv_rnn_planner.max_y//2, conv_rnn_planner.disc)
    xx, yy = np.meshgrid(x_pos, y_pos)
    # This initialization only works because x_pos shape = y_pos shape..?
    t_array = np.zeros((int(conv_rnn_planner.max_y/conv_rnn_planner.disc),
                        int(conv_rnn_planner.max_x/conv_rnn_planner.disc)))
    for i in range(t_array.shape[0]):
      for j in range(t_array.shape[1]):
        t_array[i][j] = hmap.at(xx[i][j] + x0_apex[0], yy[i][j] + x0_apex[1])

    while num_goal_nodes < num_goal_des and iters < timeout:
        if sampling_method == "distribution":
            ## RNN Changes this xx, yy ##
            prev_steps = path_from_parent2(cur_node)   
            outs, softmaxes = conv_rnn_planner.predict(1, cur_apex, t_array, [prev_steps])
            distribution = softmaxes[-1]
            next_samples = discrete_sampling2D(distribution, num_samples, conv_rnn_planner.disc)
            #############################
        elif sampling_method == "dropout":
            prev_steps = path_from_parent2(cur_node)
            next_samples = []
            for _ in range(num_samples):
                outs, hiddens = conv_rnn_planner.predict(1, cur_apex, t_array, [prev_steps])
                next_samples.append(outs[-1])
            # print("Next samples:", next_samples)
        else:
            print("Error! Unrecognized sampling method")
            return [], total_odes

        total_count = 0
        next_apexes = []
        last_flights = []
        angles = []
        ctrlr_inputs = []
        for i in range(len(next_samples)):
            xx = next_samples[i][1]
            yy = next_samples[i][0]

            x_Lstep = xx - cur_node.x_loc[0][0] + x0_apex[0]
            y_Lstep = yy - cur_node.x_loc[0][1] + x0_apex[1]
            u = step_controller.calcAngle(x_Lstep, cur_apex[3], y_Lstep, cur_apex[4], cur_apex[2])
            code, last_flight, next_apex, count = hopper2d.getNextApex2D(robot, cur_apex, u, hmap, at_apex = True, return_count = True)
            total_count += count
            # print("tried sample", next_samples[i], "got code,", code)
            if code == 0:
                last_flights.append(last_flight)
                next_apexes.append(next_apex)
                angles.append(u)
                ctrlr_inputs.append([xx + x0_apex[0], yy + x0_apex[1]])

        for i in range(len(next_apexes)):
            if i > 0 and i < len(angles) - 1:
              neighbors = [last_flights[i-1], last_flights[i+1]]
            elif len(angles) == 1:
              neighbors = []
            elif i == 0:
              neighbors = [last_flights[i+1]]
            else:
              neighbors = [last_flights[i-1]]
            
            cost = cost_fn(last_flights[i], neighbors, goal_state, step) + np.random.randn() * 1e-4
            node = GraphNode([[last_flights[i][0], last_flights[i][1]], ctrlr_inputs[i]], angles[i], next_apexes[i], step,
                             cost + cur_node.value, cur_node, [])
            cur_node.children.append(node)
            pq.put((cost, node))

        if pq.qsize() <= 0:
            break
        if pq.qsize() >= 10000:
            break

        cur_node = pq.get()[1]
        if atGoal(cur_node, goal_state, two_d = True):
            goal_nodes.append(cur_node) 
            num_goal_nodes += 1
            cur_node = pq.get()[1]
        cur_apex = cur_node.apex
        step = cur_node.step + 1
        iters += 1

    if num_goal_nodes == 0:
        return path_from_parent2(deepest_node), total_odes
    steps = path_from_parent2(goal_nodes[0])
    return steps, total_odes

def cost_fn2d(state, neighbors, apex, goal_state, step):
    x = state[0]
    y = state[1]
    return np.abs(x - goal_state[0]) + np.abs(y - goal_state[1])

def terrain_func(x, y):
    if x >= 1 and x <= 3 and y >= 0 and y <= 1:
        return -1
    if x >= 1 and x <= 5 and y >= 2 and y <= 3:
        return -1
    if x >= 3 and x <= 4 and y >= -1  and y <= 1:
        return -1
    return 0


def main():
    robot = hopper2d.Hopper2D(hopper.Constants())
    hmap = heightmap.randomDitchHeightMap(5, 2, 0.1, 0.8, 4)
    hmap.plotSteps([])
    initial_state = hopper2d.FlightState2D()
    initial_state.x = 0
    initial_state.y = 0
    initial_state.z = 1.1
    initial_state.zf = initial_state.z - robot.constants.L
    initial_apex = initial_state.getArray()
    goal = [3, 0, 0, 0]
    cost_fn = cost_fn2d
    all_locs, all_angles, odes = angleAstar2Dof(robot, initial_apex, goal, 12, 
                           3, cost_fn, hmap, get_full_tree = False)
    print(all_locs)
    return

if __name__ == "__main__":
    main()
