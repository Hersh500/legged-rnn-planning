import numpy as np
import hopper
import hopper2d
import queue

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

  # For now, just uniformly sampling angles between forward and rear limit.
  # Later, we can choose a more intelligent sampling approach.
  angles = []
  costs = []
  apexes = []
  locs = []
  last_flights = []
  total_count = 0

  # try gaussian sampling around the neutral point
  mean = np.arccos(x_apex[2] * 0.18 / 2)
  cov = (max_limit - min_limit)/cov_factor

  if neutral_angle_normal:
    pos = np.random.randn(num_samples) * cov + mean
  else:
    pos = np.linspace(min_limit, max_limit, num_samples)
    # pos = np.random.rand(num_samples) * (max_limit - min_limit) + min_limit
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


# Performs an In-order traversal through the treeeeeeeee
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
    if cur_node.x_loc >= goal[0] and node_queue.qsize() > 0:
      goal_nodes.append(cur_node)
      cur_node = node_queue.get()[1]
      cur_apex = cur_node.apex
      step = cur_node.step + 1
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
          node_queue.put((costs[i], node))

      if node_queue.qsize() <= 0:
        break
      if node_queue.qsize() >= 10000:
        break

      cur_node = node_queue.get()[1]
      cur_apex = cur_node.apex
      step = cur_node.step + 1
      iters += 1

  # now that the while loop has terminated, we can reconstruct the trajectory
  if len(goal_nodes) == 0:
    goal_nodes.append(deepest_node)
  # else:
    # print("A* found", len(goal_nodes), "goal nodes!")

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


### RNN Guided A* Utilities ###

def path_from_parent(node):
  steps = []
  while node is not None:
    steps.append(node.x_loc)
    node = node.parent
  return steps[::-1]

def path_from_parent2(node):
  steps = []
  while node is not None:
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
    planning_apex = x0_apex[0:3]
    t_array = []
    for p in pos:
      t_array.append(terrain_func(x0_apex[0] + p))
    _, softmaxes = rnn_planner.predict(1, planning_apex, t_array, prev_steps)

    distribution = softmaxes[-1].cpu().detach().numpy()[0][0]
    next_samples = discrete_sampling(distribution, num_samples, min = -3, max = 8)
    for sample in next_samples:
      input = step_controller.calcAngle(sample, cur_apex[2], 0, 0, y = cur_apex[1])
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
        new_node = GraphNode((last_flight[0], cur_apex[0] + sample), 
                             input, apex2, 
                             step, cost + cur_node.value, 
                             cur_node, [])
        tup = (cost, new_node)
        cur_node.children.append(new_node)
        if debug:
          print("Sample step loc:", last_flight[0])
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
    return path_from_parent2(deepest_node), total_odes

  # calculate the final path
  steps = path_from_parent2(goal_node)
  return steps, total_odes


##### 2D Hopper Planning #####
def sampleAngles2D(robot, num_samples_sqrt, cur_apex, terrain_func, terrain_normal_func, friction):
    pitch_angles = np.linspace(-np.arctan(friction), np.arctan(friction), num_samples_sqrt)
    roll_angles = np.linspace(-np.arctan(friction), np.arctan(friction), num_samples_sqrt)
    all_pitches, all_rolls = np.meshgrid(pitch_angles, roll_angles) 
    all_pitches, all_rolls = all_pitches.flatten(), all_rolls.flatten()
    apexes, last_flights, angles = [], [], []
    total_count = 0
    for i in range(len(all_pitches)):
        u = [all_pitches[i], all_rolls[i]]
        # print("trying angles", u)
        code, last_flight, next_apex, count = hopper2d.getNextApex2D(robot, cur_apex, u, terrain_func, terrain_normal_func,
                                                                    friction, at_apex = True, return_count = True)
        # print("next apex:", next_apex)
        if code == 0:
            apexes.append(next_apex)
            last_flights.append(last_flight)
            angles.append(u)
        total_count += count
    
    return apexes, last_flights, angles, total_count 


def atGoal(node, goal_state):
    loc = node.x_loc
    return loc[0] >= goal_state[0] and loc[1] >= goal_state[1]


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
                   num_goal_des, cost_fn, terrain_func, terrain_normal_func,
                   friction, get_full_tree = False):
    pq = queue.PriorityQueue()
    goal_nodes = []
    num_goal_nodes = 0
    iters = 0
    timeout = 1000
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
        print("on node with apex", cur_node.apex[0:2], "step loc:", cur_node.x_loc)
        next_apexes, last_flights, angles, total_count = sampleAngles2D(robot, num_samples_sqrt, cur_apex, 
                                                                        terrain_func, terrain_normal_func, friction)
        total_odes += total_count
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
            # print("considering node with loc", [last_flights[i][0], last_flights[i][1]])
            node = GraphNode([last_flights[i][0], last_flights[i][1]], angles[i], next_apexes[i], step,
                             cost + cur_node.value, cur_node, [])
            cur_node.children.append(node)
            pq.put((cost, node))

        if pq.qsize() <= 0:
            break
        if pq.qsize() >= 10000:
            break

        cur_node = pq.get()[1]
        if atGoal(cur_node, goal_state):
            goal_nodes.append(cur_node) 
            num_goal_nodes += 1
            cur_node = pq.get()[1]
        cur_apex = cur_node.apex
        step = cur_node.step + 1
        iters += 1

    if num_goal_nodes == 0:
        print("Couldn't find full path!")
        return [], []
    
    if get_full_tree:
        all_paths, all_angles = inOrderHelper2D(root_node, goal_state)
        return all_paths, total_odes
    else:
        all_locs, all_angles = [], []
        for goal_node in goal_nodes:
            locs = path_from_parent(goal_node)
            all_locs.append(locs)
    return all_locs, total_odes


def cost_fn(state, neighbors, goal_state, step):
    x = state[0]
    y = state[1]
    return np.abs(x - goal_state[0]) + 2 * np.abs(y - goal_state[1])

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
    # terrain_func = lambda x,y: 0
    terrain_normal_func = lambda x,y: np.pi/2
    friction = 0.8
    initial_state = hopper2d.FlightState2D()
    initial_state.x = 0
    initial_state.y = 0
    initial_state.z = 1.1
    initial_state.zf = initial_state.z - robot.constants.L
    initial_apex = initial_state.getArray()
    goal = [5, 0, 0, 0]
    paths, odes = angleAstar2Dof(robot, initial_apex, goal, 12, 
                           3, cost_fn, terrain_func, terrain_normal_func,
                           friction, get_full_tree = False)
    print(paths)
    return


if __name__ == "__main__":
    main()
