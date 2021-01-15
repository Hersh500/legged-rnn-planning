import torch
import numpy as np

import models
import astar_tree_search

'''
    Uses the RNN planner to guide the astar search
'''
class RNNAStarPlanner:
  def __init__(self, rnn_planner, step_controller, num_samples, fallback_samples, cost_matrix):
    self.num_samples = num_samples
    self.fallback_samples = fallback_samples
    self.rnn_planner = rnn_planner
    self.step_controller = step_controller
    def costFn(x_flight, neighbors, goal, p):
      x_pos = x_flight[0]
      x_vel = x_flight[2]
      spread = 0
      # calculate spread of neighbors
      for n in range(len(neighbors)):
        spread += np.abs(neighbors[n][0] - x_pos)/len(neighbors)
      return cost_matrix[0] * np.abs(x_pos - goal[0]) + cost_matrix[1] * np.abs(x_vel - goal[1]) + cost_matrix[2] * spread
    self.cost_fn = costFn

  def predict(self, robot, initial_apex, terrain_func, friction, goal, 
              use_fallback = False, timeout = 1000, debug = False):
    if use_fallback:
        num_samples = self.fallback_samples
    else:
        num_samples = self.num_samples
    step_sequence, count = astar_tree_search.RNNGuidedAstar(robot,
                                                     initial_apex,
                                                     goal,
                                                     self.rnn_planner,
                                                     self.step_controller,
                                                     terrain_func,
                                                     friction,
                                                     num_samples,
                                                     self.cost_fn,
                                                     debug = debug)
    return step_sequence, [], count


'''
    Convolutional Recurrent planner; the input is a 2 layer thing of
    the one-hot step and the terrain
'''
class ConvRNNPlanner:
  def __init__(self, rnn_model, device, min_limit, T = 1, ve_dim = 0):
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.min_limit = min_limit
    self.ve_dim = ve_dim
    self.T = T

  def predict(self, n, initial_apex, terrain_list, first_steps):
    try:
      _ = len(first_steps)
      seq = [first_steps]
    except TypeError:
      seq = [[first_steps]]
    outs, softmaxes, hiddens = models.evaluateConvModel(self.model,
                                                        n,
                                                        initial_apex,
                                                        seq, 
                                                        terrain_list, 
                                                        self.device,
                                                        T = self.T)
    return outs, softmaxes


'''
    Normal RNN/LSTM planner, but feeds in terrain as input at each timestep
    (appends it to the step sequence
'''
class RNNPlannerTerrainInput:
  def __init__(self, rnn_model, device, min_limit, ve_dim = 0):
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.min_limit = min_limit
    self.ve_dim = ve_dim

  def predict(self, n, initial_apex, terrain_list, first_step):
    outs, softmaxes, hiddens = evaluateModelWithTerrainInput(self.model,
                                                             n,
                                                             initial_apex,
                                                             [[first_step]], 
                                                             terrain_list, 
                                                             device)
    return outs, softmaxes


'''
    Vanilla RNN Planner that simply plans N steps by taking the argmax of the softmax
    - Doesn't take terrain as input; instead, uses terrain to initialize the hidden state.
'''
class RNNPlanner:
  def __init__(self, rnn_model, device, min_limit, ve_dim = 0):
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.min_limit = min_limit
    self.ve_dim = ve_dim

  def predict(self, n, initial_apex, terrain_list, first_step):
    if self.ve_dim > 0:
      vel_encoded = [np.sin(i * initial_apex[2]) for i in range(self.ve_dim)]
      datapoint = np.array(list(terrain_list) + initial_apex[:-1] + vel_encoded)
    else:
      datapoint = np.array(list(terrain_list) + list(initial_apex[:3]))

    init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(self.device)

    prev_steps_oh = oneHotEncodeSequences([[first_step]])
    input = torch.Tensor(np.array(prev_steps_oh)).view(1, 1, -1).to(self.device)
    outs = []
    hiddens = []
    softmaxes = []
    for i in range(n):
      out, hidden = self.model(input, init_state)
      out = out[:,-1].view(1, 1, -1)
      outs.append(softmaxToStep(F.softmax(out, dim=2))[0][0].item())
      hiddens.append(hidden)
      out_processed = F.softmax(out, dim = 2)  # TODO: make this OH
      softmaxes.append(out_processed[0][0].detach().cpu().numpy())
      # out_oh = softmaxToOH(out_processed)
      input = torch.cat((input, out_processed.float()), dim=1)
    return outs, softmaxes


'''
Feeds in a terrain decomposition to the RNN, then combines the
output softmaxes to produce the final distribution
'''
class FeatureCompositionRNNPlanner:
  def __init__(self, rnn_model, device, combination_fn, min_limit):
    self.combination_fn = combination_fn
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.min_limit = min_limit
    return
  
  def combineDistributions(self, distros):
    total_distr = self.combination_fn(distros[0], distros[1])
    for d in distros[2:]:
      total_distr = self.combination_fn(total_distr, d)
    return total_distr

  # terrain_arrays is each of the individual features.
  # returns the steps + combined distributions
  def predict(self, num_steps, initial_apex, terrain_arrays, first_step):
    prev_steps_oh = oneHotEncodeSequences([[first_step]])
    input = torch.Tensor(np.array(prev_steps_oh)).view(1, 1, -1).to(self.device)
    steps = [first_step]
    total_distributions = []
    for i in range(num_steps):
      step_distributions = []  # list of torch tensors
      for ta in terrain_arrays:
        datapoint = np.array(list(ta) + initial_apex[:3])
        init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(self.device)
        out, hidden = self.model(input, init_state)
        out_sf = F.softmax(out, dim=2)
        sf = out_sf[:,-1].view(1, 1, -1)
        step_distributions.append(sf)
      # combine the distributions and calculate the next step
      if len(terrain_arrays) > 1:
        combined = self.combineDistributions(step_distributions)
      else:
        combined = step_distributions[0]
      next_step = softmaxToStep(combined)
      steps.append(next_step.item())
      input = torch.cat((input, combined.float()), dim=1)
      total_distributions.append(combined[0][0].detach().cpu().numpy())  # this might be bugged.
    return steps, total_distributions
