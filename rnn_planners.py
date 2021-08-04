import torch
import numpy as np

import models
import astar_tree_search


# LSTM-Guided Planner
class RNNAStarPlanner:
  def __init__(self, robot, rnn_planner, step_controller, num_samples, fallback_samples, cost_matrix):
    self.num_samples = num_samples
    self.fallback_samples = fallback_samples
    self.rnn_planner = rnn_planner
    self.step_controller = step_controller
    self.robot = robot
    def costFn(x_flight, neighbors, goal, p):
      x_pos = x_flight[0]
      x_vel = x_flight[2]
      spread = 0
      # calculate spread of neighbors
      for n in range(len(neighbors)):
        spread += np.abs(neighbors[n][0] - x_pos)/len(neighbors)
      return cost_matrix[0] * np.abs(x_pos - goal[0]) + cost_matrix[1] * np.abs(x_vel - goal[1]) + cost_matrix[2] * spread
    self.cost_fn = costFn

  def predict(self, initial_apex, terrain_func, friction, goal, 
              use_fallback = False, timeout = 1000, debug = False):
    if use_fallback:
        num_samples = self.fallback_samples
    else:
        num_samples = self.num_samples
    step_sequence, locs, count = astar_tree_search.RNNGuidedAstar(self.robot,
                                                     initial_apex,
                                                     goal,
                                                     self.rnn_planner,
                                                     self.step_controller,
                                                     terrain_func,
                                                     friction,
                                                     num_samples,
                                                     self.cost_fn,
                                                     debug = debug)
    return step_sequence, locs, count


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


class RNNAStarPlanner2D:
    def __init__(self, robot, rnn_planner, step_controller, num_samples, cost_matrix, sampling_method):
        self.robot = robot
        self.rnn_planner = rnn_planner
        self.step_controller = step_controller
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        def costFn(x_flight, neighbors, goal, p):
            x_pos = x_flight[0]
            y_pos = x_flight[1]
            
            x_vel = x_flight[3]
            y_vel = x_flight[4]

            spread = 0
            for n in range(len(neighbors)):
                x_dist = (neighbors[n][0] - x_pos)**2
                y_dist = (neighbors[n][1] - y_pos)**2
                spread += np.sqrt(x_dist + y_dist)/len(neighbors)

            return (np.sqrt(cost_matrix[0] * np.abs(x_pos - goal[0])**2 + cost_matrix[1] * np.abs(y_pos - goal[1])**2) +
                   cost_matrix[2] * np.abs(x_vel - goal[2]) + cost_matrix[3] * np.abs(y_vel - goal[3])) + cost_matrix[4] * spread

        self.cost_fn = costFn
        return
    
    def predict(self, initial_apex, terrain_func, friction, goal,
                use_fallback = False, timeout = 1000, debug = False):
        
        steps, odes = astar_tree_search.RNNGuidedAStar2D(self.robot, 
                                                         self.step_controller,
                                                         self.rnn_planner,
                                                         initial_apex, goal, self.num_samples,
                                                         1, self.cost_fn,
                                                         terrain_func, lambda x,y: np.pi/2, friction,
                                                         sampling_method = self.sampling_method)
        if len(steps) > 0:
            return steps, [], odes
        else:
            return [], [], odes
        return

'''
    2D Convolutional Recurrent planner; the input is a 2 layer thing of
    the one-hot step and the terrain
'''
class ConvRNNPlanner2D:
  def __init__(self, rnn_model, device, max_x = 5, max_y = 5, disc = 0.1, T = 1, ve_dim = 0):
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.T = T
    self.max_x = max_x
    self.max_y = max_y
    self.disc = disc

  def predict(self, n, initial_apex, terrain_matrix, first_steps):
    seq = first_steps 
    outs, softmaxes, hiddens = models.evaluateConvModel2D(self.model,
                                                          n,
                                                          initial_apex,
                                                          seq, 
                                                          terrain_matrix, 
                                                          self.device,
                                                          T = self.T,
                                                          max_x = self.max_x,
                                                          max_y = self.max_y,
                                                          disc = self.disc)
    return outs, softmaxes
