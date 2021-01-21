import numpy as np
import hopper
from hopper import constants
import astar_tree_search
from astar_tree_search import aStarHelper
'''
    Uses Astarhelper in the same planning framework
'''
class AStarPlanner:
    def __init__(self, robot, num_samples, fallback_samples, max_speed, cost_matrix):
        self.max_speed = max_speed
        self.num_samples = num_samples
        self.fallback_samples = fallback_samples
        self.robot = robot

        def costFn(x_flight, neighbors, goal, p):
            x_pos = x_flight[0]
            x_vel = x_flight[2]
            spread = 0
            for n in range(len(neighbors)):
                spread += np.abs(neighbors[n][0] - x_pos)/len(neighbors)

            return cost_matrix[0] * np.abs(x_pos - goal[0]) + cost_matrix[1] * np.abs(x_vel - goal[1]) + cost_matrix[2] * spread

        self.cost_fn = costFn

    def predict(self, initial_apex, terrain_func, friction, goal, use_fallback, timeout = 1000, debug = False):
        if use_fallback:
            num_samples = self.fallback_samples
        else:
            num_samples = self.num_samples
        step_sequences, angle_sequences, count = aStarHelper(self.robot,
                                                      initial_apex,
                                                      goal, 1,
                                                      terrain_func,
                                                      lambda x: np.pi/2,
                                                      friction,
                                                      num_angle_samples = num_samples,
                                                      timeout = timeout,
                                                      neutral_angle = False,
                                                      max_speed = self.max_speed,
                                                      cost_fn = self.cost_fn,
                                                      count_odes = True)
        if len(step_sequences) > 0:
            return step_sequences[0], angle_sequences[0], count
        else:
            return [], [], count


'''
    Same as the previous planner, but instead of discretizing the friction cone to get inputs,
    samples from a Gaussian distribution centered on the neutral angle.
'''
class StochasticAStarPlanner:
    def __init__(self, num_samples, fallback_samples, max_speed, cost_matrix):
        self.max_speed = max_speed
        self.num_samples = num_samples
        self.fallback_samples = fallback_samples

        def costFn(x_flight, neighbors, goal, p):
            x_pos = x_flight[0]
            x_vel = x_flight[2]
            return cost_matrix[0] * np.abs(x_pos - goal[0]) - cost_matrix[1] * np.abs(x_vel - goal[1])

        self.cost_fn = costFn

    def predict(self, initial_apex, terrain_func, friction, goal, use_fallback, timeout = 1000):
        if use_fallback:
            num_samples = self.fallback_samples
        else:
            num_samples = self.num_samples
        step_sequences, angle_sequences = aStarHelper(initial_apex,
                                                      goal, 1,
                                                      terrain_func,
                                                      lambda x: np.pi/2,
                                                      friction,
                                                      num_angle_samples = num_samples,
                                                      timeout = timeout,
                                                      cov_factor = 2,
                                                      neutral_angle = True,
                                                      max_speed = self.max_speed,
                                                      cost_fn = self.cost_fn)
        if len(step_sequences) > 0:
            return step_sequences[0], angle_sequences[0]
        else:
            return [], []
