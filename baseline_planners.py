import numpy as np
import hopper
from hopper import constants
import astar_tree_search
from astar_tree_search import aStarHelper, angleAstar2Dof

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


class AStarPlanner2D:
    def __init__(self, robot, num_samples_sqrt, fallback_samples, max_speed, cost_matrix):
        self.num_samples_sqrt = num_samples_sqrt
        self.fallback_samples = fallback_samples
        self.robot = robot

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

            return (np.sqrt(cost_matrix[0] * np.abs(x_pos - goal[0])**2 + cost_matrix[1] * np.abs(y_pos - goal[1])**2) +
                   cost_matrix[2] * np.abs(x_vel - goal[2]) + cost_matrix[3] * np.abs(y_vel - goal[3])) + cost_matrix[4] * spread

        self.cost_fn = costFn

    def predict(self, initial_apex, terrain_func, friction, goal, use_fallback, timeout = 1000, debug = False):
        if use_fallback:
            num_samples = self.fallback_samples
        else:
            num_samples = self.num_samples_sqrt
        step_sequences, angles, count = angleAstar2Dof(self.robot,
                                               initial_apex,
                                               goal,
                                               num_samples,
                                               1,
                                               self.cost_fn,
                                               terrain_func,
                                               lambda x: np.pi/2,
                                               friction,
                                               get_full_tree = False)
        if len(step_sequences) > 0:
            return step_sequences[0], angles[0], count
        else:
            return [], [], count

class FootSpaceAStarPlanner2D:
    def __init__(self, robot, step_controller, horizon, num_samples_sqrt, cost_matrix):
        self.num_samples_sqrt = num_samples_sqrt
        self.robot = robot
        self.step_controller = step_controller
        self.horizon = horizon
        self.num_samples_sqrt = num_samples_sqrt

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


    def predict(self, initial_apex, terrain_func, friction, goal, use_fallback, timeout = 1000, debug = False):
        steps, angles, odes = astar_tree_search.footSpaceAStar2D(self.robot, 
                                                                self.step_controller,
                                                                initial_apex, goal, self.horizon,
                                                                self.num_samples_sqrt, 1, self.cost_fn,
                                                                terrain_func, lambda x,y: np.pi/2, friction)
        if len(steps) > 0:
            return steps[0], angles[0], odes
        else:
            return [], [], odes
