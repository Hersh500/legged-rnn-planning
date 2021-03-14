import numpy as np
import hopper
from hopper import constants
import astar_tree_search
from astar_tree_search import aStarHelper, angleAstar2Dof, footSpaceAStar

# Angle-space Planner
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


# Heuristic Planner that plans for a fixed stride
class HeuristicPlanner:
    def __init__(self, stride, buffer):
        self.stride = stride
        self.buffer = buffer

    # predict a fixed distance away, adjusting footsteps
    # if they land in ditches.
    # All the extra parameters are for consistency with the Astar planner interface.
    def predict(self, x0_apex, terrain_func, friction, goal, use_fallback, timeout=1000, debug = False):
        stride = max(1, self.stride * x0_apex[2])
        time_till_ground = 2 * (x0_apex[1] - hopper.constants.L)/(-hopper.constants.g)
        xstep_pred = x0_apex[0] + x0_apex[2] * time_till_ground
        next_step_loc = xstep_pred
        steps = [next_step_loc]
        use_fw = False
        while next_step_loc < goal[0]:
            next_step_loc = next_step_loc + stride
            if terrain_func(next_step_loc) < 0:
                # find the nearest step point that is on flat ground.
                s_fw = next_step_loc
                while terrain_func(s_fw) < 0:
                    s_fw = s_fw + 0.1
                s_bw = next_step_loc
                while terrain_func(s_bw) < 0:
                    s_bw = s_bw - 0.1

                if np.abs(s_fw - next_step_loc) < np.abs(s_bw - next_step_loc) or use_fw:
                    next_step_loc = s_fw + self.buffer
                    use_fw = False
                else:
                    next_step_loc = s_bw - self.buffer
                    use_fw = True
            steps.append(next_step_loc)
        return steps, [], 0  # zero ODE calls


# Step-space planner
class FootSpaceAStarPlanner:
    def __init__(self, robot, horizon, spacing, cost_matrix, step_controller):
        self.horizon = horizon
        self.spacing = spacing
        self.step_controller = step_controller
        self.robot = robot
        def costFn(x_flight, neighbors, goal, p):
            x_pos = x_flight[0]
            x_vel = x_flight[2]
            spread = 0
            for n in range(len(neighbors)):
                spread += np.abs(neighbors[n][0] - x_pos)/len(neighbors)

            return cost_matrix[0] * np.abs(x_pos - goal[0]) + cost_matrix[1] * np.abs(x_vel - goal[1]) + cost_matrix[2] * spread

        self.cost_fn = costFn

    def predict(self, initial_apex, terrain_func, friction,
                goal, use_fallback, timeout = 2000, debug = False):
        step_sequences, angle_sequences, count, loc_seq = footSpaceAStar(self.robot,
                                                                initial_apex,
                                                                goal, 1,
                                                                self.step_controller,
                                                                terrain_func,
                                                                friction,
                                                                self.horizon,
                                                                self.spacing,
                                                                cost_fn = self.cost_fn,
                                                                get_full_tree = False,
                                                                count_odes = True,
                                                                timeout = timeout,
                                                                debug = debug)
        if len(step_sequences) > 0:
            ss = step_sequences[0]
            return ss, loc_seq[0], count
        else:
            return [], [], count


### 2D Case ###
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
