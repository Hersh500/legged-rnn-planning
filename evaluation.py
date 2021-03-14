import numpy as np
import hopper
import test_utils
import time

def truncate(x, n):
    factor = 10**n
    tmp = float(int(x * factor))
    return tmp/factor


def simulateRecedingHorizonAStar(robot,
                                 x0_apex, 
                                 planner,
                                 step_controller,
                                 terrain_func,
                                 friction,
                                 time_to_replan = 3,
                                 tstep = 0.01,
                                 debug = False,
                                 prints = True,
                                 use_locs = False):
  
  goal_x = 10
  body_poses = []
  foot_poses = []
  plans = []
  t = 0
  Ts = 0.17
  Tf = 0.60
  success = False
  initial_flight_state = x0_apex
  cur_x = x0_apex[0]
  num_steps_hit = 0
  step_locs = []
  ttr = time_to_replan
  max_steps = 40
  replan = [0]
  angles = [0]
  need_to_replan = False
  total_odes = 0
  all_times = []
  while cur_x < goal_x and num_steps_hit < max_steps: # of footsteps + place of first step.
    # sample the terrain from here to 8m in the future
    code, flight_states, t_d = robot.simulateOneFlightPhaseODE(None,
                                                              tstep = tstep,
                                                              x_init = initial_flight_state,
                                                              u = np.pi/2,
                                                              debug=True,
                                                              terrain_func = terrain_func,
                                                              till_apex = True)
    

    flight_body_poses = [[x[0], x[1]] for x in flight_states]
    flight_foot_poses = [robot.getFootXYInFlight(x) for x in flight_states]
    t += t_d

    body_poses += flight_body_poses
    foot_poses += flight_foot_poses
    for _ in flight_body_poses:
      if use_locs:
        plans.append(angles)
      else:
        plans.append(replan)


    if code < 0:
      if prints:
        print("quitting simulation in flight pre-apex...")
      break

    # now this is post-apex, where we re-plan.
    apex_state = flight_states[-1]
    cur_x = apex_state[0]
    cur_y = apex_state[1]
    xdot = apex_state[2]

    if ttr == time_to_replan or need_to_replan:
      old_plan = replan
      need_to_replan = False
      start_time = time.time()
      replan, angles, count = planner.predict(apex_state, 
                                             terrain_func,
                                             friction, 
                                             goal = [max(goal_x, goal_x+4), 0],
                                             use_fallback = False,
                                             timeout = 2000,
                                             debug = debug)
      plan_time = time.time()
      all_times.append(plan_time - start_time)
      if prints:
        print("time to plan:", plan_time - start_time)
      total_odes += count
      if len(replan) <= time_to_replan:
        if prints:
          print("A* failed to find long enough path first try!")
        replan, angles, count = planner.predict(apex_state, 
                                               terrain_func, 
                                               friction,
                                               goal = [max(goal_x, goal_x+4), 0],
                                               use_fallback = True,
                                               timeout = 2000,
                                               debug = debug)
        # # TEMPORARY (DELETE THIS NEPHEW)
        # replan = dataset_generation.processSeq(replan, 0.2, 0.2, terrain_func)
        total_odes += count
        if len(replan) <= time_to_replan and len(old_plan[time_to_replan:]) > 2:
          replan = old_plan[time_to_replan:]
          need_to_replan = True
        elif len(replan) <= time_to_replan:
          if prints:
            print("Couldn't find plan at all!, len of old plan = ", len(old_plan))
          code = hopper.sim_codes["A*_NO_PATH"]
          break
      if prints:
        print("======REPLAN=====")
        print(replan)
        print(angles)
        print("for apex", apex_state[:3])
      Lstep = replan[1] - cur_x
      ttr = 1
    else:
      Lstep = replan[ttr+1] - cur_x
      ttr += 1
    if prints:
      print("using LStep = ", Lstep)
    leg_angle_des = step_controller.calcAngle(Lstep, xdot, Tf, Ts, y = cur_y)
    # leg_angle_des = angles[s+1]
    if debug:
      print("--- Approaching Step:", num_steps_hit)
      print("leg angle des = ", truncate(leg_angle_des, 2), "For LStep =", Lstep)
      if len(angles) > ttr:
        print("A* predicted angle = ", truncate(angles[ttr], 2))
      # print("A* predicted angle:", angles[s])

    code, flight_states, t_d = robot.simulateOneFlightPhaseODE(None,
                                                      tstep = tstep,
                                                      x_init = apex_state,
                                                      u = leg_angle_des,
                                                      debug=True,
                                                      terrain_func = terrain_func,
                                                      hit_apex = True)
    flight_body_poses = [[x[0], x[1]] for x in flight_states]
    flight_foot_poses = [robot.getFootXYInFlight(x) for x in flight_states]
    body_poses += flight_body_poses
    foot_poses += flight_foot_poses
    for _ in flight_body_poses:
      if use_locs:
        plans.append(angles)
      else:
        plans.append(replan)

    if code < 0:
      if prints:
        print("Failure in post-apex flight")
      break

    last_flight_state = flight_states[len(flight_states) - 1]
    t += t_d

    # print("flight time", t_d)
    stance_foot_pose = robot.getFootXYInFlight(last_flight_state)
    if debug:
      print("-----Step", num_steps_hit, "-----")
      print("step foot loc=", stance_foot_pose)
      print("step body loc=", last_flight_state[0], "vs. planned loc", replan[ttr-1])

    step_locs.append(last_flight_state[0])
    code, stance_states, t_d = robot.simulateOneStancePhase(last_flight_state,
                                                          tstep=tstep,
                                                          terrain_func = terrain_func,
                                                         terrain_normal_func = lambda x: np.pi/2, 
                                                         friction = friction)
    stance_body_poses = [robot.getBodyXYInStance(x, stance_foot_pose) for x in stance_states]
    stance_foot_poses = [stance_foot_pose for x in stance_states]

    for _ in stance_body_poses:
      if use_locs:
        plans.append(angles)
      else:
        plans.append(replan)


    body_poses += stance_body_poses
    foot_poses += stance_foot_poses
    if code < 0:
      if prints:
        print("Quitting simulation in stance...")
      break
    initial_flight_state = robot.stanceToFlight(stance_states[len(stance_states) - 1], stance_foot_pose)
    Ts = t_d  # estimate the stance time
    t += t_d
    num_steps_hit += 1
    # print("stance time", t_d)
    cur_x = initial_flight_state[0]
  if prints:
    print("Step locs:", step_locs)
  if cur_x >= goal_x:
    code = hopper.sim_codes["SUCCESS"]
  elif code == 0:
    code = hopper.sim_codes["TOO_MANY_STEPS"]
  avg_time = np.mean(all_times)
  return code, body_poses, foot_poses, plans, num_steps_hit, total_odes, avg_time


# used to create animations for a particular test in the assay, for visualization purposes
# Also for debugging
def testPlannerSingle(robot, planner, step_controller, test_matrix, time_to_replan, terrain_idx, apex_idx, friction, tstep = 0.01):
  terrain_func = test_matrix.getFunctions()[terrain_idx]
  initial_apex = test_matrix.apexes[apex_idx]

  code, body_poses, foot_poses, plans, num_steps_hit, total_odes, avg_time = simulateRecedingHorizonAStar(robot,
                                                                                                initial_apex,
                                                                                                planner,
                                                                                                step_controller,
                                                                                                terrain_func,
                                                                                                time_to_replan = time_to_replan,
                                                                                                friction = friction,
                                                                                                tstep = tstep,
                                                                                                debug = True)
  return code, body_poses, foot_poses, plans, num_steps_hit, total_odes, avg_time


# Used to evaluate Angle-space, Step-space, Heuristic, and LSTM-Guided Planners
def testSearchPlannerOnMatrix(robot, planner, step_controller, test_matrix, time_to_replan, friction, tstep = 0.01, prints = True):
  pct_success = 0
  mstf = 0  # mean steps to failure
  mdtf = 0  # mean distance to failure

  terrain_arrays = test_matrix.arrays
  terrain_functions = test_matrix.getFunctions()
  initial_apexes = test_matrix.apexes
  success_matrix = np.zeros((len(terrain_arrays), len(initial_apexes)))
  failure_counts = {k:0 for k in hopper.sim_codes_rev.keys()}
  count = 0
  num_ode_calls = 0
  tot_avg_time = 0
  for i in range(len(terrain_arrays)):
    for j in range(len(initial_apexes)):
      count += 1
      if prints:
        print("TERRAIN", i, "APEX", j)
      code, body_poses, foot_poses, plans, num_steps_hit, total_odes, avg_time = simulateRecedingHorizonAStar(robot,
                                                                                                    initial_apexes[j],
                                                                                                    planner,
                                                                                                    step_controller,
                                                                                                    terrain_functions[i],
                                                                                                    time_to_replan = time_to_replan,
                                                                                                    friction = friction,
                                                                                                    tstep = tstep,
                                                                                                    prints = prints)
      tot_avg_time += avg_time/(len(terrain_arrays) * len(initial_apexes))
      if code == 0:
        pct_success += 1
        success_matrix[i][j] = 1
        num_ode_calls += total_odes  # only count in the case of successes?
      else:
        success_matrix[i][j] = code
      failure_counts[code] += 1
      mstf += num_steps_hit
      if len(body_poses) > 0:
        mdtf += body_poses[-1][0]
      else:
        mdtf += 0
  pct_success /= (len(terrain_arrays) * len(initial_apexes))
  mstf /= (len(terrain_arrays) * len(initial_apexes))
  mdtf /= (len(terrain_arrays) * len(initial_apexes))
  test_results = test_utils.TestMetrics(pct_success, mstf, mdtf, success_matrix, failure_counts, num_ode_calls, tot_avg_time)
  return test_results


def simulateRecedingHorizonRNN(robot, 
                              x0_apex,
                              rnn_planner,
                              step_controller,
                              terrain_func,
                              friction,
                              time_to_replan = 3,
                              tstep = 0.01,
                              prints = True):
  goal_x = 10
  body_poses = []
  foot_poses = []
  plans = []
  t = 0
  n = 5

  # Unused
  Ts = 0.17
  Tf = 0.60

  cur_x = x0_apex[0]
  pos = np.arange(0, 8.0, 0.1)
  t_array = []
  for p in pos:
    t_array.append(terrain_func(p))
    
  initial_flight_state = x0_apex
  num_steps_hit = 0
  ttr = time_to_replan
  replan = []
  # control phase
  step_locs = []
  all_times = []
  while cur_x < goal_x and num_steps_hit < 20:
    code, flight_states, t_d = robot.simulateOneFlightPhaseODE(None,
                                                          tstep = tstep,
                                                          x_init = initial_flight_state,
                                                          u = np.pi/2,
                                                          debug=True,
                                                          terrain_func = terrain_func,
                                                          till_apex = True)
    

    flight_body_poses = [[x[0], x[1]] for x in flight_states]
    flight_foot_poses = [robot.getFootXYInFlight(x) for x in flight_states]
    plans += list([replan]) * len(flight_body_poses)
    t += t_d

    body_poses += flight_body_poses
    foot_poses += flight_foot_poses

    if code < 0:
      print("quitting simulation in flight pre-apex...")
      break

    # now this is post-apex, where we re-plan.
    apex_state = flight_states[-1]
    if prints:
      print("Apex height:", apex_state[1], "; Apex Velocity", apex_state[2])
    cur_x = apex_state[0]
    cur_y = apex_state[1]
    xdot = apex_state[2]

    pos = np.arange(0, 8.0, 0.1)
    t_array = []
    for p in pos:
      t_array.append(terrain_func(cur_x + p))

    # Planning Loop
    time_till_ground = 2 * (cur_y - robot.constants.L - terrain_func(cur_x))/(-robot.constants.g)
    xstep_pred = xdot * time_till_ground


    if ttr == time_to_replan:
      planning_apex = [0, apex_state[1], apex_state[2]]
      n = 5
      start_time = time.time()
      replan, softmaxes = rnn_planner.predict(n, planning_apex, t_array, xstep_pred)
      replan = replan + cur_x
      plan_time = time.time()
      all_times.append(plan_time - start_time)
      if prints:
        print(list(replan))
        print("for apex", [cur_x, apex_state[1], apex_state[2]])
        print("time to plan:", plan_time - start_time)
      Lstep = replan[0] - cur_x
      ttr = 1
    else:
      Lstep = replan[ttr] - cur_x
      ttr += 1

    leg_angle_des = step_controller.calcAngle(Lstep, xdot, Tf, Ts, upper_lim = 2.1)
    if prints:
      print("leg angle des = ", truncate(leg_angle_des, 2), "For LStep =", Lstep)

    code, flight_states, t_d = robot.simulateOneFlightPhaseODE(None,
                                                      tstep = tstep,
                                                      x_init = apex_state,
                                                      u = leg_angle_des,
                                                      debug = True,
                                                      terrain_func = terrain_func,
                                                      hit_apex = True)
    flight_body_poses = [[x[0], x[1]] for x in flight_states]
    flight_foot_poses = [robot.getFootXYInFlight(x) for x in flight_states]
    body_poses += flight_body_poses
    foot_poses += flight_foot_poses
    plans += list([replan]) * len(flight_body_poses)

    if code < 0:
      print("Failure in post-apex flight")
      break

    last_flight_state = flight_states[len(flight_states) - 1]
    t += t_d

    # print("flight time", t_d)
    stance_foot_pose = robot.getFootXYInFlight(last_flight_state)
    if prints:
      print("step loc=", stance_foot_pose)
    code, stance_states, t_d = robot.simulateOneStancePhase(last_flight_state,
                                                          tstep=tstep,
                                                          terrain_func = terrain_func,
                                                         terrain_normal_func = lambda x: np.pi/2, 
                                                         friction = friction)
    stance_body_poses = [robot.getBodyXYInStance(x, stance_foot_pose) for x in stance_states]
    stance_foot_poses = [stance_foot_pose for x in stance_states]
    body_poses += stance_body_poses
    foot_poses += stance_foot_poses

    plans += list([replan]) * len(stance_body_poses)
    if code < 0:
      print("Quitting simulation in stance...")
      break
    initial_flight_state = robot.stanceToFlight(stance_states[len(stance_states) - 1], stance_foot_pose)
    Ts = t_d  # estimate the stance time
    t += t_d
    num_steps_hit += 1
    # print("stance time", t_d)
    cur_x = initial_flight_state[0]
  if cur_x >= goal_x:
    success = True
  else:
    success = False
  return code, body_poses, foot_poses, plans, num_steps_hit, np.mean(all_times)


def testRNNPlannerOnMatrix(robot, rnn_planner, step_controller, test_matrix, time_to_replan, friction, tstep = 0.01, prints = True):
  pct_success = 0
  mstf = 0  # mean steps to failure
  mdtf = 0  # mean distance to failure

  terrain_arrays = test_matrix.arrays
  terrain_functions = test_matrix.getFunctions()
  initial_apexes = test_matrix.apexes
  success_matrix = np.zeros((len(terrain_arrays), len(initial_apexes)))
  failure_counts = {k:0 for k in hopper.sim_codes_rev.keys()}
  count = 0
  for i in range(len(terrain_arrays)):
    for j in range(len(initial_apexes)):
      count += 1
      if prints:
        print("TEST", count)
      first_step = initial_apexes[j][2]/3
      code, body_poses, foot_poses, plans, num_steps_hit, avg_time = simulateRecedingHorizonRNN(robot,
                                                                                      initial_apexes[j],
                                                                                      rnn_planner,
                                                                                      step_controller,
                                                                                      terrain_functions[i],
                                                                                      friction,
                                                                                      time_to_replan = time_to_replan,
                                                                                      tstep = tstep,
                                                                                      prints = prints)  
      if code == 0:
        pct_success += 1
        success_matrix[i][j] = 1
      else:
        success_matrix[i][j] = code
      failure_counts[code] += 1
      mstf += num_steps_hit
      if len(body_poses) > 0:
        mdtf += body_poses[-1][0]
      else:
        mdtf += 0
  pct_success /= (len(terrain_arrays) * len(initial_apexes))
  mstf /= (len(terrain_arrays) * len(initial_apexes))
  mdtf /= (len(terrain_arrays) * len(initial_apexes))
  test_results = test_utils.TestMetrics(pct_success, mstf, mdtf, success_matrix, failure_counts, avg_time = avg_time)
  return test_results
