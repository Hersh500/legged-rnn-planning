import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import random


### PLOTTING UTILITIES ###
'''
Animates the robot motions with a scrolling x axis.
plans is a (len(body_poses), 3) shape array
'''
def animateMovingXAxis(body_poses, foot_poses, anim_name = "animation.mp4", plans = None, nogo = None, terrain_func = lambda x:0, fps = 100):
    if len(body_poses) != len(foot_poses):
        print("Error: body poses array is not the same length as foot poses array")
        return

    fig, ax = plt.subplots()

    def animFunc(i):
        ax.clear()
        ax.set_ylim(-1, 1.5)
        x_minus = body_poses[i][0] - 1.0
        x_plus = body_poses[i][0] + 6.0
        # ax.set_xlim(-0.5, 8)
        ax.set_xlim(x_minus, x_plus)
        ax.scatter(body_poses[i][0], body_poses[i][1], s=500, color="blue")
        ax.scatter(foot_poses[i][0], foot_poses[i][1], s=100, color="red")
        ax.plot([foot_poses[i][0], body_poses[i][0]], [foot_poses[i][1], body_poses[i][1]], color='blue')
        if plans is not None:
            ax.scatter(plans[i], [0 for p in plans[i]])
        if nogo is not None:
            for point in nogo:
                ax.plot(point, [0, 0], color='red')
        plot_terrain(ax, x_minus=-0.9, x_plus = x_plus, terrain_func = terrain_func)
        return

    sim = animation.FuncAnimation(fig, animFunc, frames=range(len(body_poses)))
    sim.save(filename = anim_name, fps = fps, dpi = 300)
    return


def plotTerrainArrayAndSequence(terrain_array, sequence):
    fig = plt.figure()
    ax = fig.gca()
    xs = np.arange(0, 80, 1)
    for i in range(1, len(xs)):
        ax.plot([xs[i-1]/10, xs[i]/10], [terrain_array[xs[i-1]], terrain_array[xs[i]]], color = "black")
    for step in sequence:
        plt.scatter(step, 0, color = "blue")
    return


def plot_terrain(ax, x_minus, x_plus, terrain_func, fill = False):
    segments = np.arange(start = x_minus, stop = x_plus + 0.1, step = 0.1)
    segments[segments.size - 1] = min(segments[segments.size - 1], x_plus)
    for i in range(0, segments.size - 1):
        ax.plot([segments[i], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i])], color = "gray")
        ax.plot([segments[i+1], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i+1])], color = "gray")
        if fill:
            ax.fill_between([-5, 0], [0, 0], -1, facecolor = "gray")
            ax.fill_between([segments[i], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i])], -1, facecolor = "gray")
            ax.fill_between([segments[i+1], segments[i+1]], [terrain_func(segments[i]), terrain_func(segments[i+1])], -1, facecolor = "gray")
    return


def plotRobotWithArrow(ax, initial_apex, no_arrow = False, foot_pos = None, color="blue"):
    ax.scatter(initial_apex[0], initial_apex[1], s=500, color = color)
    if foot_pos is None:
        foot_pos = (initial_apex[0], initial_apex[1] - 0.5)
    ax.scatter(foot_pos[0], foot_pos[1], s = 100, color = color)
    ax.plot([initial_apex[0], foot_pos[0]], [initial_apex[1], foot_pos[1]], color=color)
      
    # plot an arrow representing the robot's forward velocity
    if not no_arrow:
        ax.arrow(initial_apex[0], initial_apex[1], 0.5 * initial_apex[2], 0, head_width = 0.1, color = "black")
    return


def plotStepsAndRobotOverTerrainArray(ax, initial_apex, prev_steps, terrain_array, color = "red", plot_text = True):
  plotRobotWithArrow(ax, initial_apex)
  def terrain_func(x):
    x_disc = int(x * 10)
    return terrain_array[x_disc]

  plot_terrain(ax, 0, 7.9, terrain_func, fill = True)
  ys = [terrain_func(step) for step in prev_steps]
  ax.scatter(prev_steps, ys, color=color)
  if plot_text:
    for i, step in enumerate(prev_steps):
      ax.text(step, terrain_func(step) + 0.05, str(i+1))
  return


def plotStepsOverTerrainArray(prev_steps, terrain_array, color = "red"):
  def terrain_func(x):
    x_disc = int(x * 10)
    return terrain_array[x_disc]

  fig = plt.figure()
  ax = plt.gca()
  plot_terrain(ax, 0, 7.9, terrain_func)
  ys = [terrain_func(step) for step in prev_steps]
  ax.scatter(prev_steps, ys, color=color)
  for i, step in enumerate(prev_steps):
    ax.text(step, terrain_func(step) + 0.05, str(i+1))
  plt.show()


def plotManyProbabilitiesOverTerrain(prev_steps,
                                 initial_apex,
                                 softmax_probs,
                                 terrain_func,
                                 colors,
                                 title_text = None,
                                 fname = None,
                                 outside_axis= None,
                                 no_arrow = False):
  if outside_axis is None:
    fig = plt.figure()
    ax = plt.gca()
  else:
    ax = outside_axis

  plot_terrain(ax, -3, 7.9, terrain_func, True)
  pos_array = np.arange(-3, 8.0, 0.1)
  ax.set_xlim(-1, 8)
  ax.set_ylim(-0.75, 1.5)

  # plot the distribution
  for s, c in zip(softmax_probs, colors):
    s_normed = s/(np.max(s))
    ax.plot(pos_array, s_normed, color = c)

  plotRobotWithArrow(ax, initial_apex, no_arrow = no_arrow)
  '''
  # plot the robot
  ax.scatter(initial_apex[0], initial_apex[1], s=500, color = "blue")
  ax.scatter(initial_apex[0], initial_apex[1] - 0.5, s = 100, color = "blue")
  ax.plot([initial_apex[0], initial_apex[0]], [initial_apex[1], initial_apex[1] - 0.5], color = "blue")
  '''

  # plot the previous steps
  ys = [terrain_func(step) for step in prev_steps]
  ax.scatter(prev_steps, ys, color="red")
  for i, step in enumerate(prev_steps):
    ax.text(step, terrain_func(step) + 0.05, str(i+1))
  if title_text is not None:
    ax.set_title(title_text)
  ax.set_xlabel("x(m)")
  ax.set_ylabel("y(m)")
  if fname is not None:
    plt.savefig(fname)
  if outside_axis is None:
    plt.show()


def plotProbabilitiesOverTerrain(prev_steps,
                                 initial_apex,
                                 softmax_probs,
                                 terrain_func,
                                 title_text = None):
  fig = plt.figure()
  ax = plt.gca()
  plot_terrain(ax, -3, 7.9, terrain_func)
  softmax_normed = softmax_probs/(np.max(softmax_probs))
  pos_array = np.arange(-3, 8.0, 0.1)

  # plot the distribution
  ax.plot(pos_array, softmax_normed, color = "green")

  # plot the robot
  ax.scatter(initial_apex[0], initial_apex[1], s=500, color = "blue")
  ax.scatter(initial_apex[0], initial_apex[1] - 0.5, s = 100, color = "blue")
  ax.plot([initial_apex[0], initial_apex[0]], [initial_apex[1], initial_apex[1] - 0.5], color = "blue")

  # plot the previous steps
  ys = [terrain_func(step) for step in prev_steps]
  ax.scatter(prev_steps, ys, color="red")
  for i, step in enumerate(prev_steps):
    ax.text(step, terrain_func(step) + 0.05, str(i+1))
  if title_text is not None:
    ax.set_title(title_text)
  plt.show()


def plotHiddens(hidden_state):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(hidden_state)


### DATA UTILITIES ###
def stepVectorToMatrix(step_vector, rows, cols):
    return step_vector.reshape((rows, cols))

# returns x,y in meters
def softmaxMatrixToXY(matrix, max_x, max_y, disc):
    vec = matrix.reshape((1, -1))
    idx = np.argmax(vec)
    num_cols = int(max_x/disc)
    x = idx%num_cols * disc
    y = idx//num_cols * disc
    return [x, y]


def oneHotEncodeSequences2D(sequences, max_x, max_y, disc):
    output = []
    for sequence in sequences:
        seq_oh = []
        for pos in sequence:
            arr = np.zeros((int(max_y/disc), int(max_x/disc)))
            x = np.clip(int(pos[0]/disc), 0, int(max_x/disc) - 1)
            y = np.clip(int(pos[1]/disc), 0, int(max_y/disc) - 1)
            arr[y][x] = 1
            seq_oh.append(arr.tolist())
        output.append(seq_oh)
    return output


# sequences is a (n x sequence_length array)
# outputs a (n_seq x seq_len x pos_len) list 
def oneHotEncodeSequences(sequences, max_lim = 8, min_lim = -3, disc = 0.1):
  output = []
  for sequence in sequences:
    lst = []
    for pos in sequence:
      pos_lst = [0] * int((max_lim - min_lim) / disc)
      if pos < min_lim:
        pos = min_lim - disc
      if pos > max_lim:
        pos = max_lim - disc
      disc_pos =  int((pos - min_lim)/disc)
      pos_lst[disc_pos] = 1
      lst.append(pos_lst)
    output.append(lst)
  return output


def concatenateTerrainsAndOHSteps(initial_states, sequences_oh):
  new_sequences = []
  new_initial_states = []
  for i in range(len(initial_states)):
    sequence = []
    terrain =  initial_states[i][:80]
    for seq in sequences_oh[i]:
      sequence.append(list(terrain) + list(seq))
    new_sequences.append(sequence)
    new_initial_states.append(initial_states[i][80:])
  return new_initial_states, new_sequences


def softmaxToStep(probs_array, min_lim = -3, disc = 0.1):
  argmax = torch.argmax(probs_array, dim = 2)
  return argmax * disc + min_lim


# Uses the mode of the distribution
# Uses torch tensors
def softmaxToOH(softmax_torch):
  oh_tensor = torch.zeros(softmax_torch.size())
  am = torch.argmax(softmax_np, dim = 2)
  oh_tensor[0][0][am] = 1
  return oh_tensor


def batch_sequence_data2D(initial_states, terrains, sequences, batch_size = 64):
    sequence_len_map = {}
    for index, sequence in enumerate(sequences):
        if len(sequence) in sequence_len_map.keys():
            sequence_len_map[len(sequence)].append(index)
        else:
            sequence_len_map[len(sequence)] = [index]
    sequence_batches = []
    initial_states_batches = []
    terrains_batches = []
    for seq_len in sequence_len_map.keys():
        indices = sequence_len_map[seq_len]
        count = 0
        while count < len(indices):
            end_idx = min(len(indices), count + batch_size)
            seq_batch = []
            iv_batch = []
            t_batch = []
            for k in range(count, end_idx):
                seq_batch.append(sequences[indices[k]])
                iv_batch.append(initial_states[indices[k]])
                t_batch.append(terrains[indices[k]])
            sequence_batches.append(seq_batch)
            initial_states_batches.append(iv_batch)
            terrains_batches.append(t_batch)
            count = end_idx
    return sequence_batches, terrains_batches, initial_states_batches


def batch_sequence_data(initial_values, sequences, batch_size = 64):
  sequence_len_map = {}
  for index, sequence in enumerate(sequences):
    if len(sequence) in sequence_len_map.keys():
      sequence_len_map[len(sequence)].append(index)
    else:
      sequence_len_map[len(sequence)] = [index]
  sequence_batches = []
  initial_values_batches = []
  for seq_len in sequence_len_map.keys():
    indices = sequence_len_map[seq_len]
    count = 0
    while count < len(indices):
      end_idx = min(len(indices), count + batch_size)
      seq_batch = []
      iv_batch = []
      for k in range(count, end_idx):
        seq_batch.append(sequences[indices[k]])
        iv_batch.append(initial_values[indices[k]])
      sequence_batches.append(seq_batch)
      initial_values_batches.append(iv_batch)
      count = end_idx
  return sequence_batches, initial_values_batches


def shuffleLists(list1, list2, list3 = None):
  if list3 is not None:
    c = list(zip(list1, list2, list3))
    random.shuffle(c)
    list1_shuf, list2_shuf, list3_shuf = zip(*c)
    return list1_shuf, list2_shuf, list3_shuf
  else:
    c = list(zip(list1, list2))
    random.shuffle(c)
    list1_shuf, list2_shuf = zip(*c)
    return list1_shuf, list2_shuf


def prepareBatches(seqs, init_vals):
  seqs = np.array([np.array(i, dtype = np.float32) for i in seqs])
  ivs = np.array([np.array(i, dtype = np.float32) for i in init_vals])

  input_seqs = seqs[:,:-1]
  target_seqs = seqs[:,1:]

  # input_seqs = np.reshape(input_seqs, (input_seqs.shape[0], input_seqs.shape[1], 1))
  # target_seqs = np.reshape(target_seqs, (target_seqs.shape[0], target_seqs.shape[1], 1))
  return input_seqs, target_seqs, ivs


def createDataBatches(inputs, init_states, batch_size = 64, train_pct = 0.9):
  input_batches, initial_states_batches = batch_sequence_data(init_states,
                                                              inputs,
                                                              batch_size = 64)
  
  num_batches = len(input_batches)
  seq_shuf, iv_shuf = shuffleLists(input_batches, initial_states_batches)
  train_seq_batches = seq_shuf[:int(train_pct * num_batches)]
  train_iv_batches = iv_shuf[:int(train_pct * num_batches)]

  test_seq_batches = seq_shuf[int(train_pct * num_batches):]
  test_iv_batches = iv_shuf[int(train_pct * num_batches):]

  return train_seq_batches, train_iv_batches, test_seq_batches, test_iv_batches


def prepareBatches2Dof(seqs, init_vals, terrains):
  seqs = np.array([np.array(i, dtype = np.float32) for i in seqs])
  ivs = np.array([np.array(i, dtype = np.float32) for i in init_vals])
  terrans = np.array([np.array(i, dtype = np.float32) for i in terrains])

  input_seqs = seqs[:,:-1]
  target_seqs = seqs[:,1:]

  # input_seqs = np.reshape(input_seqs, (input_seqs.shape[0], input_seqs.shape[1], 1))
  # target_seqs = np.reshape(target_seqs, (target_seqs.shape[0], target_seqs.shape[1], 1))
  return input_seqs, target_seqs, ivs, terrains


def createDataBatches2Dof(sequences, terrains, init_states, batch_size = 64, train_pct = 0.9):
  input_batches, terrain_batches, initial_states_batches = batch_sequence_data2D(init_states,
                                                                                 terrains,
                                                                                 sequences,
                                                                                 batch_size = 64)
  
  num_batches = len(input_batches)
  seq_shuf, iv_shuf, terrain_shuf = shuffleLists(input_batches, initial_states_batches, terrain_batches)
  train_seq_batches = seq_shuf[:int(train_pct * num_batches)]
  train_iv_batches = iv_shuf[:int(train_pct * num_batches)]
  train_terrain_batches = terrain_shuf[:int(train_pct * num_batches)]

  test_seq_batches = seq_shuf[int(train_pct * num_batches):]
  test_iv_batches = iv_shuf[int(train_pct * num_batches):]
  test_terrain_batches = terrain_shuf[int(train_pct * num_batches):]

  return train_seq_batches, train_terrain_batches, train_iv_batches, test_seq_batches, test_terrain_batches, test_iv_batches
