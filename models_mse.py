import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.utils import shuffle

import utils

# Use this for models that don't use CE loss
# TODO: is there a way to do this that also incorporates the idea of
# doing this "in-image"?
# TODO: need better file organization; this should be with the other models 

# Only need terrain encoder
# model should output two Reals: x,y. The error is the MSE loss between this and the expected.
class StepSequenceModel2D_MSE(nn.Module):
  def __init__(self, init_dim, in_width, in_height, terrain_dim,
               hidden_dim, n_layers, terrain_encoder, dropout_p = 0.3):
    super(StepSequenceModel2D_MSE, self).__init__()

    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # self.input_size = input_size
    self.input_size = 2 + terrain_dim
    self.terrain_dim = terrain_dim
    self.output_size = 2
    self.init_dim = init_dim
    self.dropout = nn.Dropout(dropout_p)

    self.rnn = nn.LSTM(self.input_size, hidden_dim, n_layers, batch_first = True)
    # self.init_net = nn.Sequential(nn.Linear(self.init_dim + self.terrain_dim, self.hidden_dim),
    #                               nn.Tanh(),
    #                               nn.Linear(self.hidden_dim, self.hidden_dim))
    
    self.init_net = nn.Sequential(nn.Linear(self.init_dim, self.hidden_dim),
                                  nn.Tanh(),
                                  nn.Linear(self.hidden_dim, self.hidden_dim))

    self.terrain_encoder = terrain_encoder.cpu()
    self.terrain_encoder.requires_grad = True
    # linear_layer_input_size = torch.numel(self.terrain_encoder.encode(torch.zeros(1, 1, in_height, in_width))) + 2
    # self.input_fc = nn.Linear(linear_layer_input_size, self.input_size)
    self.out_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.Sigmoid(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim//2),
                                 nn.Sigmoid(),
                                 nn.Linear(self.hidden_dim//2, self.output_size))

  # terrain is (batch_size, height, width)
  # sequence is (batch_size, seq_len, 2) (x, y) (normalized to 0-1)
  def forward(self, sequence, terrain, init):
    # t_encoded = self.terrain_encoder.encode(terrain.view(terrain.size(0), 1, terrain.size(1), terrain.size(2)))
    # init_and_t = torch.cat([init[0], t_encoded], dim = 1)
    # init_and_t = init_and_t.view(1, init_and_t.size(0), init_and_t.size(1))
    # hiddens0 = self.init_net(init_and_t)
    # cells0 = self.init_net(init_and_t)
    hiddens0 = self.init_net(init.view(init.size(0), init.size(1), init.size(2)))
    cells0 = self.init_net(init.view(init.size(0), init.size(1), init.size(2)))
    hiddens = hiddens0
    cells = cells0
    for k in range(1, self.n_layers):
      hiddens = torch.cat((hiddens, hiddens0.detach().clone()), dim = 0)
      cells = torch.cat((cells, cells0.detach().clone()), dim = 0)
    rnn_input = torch.zeros((sequence.size(0), sequence.size(1), self.input_size)).to(sequence.device)
    for i in range(0, sequence.size(1)):
      interm = self.terrain_encoder.encode(terrain.view(terrain.size(0), 1, terrain.size(1), terrain.size(2))).view(terrain.size(0), -1)
      interm = torch.cat([interm, sequence[:,i,:].view(sequence.size(0), 2)], dim = 1)
      # interm = self.input_fc(interm)
      rnn_input[:,i,:] = interm

    # rnn_input = sequence
    out, hidden = self.rnn(rnn_input, (hiddens, cells))
    out = self.dropout(out)
    out = self.out_net(out)
    return out, hidden
  
  
  # x is tensor of the shape (seq_len, 2, y, x)
  def view_conv_output(self, x):
    outputs = []
    for i in range(0, x.size(1)):
      outputs.append(self.conv_net(x[i,:,:,:]))
    return outputs


class CurriculumDataset2D(data.Dataset):
    def __init__(self, individual_datasets):
        self.datasets = individual_datasets
        # use active_set variable to set which of the 8 datasets we're using.
        self.active_set = 0
    
    def __len__(self):
        return len(self.datasets[self.active_set])
    
    def __getitem__(self, idx):
        return self.datasets[self.active_set][idx]

    def setActiveSet(self, x):
        self.active_set = x


class SequenceDataset2D_MSE(data.Dataset):
    def __init__(self, all_sequences, all_terrains, all_ivs, max_x, max_y, disc, batch_size = 64):
        self.seq_batches, self.terrain_batches, self.is_batches = utils.batch_sequence_data2D(all_ivs, all_terrains, all_sequences, batch_size)

        self.max_x = max_x
        self.max_y = max_y
        self.disc = disc

    def __len__(self):
        return len(self.seq_batches)

    def __getitem__(self, idx):
        # seq batch is currently of the shape shape [64, seq_len, 2]
        seq_batch_list = self.seq_batches[idx]
        # now seq_batch is [64, seq_len, 2]
        seq_batch = np.array(seq_batch_list, dtype=np.float32)
        seq_batch[:,:,0] /= self.max_x
        seq_batch[:,:,0] = np.clip(seq_batch[:,:,0], 0, 1)
        seq_batch[:,:,1] /= self.max_y
        seq_batch[:,:,1] = np.clip(seq_batch[:,:,1], 0, 1)

        iv_batch = np.array(self.is_batches[idx], dtype=np.float32)

        # terrain batch is [64, y, x]
        terrain_batch = np.array(self.terrain_batches[idx], dtype=np.float32)

        # make target_seqs
        input_seq_batch = seq_batch[:,:-1]
        target_seq_batch = seq_batch[:,1:]

        # need to stack terrains with the input sequences
        return input_seq_batch, target_seq_batch, terrain_batch, iv_batch


def trainConvRNN2D_Curr(model,
                        schedule,
                        lr, 
                        train_loaders,
                        device,
                        test_arguments,
                        teacher_force_ratio = 0.5):

  model = model.to(device)
  model = model.train()
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  success_per_epoch = []

  for k in range(len(schedule)):
    print("On terrain type", k)
    for epoch in range(schedule[k]):
      losses = []
      # iterating over batches
      for num, batch in enumerate(train_loaders[k]):
        optimizer.zero_grad()
        seq_batch = batch[0]
        target_seq_batch = batch[1]
        terrain_batch = batch[2]
        ivs = batch[3]

        torch_inputs = seq_batch.float().to(device)
        if torch_inputs.size(1) == 0:
          continue
        torch_targets = target_seq_batch.reshape(-1, 2).to(device)
        torch_terrains = terrain_batch.float().to(device)
        torch_ivs = ivs.float().to(device)
        torch_ivs = torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1))
        teacher_force = np.random.rand() < teacher_force_ratio
        if teacher_force: # use the full input sequence
          output, hidden = model(torch_inputs, torch_terrains, torch_ivs)
          output = output.to(device).view(-1, 2)
        else:
          batch_size = torch_inputs.size(0)
          input = torch_inputs[:,0].view(batch_size, 1, 2)
          for i in range(0, torch_inputs.size(1)):
            out, hidden = model(input, torch_terrains, torch_ivs)
            out_last = out[:,-1,:].view(batch_size, 1, 2)
            input = torch.cat((input, out_last.float()), dim=1)
          output = out.view(-1, 2)
        loss = criterion(output, torch_targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if num % 100 == 0:
          print("Epoch", epoch, "Batch ", num, "Loss:", loss.item())

    print('Epoch: {}/{}.............'.format(epoch, schedule[k]), end=' ')
    print("Loss: {:.4f}".format(np.mean(losses)))

    '''
    # evaluate the model on the test matrix
    if len(test_arguments) > 0:
      robot = test_arguments[0]
      step_controller = test_arguments[1]
      test_matrix = test_arguments[2]

      planner = ConvRNNPlanner2D_MSE(model, device, max_x = 8, max_y = 4)
      rnn_test_results = testRNNOnMatrix2D(robot, planner, step_controller, 
                                          test_matrix, time_to_replan = 2,
                                          friction = 0.8, tstep = 0.01, prints = False)
      success_per_epoch.append(rnn_test_results.percent_success)
      model = model.train()
    print("Success %:", rnn_test_results.percent_success)
    '''
  return model, success_per_epoch

# Also need a new training loop that uses MSE Loss
def trainConvRNN2DMSE(model,
                      n_epochs,
                      lr,
                      train_loader,
                      device,
                      test_arguments = [],
                      teacher_force_ratio = 0.5):
  model = model.to(device)
  model = model.train()
  criterion = nn.MSELoss()
  # criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  success_per_epoch = []

  for epoch in range(n_epochs):
    losses = []
    # iterating over batches
    for num, batch in enumerate(train_loader):
      optimizer.zero_grad()
      seq_batch = batch[0]
      target_seq_batch = batch[1]
      terrain_batch = batch[2]
      ivs = batch[3]

      torch_inputs = seq_batch.float().to(device)
      if torch_inputs.size(1) == 0:
        continue
      torch_targets = target_seq_batch.reshape(-1, 2).to(device)
      torch_terrains = terrain_batch.float().to(device)
      torch_ivs = ivs.float().to(device)
      torch_ivs = torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1))
      teacher_force = np.random.rand() < teacher_force_ratio
      if teacher_force: # use the full input sequence
        output, hidden = model(torch_inputs, torch_terrains, torch_ivs)
        output = output.to(device).view(-1, 2)
      else:
        batch_size = torch_inputs.size(0)
        input = torch_inputs[:,0].view(batch_size, 1, 2)
        for i in range(0, torch_inputs.size(1)):
          out, hidden = model(input, torch_terrains, torch_ivs)
          out_last = out[:,-1,:].view(batch_size, 1, 2)
          input = torch.cat((input, out_last.float()), dim=1)
        output = out.view(-1, 2)
      loss = criterion(output, torch_targets)
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
      if num % 100 == 0:
        print("Epoch", epoch, "Batch ", num, "Loss:", loss.item())
        # print(seq_batch[0:5,:])

    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(np.mean(losses)))

    '''
    # evaluate the model on the test matrix
    if len(test_arguments) > 0:
      robot = test_arguments[0]
      step_controller = test_arguments[1]
      test_matrix = test_arguments[2]

      planner = ConvRNNPlanner2D_MSE(model, device, max_x = 8, max_y = 4)
      rnn_test_results = testRNNOnMatrix2D(robot, planner, step_controller, 
                                          test_matrix, time_to_replan = 2,
                                          friction = 0.8, tstep = 0.01, prints = False)
      success_per_epoch.append(rnn_test_results.percent_success)
      model = model.train()
    print("Success %:", rnn_test_results.percent_success)
    '''
  return model, success_per_epoch


# Also need a new evaluateModel function
def evaluateConvModel2D_MSE(model, n, initial_apex, first_step,
                        terrain_array, device, max_x = 5,
                        max_y = 5):
  # model = model.eval()
  dim0 = terrain_array.shape[0]
  dim1 = terrain_array.shape[1]
  datapoint = np.array([initial_apex[3], initial_apex[4], initial_apex[2]])
  init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(device)
  terrain_array = terrain_array.reshape(1, dim0, dim1)
  torch_terrain = torch.from_numpy(terrain_array).float().to(device).view(1, dim0, dim1)
  input = torch.from_numpy(np.array(first_step)).float().to(device)
  input[:,:,0] = input[:,:,0]/max_x
  input[:,:,1] = input[:,:,1]/max_y
  outs = []
  hiddens = []
  softmaxes = []
  for i in range(n):
    out, hidden = model(input, torch_terrain, init_state)
    out_last = out[:,-1,:]
    temp = [out_last[0][0].item() * max_x, out_last[0][1].item() * max_y]
    outs.append(temp)
    hiddens.append(hidden[0][-1])
    input = torch.cat((input, out_last.view(1, 1, 2).float()), dim=1)
  return outs, hiddens


# Also need a new planner object.
class ConvRNNPlanner2D_MSE:
  def __init__(self, rnn_model, device, max_x = 5, max_y = 5, disc = 0.2):
    self.model = rnn_model
    self.device = device
    self.model = self.model.to(device)
    self.max_x = max_x
    self.max_y = max_y
    self.disc = disc

  def predict(self, n, initial_apex, terrain_matrix, first_steps):
    seq = first_steps 
    outs, hiddens = evaluateConvModel2D_MSE(self.model,
                                            n,
                                            initial_apex,
                                            seq, 
                                            terrain_matrix, 
                                            self.device,
                                            max_x = self.max_x,
                                            max_y = self.max_y)
    return outs, hiddens
