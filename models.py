import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils
'''
    Vanilla RNN that uses the terrain + initial apex to initialize the hidden state
    Input: previous step (as a one-hot vector)
    Output: next step (as a one-hot vector)
'''
class StepSequenceModel(nn.Module):
  def __init__(self, init_dim, input_size, output_size, hidden_dim, n_layers):
    super(StepSequenceModel, self).__init__()

    # Defining some parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    # Defining the layers
    # RNN Layer
    self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity = 'tanh')

    # initialization from terrain + apex to next step
    # enforcing the prior that the hidden state must be some function of the
    # terrain and apex
    self.init_fc1 = nn.Linear(init_dim, hidden_dim)
    self.init_fc2 = nn.Linear(hidden_dim, hidden_dim)

    # Fully connected layer
    self.fc1 = nn.Linear(hidden_dim, output_size)
    self.fc2 = nn.Linear(hidden_dim, output_size)

  def forward(self, x, init_data):
    batch_size = x.size(0)
    # if init_data.size(0) != x.size(0):
    #   print("Error!: initial states size does not match inputs size")
    # Initializing hidden state for first input using method defined below
    # hiddens = self.init_hidden(init_data)
    hiddens0 = self.init_hidden(init_data)
    hiddens = hiddens0
    for k in range(1, self.n_layers):
      hiddens = torch.cat((hiddens, hiddens0.detach().clone()), dim = 0)

    # hiddens = hiddens.view(1, hiddens.size(0), hiddens.size(1)) # for now only handle case with 1 layer
    # Passing in the input and hidden state into the model and obtaining outputs
    out, hidden = self.rnn(x, hiddens)
    
    # Reshaping the outputs such that it can be fit into the fully connected layer
    # out = out.contiguous().view(-1, self.hidden_dim)
    # out = self.fc2(F.tanh(self.fc1(out)))
    out = self.fc1(out)
    
    return out, hidden
    
  # init_data is (batch_size, init_dim)
  def init_hidden(self, init_data):
    # This method generates the first hidden state of zeros which we'll use in the forward pass
    # We'll send the tensor holding the hidden state to the device we specified earlier as well
    hiddens = self.init_fc2(F.tanh(self.init_fc1(init_data)))
    # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
    return hiddens



'''
    Same as the above, but uses LSTM instead of RNN
'''
class StepSequenceModelLSTM(nn.Module):
  def __init__(self, init_dim, input_size, output_size, hidden_dim, n_layers):
    super(StepSequenceModelLSTM, self).__init__()

    # Defining some parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    # Defining the layers
    # RNN Layer
    self.r_layer = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)

    # initialization from terrain + apex to next step
    # enforcing the prior that the hidden state must be some function of the
    # terrain and apex
    self.init_fc = nn.Linear(init_dim, hidden_dim)
    self.init_fc2 = nn.Linear(hidden_dim, hidden_dim)

    # Fully connected layer
    self.fc1 = nn.Linear(hidden_dim, output_size)
    self.fc2 = nn.Linear(init_dim, hidden_dim)

  def forward(self, x, init_data):
    batch_size = x.size(0)
    # if init_data.size(0) != x.size(0):
    #   print("Error!: initial states size does not match inputs size")
    # Initializing hidden state for first input using method defined below
    # hiddens = self.init_hidden(init_data)
    # if self.hidden_dim == init_data.size(2):
    if False:
      hiddens0 = init_data.detach().clone()
      cells0 = init_data.detach().clone()
      hiddens = hiddens0
      cells = cells0
    else:
      hiddens0, cells0 = self.init_hidden_and_cell(init_data)
      hiddens = hiddens0
      cells = cells0
    for k in range(1, self.n_layers):
      hiddens = torch.cat((hiddens, torch.zeros(hiddens0.size()).to(hiddens.device)), dim = 0)
      cells = torch.cat((cells, torch.zeros(cells0.size()).to(cells.device)), dim = 0)
      # cells = torch.cat((cells, cells0.detach().clone()), dim = 0)
      # hiddens = torch.cat((hiddens, hiddens0.detach().clone()), dim = 0)
    # hiddens = hiddens.view(1, hiddens.size(0), hiddens.size(1))
    # hiddens = hiddens.view(1, hiddens.size(0), hiddens.size(1)) # for now only handle case with 1 layer
    # Passing in the input and hidden state into the model and obtaining outputs
    out, hidden = self.r_layer(x, (hiddens, cells))
    
    # Reshaping the outputs such that it can be fit into the fully connected layer
    # out = out.contiguous().view(-1, self.hidden_dim)
    # out = self.fc2(F.tanh(self.fc1(out)))
    out = self.fc1(out)
    
    return out, hidden
    
  # init_data is (batch_size, init_dim)
  def init_hidden_and_cell(self, init_data):
    # This method generates the first hidden state of zeros which we'll use in the forward pass
    # We'll send the tensor holding the hidden state to the device we specified earlier as well
    hiddens = self.init_fc2(F.tanh(self.init_fc(init_data)))
    cells = self.init_fc2(F.tanh(self.init_fc(init_data)))
    # print(hiddens.size())
    # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
    return hiddens, cells


'''
    LSTM model that has convolutional layers on the inputs. Takes a stacked input of
    terrain + one hot step over the same discretized space.
    TODO: need to fully parameterize this, some values are currently hardcoded.
'''
class StepSequenceModelConv(nn.Module):
  def __init__(self, init_dim, input_size, output_size, hidden_dim, n_layers,
               use_lstm, ksize):
    super(StepSequenceModelConv, self).__init__()

    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.using_lstm = use_lstm
    self.input_size = input_size

    # not actually used in the convolution kernel definition
    dilation = 1
    padding = 0
    
    num_layers2 = 3
    l1_size = (input_size - 2 * padding - dilation * (ksize - 1) - 1) + 1
    l2_size = (l1_size - 2 * padding - dilation * (ksize - 1) -1) + 1
    linear_layer_input_size = l2_size * num_layers2
    
    if use_lstm:
      self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first = True)
    else:
      self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first = True, 
                        nonlinearity = "tanh")
    
    self.init_net = nn.Sequential(nn.Linear(init_dim, hidden_dim),
                                  nn.Tanh()
                                  )
    # the dimension of the output of this will be (1 x something...)
    self.conv_input = nn.Sequential(nn.Conv1d(2, 3, ksize),
                                    nn.ReLU(),
                                    nn.Conv1d(3, num_layers2, ksize))
    self.input_fc = nn.Linear(linear_layer_input_size, input_size)

    self.out_net = nn.Sequential(nn.Linear(hidden_dim, output_size))

  def forward(self, x, init):
    hiddens0 = self.init_net(init)
    cells0 = self.init_net(init)
    hiddens = hiddens0
    cells = cells0
    for k in range(1, self.n_layers):
      hiddens = torch.cat((hiddens, hiddens0.detach().clone()), dim = 0)
      cells = torch.cat((cells, cells0.detach().clone()), dim = 0)
    
    # this is a bigly hack.
    rnn_input = torch.zeros((x.size(0), x.size(1), self.input_size)).to(x.device)
    for i in range(0, x.size(1)):
      interm = self.conv_input(x[:,i,:,:]).view(x.size(0), -1)
      interm = interm.view(x.size(0), -1)
      interm = self.input_fc(interm)
      rnn_input[:,i,:] = interm
    if self.using_lstm:
      out, hidden = self.rnn(rnn_input, (hiddens, cells))
    else:
      out, hidden = self.rnn(rnn_input, hiddens)

    out = self.out_net(out)
    return out, hidden


# TODO: fix this model
class StepSequenceModelConv2D(nn.Module):
  def __init__(self, init_dim, input_size1, input_size2, output_size, hidden_dim, n_layers,
               use_lstm, ksize):
    super(StepSequenceModelConv2D, self).__init__()

    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.using_lstm = use_lstm
    self.input_size = input_size

    # not actually used in the convolution kernel definition
    dilation = 1
    padding = 0
    
    num_layers2 = 3
    l1_size = (input_size - 2 * padding - dilation * (ksize - 1) - 1) + 1
    l2_size = (l1_size - 2 * padding - dilation * (ksize - 1) -1) + 1
    linear_layer_input_size = l2_size * num_layers2
    
    if use_lstm:
      self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first = True)
    else:
      self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first = True, 
                        nonlinearity = "tanh")
    
    self.init_net = nn.Sequential(nn.Linear(init_dim, hidden_dim),
                                  nn.Tanh()
                                  )
    # the dimension of the output of this will be (1 x something...)
    self.conv_input = nn.Sequential(nn.Conv1d(2, 3, ksize),
                                    nn.ReLU(),
                                    nn.Conv1d(3, num_layers2, ksize))
    self.input_fc = nn.Linear(linear_layer_input_size, input_size)

    self.out_net = nn.Sequential(nn.Linear(hidden_dim, output_size))

  def forward(self, x, init):
    hiddens0 = self.init_net(init)
    cells0 = self.init_net(init)
    hiddens = hiddens0
    cells = cells0
    for k in range(1, self.n_layers):
      hiddens = torch.cat((hiddens, hiddens0.detach().clone()), dim = 0)
      cells = torch.cat((cells, cells0.detach().clone()), dim = 0)
    
    # this is a bigly hack.
    rnn_input = torch.zeros((x.size(0), x.size(1), self.input_size)).to(x.device)
    for i in range(0, x.size(1)):
      interm = self.conv_input(x[:,i,:,:]).view(x.size(0), -1)
      interm = interm.view(x.size(0), -1)
      interm = self.input_fc(interm)
      rnn_input[:,i,:] = interm
    if self.using_lstm:
      out, hidden = self.rnn(rnn_input, (hiddens, cells))
    else:
      out, hidden = self.rnn(rnn_input, hiddens)

    out = self.out_net(out)
    return out, hidden


### MODEL EVALUATION FUNCTIONS ###

# assume input_seq is of shape (batch_size, seq_len, 190)
# return is shaped (batch_size, seq_len, 2, 110)
def stackDataforConvNet(input_seqs, oh_dim = 110, forward_terrain_dim = 80):
  batch_size = input_seqs.shape[0]
  seq_len = input_seqs.shape[1]
  output = np.zeros((batch_size, seq_len, 2, oh_dim))
  output[:,:,1] = input_seqs[:,:,forward_terrain_dim:]
  output[:,:,0,oh_dim - forward_terrain_dim:] = input_seqs[:,:,:forward_terrain_dim]
  return output


'''
    Evaluate StepSequenceModelConv on a terrain + apex scenario
    TODO: beautify this code it looks disgusting
'''
def evaluateConvModel(model, n, initial_apex, first_step, terrain_list, device, T=1):
  datapoint = np.array(initial_apex[:3])
  init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(device)
  terrain_np = np.zeros((110))
  terrain_np[30:] = np.array(terrain_list)
  torch_terrain = torch.from_numpy(terrain_np).float().to(device)
  torch_terrain = torch_terrain.view(1, 1, -1)
  prev_steps_oh = utils.oneHotEncodeSequences(first_step)
  inp = np.zeros((len(prev_steps_oh[0]), len(terrain_list) + len(prev_steps_oh[0][0])))
  for i in range(len(prev_steps_oh[0])):
    inp[i] = np.array([list(terrain_list) + prev_steps_oh[0][i]])
  inp = np.reshape(inp, (1, len(prev_steps_oh[0]), -1))
  inp_reshaped = stackDataforConvNet(inp, oh_dim = 110, forward_terrain_dim= 80)
  input = torch.Tensor(inp_reshaped).to(device)
  outs = []
  hiddens = []
  softmaxes = []
  for i in range(n):
    out, hidden = model(input, init_state)
    out = out[:,-1].view(1, 1, -1)  # dividing by T is temperature scaling
    outs.append(utils.softmaxToStep(out)[0][0].item()) # out isnt' softmaxed..but that's okay for taking the argmax.
    hiddens.append(hidden)
    out_processed = F.softmax(out, dim = 2)
    softmaxes.append(F.softmax(out/T, dim = 2))
    out_and_terrain = torch.cat((torch_terrain, out_processed), dim = 1)
    out_and_terrain = out_and_terrain.view(1 ,1, 2, -1)
    input = torch.cat((input, out_and_terrain.float()), dim=1)
  return outs, softmaxes, hiddens


'''
    Evaluate a StepSequenceModel or StepSequenceModelLSTM that has been trained
    to use terrain+OH step as input (as opposed to using it to initialize the hidden state)
'''
def evaluateModelWithTerrainInput(model, n, initial_apex, first_step, terrain_list, device):
  datapoint = np.array(initial_apex[:3])
  init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(device)
  torch_terrain = torch.from_numpy(np.array(terrain_list)).float().to(device)
  torch_terrain = torch_terrain.view(1, 1, -1)
  prev_steps_oh = utils.oneHotEncodeSequences(first_step)
  inp = np.array([list(terrain_list) + prev_steps_oh[0][0]])
  input = torch.Tensor(inp).view(1, len(first_step[0]), -1).to(device)
  outs = []
  hiddens = []
  softmaxes = []
  for i in range(n):
    out, hidden = model(input, init_state)
    out = out[:,-1].view(1, 1, -1)
    outs.append(utils.softmaxToStep(out)[0][0].item())  # out isnt' softmaxed..but that's okay for taking the argmax. 
    hiddens.append(hidden)
    out_processed = F.softmax(out, dim = 2)
    # out_processed = out
    softmaxes.append(out_processed)
    out_and_terrain = torch.cat((torch_terrain, out_processed), dim = 2)
    input = torch.cat((input, out_and_terrain.float()), dim=1)
  return outs, softmaxes, hiddens



'''
    Evaluate StepSequenceModel or StepSequenceModelLSTM that has been trained to use
    terrain+apex to initialize the hidden state, and takes in only the OH previous step as input.
'''
def evaluateModelNSteps(model, n, initial_apex, first_step, terrain_list, device, ve_dim = 0):
  if ve_dim > 0:
    vel_encoded = [np.sin(i * initial_apex[2]) for i in range(ve_dim)]
    datapoint = np.array(list(terrain_list) + initial_apex[:-1] + vel_encoded)
  else:
    datapoint = np.array(list(terrain_list) + initial_apex)
  init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(device)

  prev_steps_oh = utils.oneHotEncodeSequences(first_step)
  input = torch.Tensor(np.array(prev_steps_oh)).view(1, len(first_step[0]), -1).to(device)
  outs = []
  hiddens = []
  softmaxes = []
  for i in range(n):
    out, hidden = model(input, init_state)
    out = out[:,-1].view(1, 1, -1)
    outs.append(utils.softmaxToStep(F.softmax(out, dim=2))[0][0].item())
    hiddens.append(hidden)
    out_processed = F.softmax(out, dim = 2)
    softmaxes.append(out_processed)
    input = torch.cat((input, out_processed.float()), dim=1)
  return outs, softmaxes, hiddens
