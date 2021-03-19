import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from conv_lstm import ConvLSTM

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


def convOutputSize(width, height, ksize, padding, stride):
    out_width = (width - ksize + 2 * padding)/stride + 1
    out_height = (height - ksize + 2 * padding)/stride + 1
    return int(out_width), int(out_height)


class StepSequenceModelConvLSTM2D(nn.Module):
    def __init__(self, init_dim, in_width, in_height, kernel_size, num_filter, device):
        super(StepSequenceModelConvLSTM2D, self).__init__()
        self.init_dim = init_dim
        self.in_width = in_width
        self.in_height = in_height
        self.kernel_size = kernel_size
        self.device = device
        self.num_filter = num_filter
        self.output_size = in_width * in_height
        self.b_h_w = (0, in_height//1, in_width//1)

        self.lstm = ConvLSTM(2, self.num_filter, self.b_h_w, self.kernel_size, self.device, padding = kernel_size//2, stride = 1)
        
        self.hidden_dim = self.b_h_w[1] * self.b_h_w[2] * self.num_filter
        self.init_net = nn.Sequential(nn.Linear(self.init_dim, self.hidden_dim))
                                      
        self.out_net = nn.Sequential(nn.Linear(self.hidden_dim, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, self.output_size))
        # self.out_net = nn.Sequential(nn.Conv2d(1, 1, 7, padding = 3))
        
        # General Architecture: give the stacked [terrain, step] as input to the lstm

    # x is of shape (batch_size, seq_len, 2, y, x)
    def forward(self, x, init):
        init_h = self.init_net(init) 
        init_c = self.init_net(init)
        # init_h = init_h.view(x.size(0), 1, x.size(3), x.size(4))
        # init_c = init_c.view(x.size(0), 1, x.size(3), x.size(4))
        init_h = init_h.view(x.size(0), 1, self.b_h_w[1], self.b_h_w[2])
        init_c = init_c.view(x.size(0), 1, self.b_h_w[1], self.b_h_w[2])
        outputs, h = self.lstm(inputs = x, states = (init_h, init_c), seq_len = x.size(1))
        # outputs is of the shape (seqs, batch_size, seq_len, y, x) ??
        # outputs = outputs.view(outputs.size(0) * outputs.size(1), 1, self.in_height, self.in_width)
        # outputs = self.out_net(outputs)
        outputs = outputs.view(x.size(0), x.size(1), self.output_size)
        # outputs = outputs.view(outputs.size(0), outputs.size(1), self.output_size)
        return self.out_net(outputs), h 
        # return outputs, h


class StepSequenceModelConv2D(nn.Module):
  def __init__(self, init_dim, in_width, in_height, input_size,
               hidden_dim, n_layers, ksize):
    super(StepSequenceModelConv2D, self).__init__()

    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.input_size = input_size
    self.output_size = in_width * in_height
    self.init_dim = init_dim

    # not actually used in the convolution kernel definition
    dilation = 1
    stride = 4
    padding = 0
    
    num_layers2 = 3
    out_width, out_height = convOutputSize(in_width, in_height, ksize, padding, 1)
    out_width, out_height = convOutputSize(out_width, out_height, ksize, padding, 1)

    linear_layer_input_size = out_width * out_height * num_layers2
    
    self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first = True)
    '''
    else:
      self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first = True, 
                        nonlinearity = "tanh")
    '''
    
    self.init_net = nn.Sequential(nn.Linear(self.init_dim, self.hidden_dim),
                                  nn.Tanh())

    # the dimension of the output of this will be (1 x something...)
    self.conv_input = nn.Sequential(nn.Conv1d(2, 3, ksize),
                                    nn.ReLU(),
                                    nn.Conv1d(3, num_layers2, ksize))
    self.input_fc = nn.Linear(linear_layer_input_size, self.input_size)
    self.out_net = nn.Sequential(nn.Linear(self.hidden_dim, self.output_size))

  # x is (batch_size, seq_len, 2, y, x)
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
      interm = self.conv_input(x[:,i,:,:,:]).view(x.size(0), -1)
      # interm = interm.view(x.size(0), -1)
      interm = self.input_fc(interm)
      rnn_input[:,i,:] = interm
    if self.using_lstm:
      out, hidden = self.rnn(rnn_input, (hiddens, cells))
    else:
      out, hidden = self.rnn(rnn_input, hiddens)

    out = self.out_net(out)
    return out, hidden


### MODEL EVALUATION FUNCTIONS ###
# sequences shape: (batch_size, seq_len, y, x)
# terrains shape: (batch_size, y, x)
# output shape: (batch_size, seq_len, 2, y, x)
def stackDataforConvNet2D(terrains, sequences):
    batch_size = sequences.shape[0]
    seq_len = sequences.shape[1]
    output = np.zeros((batch_size, seq_len, 2, sequences.shape[2], sequences.shape[3]))
    output[:,:,1,:,:] = sequences
    for i in range(0, seq_len):
        output[:,i,0,:,:] = terrains
    return output

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
    hiddens.append(hidden[-1])
    out_processed = F.softmax(out, dim = 2)
    softmaxes.append(F.softmax(out/T, dim = 2))
    out_and_terrain = torch.cat((torch_terrain, out_processed), dim = 1)
    out_and_terrain = out_and_terrain.view(1 ,1, 2, -1)
    input = torch.cat((input, out_and_terrain.float()), dim=1)
  return outs, softmaxes, hiddens



'''
    Evaluate StepSequenceModelConv on a terrain + apex scenario
    TODO: beautify this code it looks disgusting
    TODO: fix this
'''
def evaluateConvModel2D(model, n, initial_apex, first_step, terrain_array, device, T=1, max_x = 5, max_y = 5, disc = 0.1):
  model = model.eval()
  dim0 = terrain_array.shape[0]
  dim1 = terrain_array.shape[1]
  datapoint = np.array([initial_apex[3], initial_apex[4], initial_apex[2]])
  init_state = torch.FloatTensor(datapoint).view(1, 1, -1).to(device)
  prev_steps_oh = np.array(utils.oneHotEncodeSequences2D(first_step, max_x, max_y, disc))
  prev_steps_oh = prev_steps_oh.reshape(1, prev_steps_oh.shape[1], prev_steps_oh.shape[2], prev_steps_oh.shape[3])
  terrain_array = terrain_array.reshape(1, dim0, dim1)
  torch_terrain = torch.from_numpy(terrain_array).float().to(device).view(1, 1, dim0, dim1)
  inp = stackDataforConvNet2D(terrain_array, prev_steps_oh)
  input = torch.from_numpy(inp).float().to(device)
  outs = []
  hiddens = []
  softmaxes = []
  for i in range(n):
    out, hidden = model(input, init_state)
    out = out[-1, -1].view(1, 1, dim0, dim1)  # dividing by T is temperature scaling
    out_to_sm = out.view(dim0 * dim1)
    hiddens.append(hidden[-1])
    out_processed = F.softmax(out_to_sm)
    out_processed_t = F.softmax(out_to_sm/T)
    softmaxes.append(out_processed_t.detach().cpu().numpy().reshape(dim0, dim1))
    out_and_terrain = torch.cat((torch_terrain, out_processed.view(1, 1, dim0, dim1)), dim = 1)
    out_and_terrain = out_and_terrain.view(1 ,1, 2, dim0, dim1)
    input = torch.cat((input, out_and_terrain.float()), dim=1)
    out_mat = out.detach().cpu().numpy()[0][0]
    xy = utils.softmaxMatrixToXY(out_mat, max_x, max_y, disc)
    outs.append(xy)
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
