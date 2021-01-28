import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import models
import utils

def trainConvRNN(model,
                 n_epochs, 
                 lr, 
                 train_seq_batches,
                 train_iv_batches,
                 device):
  model = model.to(device)
  model = model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  
  for epoch in range(1, n_epochs):
    seq_shuf, iv_shuf = utils.shuffleLists(train_seq_batches, train_iv_batches)
    losses = []
    # iterating over batches
    for i in range(0, len(seq_shuf)):
      optimizer.zero_grad()
      seqs = seq_shuf[i]
      init_vals = iv_shuf[i]
      input_seqs, target_seqs, ivs = utils.prepareBatches(seqs, init_vals)
      input_data = models.stackDataforConvNet(input_seqs, oh_dim = 110)
      target_seqs = target_seqs[:,:,80:]
      torch_inputs = torch.from_numpy(input_data).float().to(device)
      if torch_inputs.size(1) == 0:
        continue
      torch_targets = torch.from_numpy(target_seqs).to(device)
      
      torch_ivs = torch.from_numpy(ivs).float().to(device)
      torch_ivs = torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1))
      output, hidden = model(torch_inputs, torch_ivs)
      output = output.view(-1)
      output = output.to(device).view(-1, 110)

      torch_targets = torch_targets.reshape(-1, 110)

      loss = criterion(output, torch.argmax(torch_targets, dim=1))
      loss.backward() 
      optimizer.step()
      losses.append(loss.item())
      if i % 100 == 0:
        print("Epoch", epoch, "Batch ", i, "Loss:", loss.item())
  
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(np.mean(losses)))

  return model


def trainConvRNN2Dof(model,
                    n_epochs, 
                    lr, 
                    train_seq_batches,
                    train_terrain_batches,
                    train_iv_batches,
                    device):
  model = model.to(device)
  model = model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  
  for epoch in range(1, n_epochs):
    seq_shuf, iv_shuf, terrain_shuf = utils.shuffleLists(train_seq_batches, train_iv_batches, train_terrain_batches)
    losses = []
    # iterating over batches
    for i in range(0, len(seq_shuf)):
      optimizer.zero_grad()
      seqs = seq_shuf[i]
      init_vals = iv_shuf[i]
      terrains = terrain_shuf[i]
      input_seqs, target_seqs, ivs, terrains = utils.prepareBatches2Dof(seqs, init_vals, terrains)
      input_data = models.stackDataforConvNet2D(terrains, input_seqs)

      # need to reshape target seqs from matrices to vectors
      target_seqs = target_seqs.reshape((target_seqs.shape[0], -1))
      torch_inputs = torch.from_numpy(input_data).float().to(device)
      if torch_inputs.size(1) == 0:
        continue
      torch_targets = torch.from_numpy(target_seqs).to(device)
      torch_ivs = torch.from_numpy(ivs).float().to(device)
      torch_ivs = torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1))
      output, hidden = model(torch_inputs, torch_ivs)
      output = output.view(-1)

      # need to reshape this
      output = output.to(device).view(-1, target_seqs.shape[1])

      # torch_targets = torch_targets.reshape(-1, 110)

      loss = criterion(output, torch.argmax(torch_targets, dim=1))
      loss.backward() 
      optimizer.step()
      losses.append(loss.item())
      if i % 100 == 0:
        print("Epoch", epoch, "Batch ", i, "Loss:", loss.item())
  
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(np.mean(losses)))

  return model


def convRNNValidation(model_eval, test_seq_batches, test_iv_batches, device):
    batch_size = len(test_seq_batches[0])
    num_batches = len(test_seq_batches)
    test_loss = 0
    for i, batch in enumerate(test_seq_batches):
        seqs = test_seq_batches[i]
        init_vals = test_iv_batches[i]
        input_seqs, target_seqs, ivs = utils.prepareBatches(seqs, init_vals)
        input_data = models.stackDataforConvNet(input_seqs, oh_dim = 110)
        target_seqs = target_seqs[:,:,80:]
        torch_inputs = torch.from_numpy(input_data).float().to(device)
        torch_targets = torch.from_numpy(target_seqs).to(device)
        torch_ivs = torch.from_numpy(ivs).to(device)
        if torch_ivs.size(0) != batch_size or torch_inputs.size(0) != batch_size or torch_inputs.size(1) == 0:
            continue
        output, hidden = model_eval(torch_inputs, torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1)))
        output = output.view(-1)
        torch_targets = torch_targets.reshape(-1)
        output = output.to(device)
        output = output.view(-1, 110)
        torch_targets = torch_targets.reshape(-1, 110).long()

        loss = criterion(output, torch.argmax(torch_targets, dim=1))
        avg_loss = loss.item()
        #print(avg_loss)
        test_loss += avg_loss/num_batches

    print("test loss = ", test_loss) 
