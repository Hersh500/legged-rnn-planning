# functions copied from ipynb
import utils
import models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data as data
import torchvision

# need to pass in dataset files, heightmap configs
class SequenceDataset2D(data.Dataset):
    def __init__(self, all_sequences, all_terrains, all_ivs, max_x, max_y, disc, batch_size = 64):
        self.seq_batches, self.terrain_batches, self.is_batches = utils.batch_sequence_data2D(all_ivs, all_terrains, all_sequences, batch_size)

        self.max_x = max_x
        self.max_y = max_y
        self.disc = disc
        self.state_dim = len(all_ivs[0][0])
        self.batch_size = batch_size

    def __len__(self):
        return len(self.seq_batches)

    def __getitem__(self, idx):
        # seq batch is currently of the shape shape [batch_size, seq_len, 2]
        seq_batch_list = self.seq_batches[idx]
        seq_batch_oh = utils.oneHotEncodeSequences2D(seq_batch_list, self.max_x, self.max_y, self.disc)
        # now seq_batch is [batch_size, seq_len, y, x]
        seq_batch = np.array(seq_batch_oh, dtype=np.float32)
        iv_batch = np.array(self.is_batches[idx], dtype=np.float32)

        # terrain batch is [batch_size, y, x]
        terrain_batch = np.array(self.terrain_batches[idx], dtype=np.float32)

        # make target_seqs
        input_seq_batch = seq_batch[:,:-1]
        target_seq_batch = seq_batch[:,1:]

        # reshape into one hot vector
        target_seq_batch = target_seq_batch.reshape((-1, int(self.max_x/self.disc * self.max_y/self.disc)))

        # need to stack terrains with the input sequences
        # this method should be in utils, probably
        stacked_batch = models.stackDataforConvNet2D(terrain_batch, input_seq_batch)
        return stacked_batch, target_seq_batch, iv_batch


def build_dataset(sequences_f, initial_states_f, terrains_f, dataset_config, batch_size):
    # load data from files
    all_initial_states = np.load(initial_states_f, allow_pickle=True)
    all_sequences = np.load(sequences_f, allow_pickle=True)
    all_terrains = np.load(terrains_f, allow_pickle=True)

    # assemble into pytorch dataset
    return SequenceDataset2D(all_sequences, all_terrains, all_initial_states,
                             dataset_config["max_x"], dataset_config["max_y"],
                             dataset_config["disc"], batch_size)


# ideally should also specify whether or not to pretrain the fusion net 
# should split this up into some sub functions, just for readability reasons
def train_2drnn_geomloss(model_params, dataset, device, model = None):
    width = int((dataset.max_x - dataset.min_x)/dataset.disc)
    height = int((dataset.max_y - dataset.min_y)/dataset.disc)
    if model is None:
        # TODO: load pretrained fusion net
        fusion_net = torch.load(model_params["fusion_net_path"])
        model = models.StepSequenceModelFusion2D(dataset.state_dim, width, height,
                                                 model_params["hidden_dim"],
                                                 model_params["n_layers"],
                                                 fusion_net,
                                                 model_params["fusion_net_dim"],
                                                 model_params["skip_for_output"])

    train_loader = data.DataLoader(dataset,
                                   batch_size = None,
                                   shuffle = True,
                                   collate_fn = None)

    model = model.to(device).train()
    # should the fusion net parameters be excluded from this? 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(model_params["n_epochs"]):
        losses = [] 
        for num, batch in enumerate(train_loader):
            optimizer.zero_grad() 
            input_data = batch[0]
            torch_targets = batch[1]
            ivs = batch[2]
            torch_inputs = input_data.float().to(device)
            if torch_inputs.size(1) == 0:
                continue
            torch_targets = torch_targets.to(device)
            torch_ivs = ivs.float().to(device)
            torch_ivs = torch_ivs.view(1, torch_ivs.size(0), torch_ivs.size(1))
            teacher_force = np.random.rand() < teacher_force_ratio
            if teacher_force: # use ground truth as inputs
                output, hidden = model(torch_inputs, torch_ivs)
                output = output.to(device).view(-1, height, width)
            else:
              batch_size = torch_inputs.size(0)
              input = torch_inputs[:,0].view(batch_size, 1, 2, height, width)
              torch_terrain = torch_inputs[:,0,0].view(batch_size, 1, height, width)
              for i in range(0, torch_inputs.size(1)):
                out, hidden = model(input, torch_ivs)
                out_last = out[:,-1].view(batch_size, 1, height, width)
                out_to_sm = out_last.view(batch_size, height * width)
                out_processed = F.softmax(out_to_sm, dim = 1)
                out_and_terrain = torch.cat((torch_terrain,
                                            out_processed.view(batch_size, 1, height, width)),
                                            dim = 1)
                out_and_terrain = out_and_terrain.view(batch_size, 1, 2, height, width)
                input = torch.cat((input, out_and_terrain.float()), dim=1)
              output = out.view(-1, height, width)
            output = output.view(-1, height*width)
            ent = -torch.sum(F.softmax(output, dim=1) * F.log_softmax(output, dim=1), dim = 1)
            output = F.softmax(output, dim = 1).view(-1, height, width)
            torch_targets = torch_targets.view(-1, height, width)
            # torch_targets = torchvision.transforms.GaussianBlur(3, sigma = 1).forward(torch_targets)
            # how did I calculate what size this was? was this from the gaussian blur?
            # was it to make the size the nearest power of two? -- that would make the most sense
            output = torch.nn.functional.pad(output, (12, 12, 22, 22)).view(-1, 1, 64, 64)
            torch_targets = torch.nn.functional.pad(torch_targets, (12, 12, 22, 22)).view(-1, 1, 64, 64)
            loss = sinkhorn_images.sinkhorn_divergence(output, torch_targets, scaling = 0.7)
            # TODO: how should this be tuned? What is the right conditioning for this?
            loss = loss - 1/5000 * ent
            loss = torch.sum(loss)/output.size(0)
            try:
              loss.backward()
            except RuntimeError:
              print("skipping batch!")
              continue
            optimizer.step()
            losses.append(loss.item())
            if num % 100 == 0:
              print("Epoch", epoch, "Batch ", num, "Loss:", loss.item())
          print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
          print("Loss: {:.4f}".format(np.mean(losses)))
 
    return model
