import multiprocessing as mp
import numpy as np

import hopper
import hopper2d
import towr_interface
import dataset_generation

def generate_data_Cassie():
    return

def generate_data_SLIP(dataset_params):
    
    return

def generate_data_SLIP2D(dataset_params, num_proc):
    # create the robot model
    robot = hopper2d.Hopper2D(hopper.Constants)
    # create the desired cost function
    def cost_function(node):
        return

    def gen_args(seed):
        return (robot,
                dataset_params["num_terrains"],
                dataset_params["num_states"],
                dataset_params["num_seqs_per"],
                dataset_params["min_num_steps"],
                dataset_params["friction"],
                cost_function,
                False,
                False,
                dataset_params["min_progress_per_step"],
                dataset_params["max_x"],
                dataset_params["max_y"],
                dataset_params["disc"], seed)

    all_args = [gen_args(seed) for seed in np.random.choice(1000, num_proc, replace=False)]
    p = mp.Pool(processes=num_proc)
    return_val_array = p.starmap(dataset_generation.generateRandomSequences2D, all_args)
    all_initial_states = []
    all_sequences = []
    all_terrains = []

    for r in return_val_array:
        initial_states = r[0]
        initial_terrains = r[1]
        sequences = r[2]
        for i in range(len(initial_states)):
            all_initial_states.append(initial_states[i])
            all_terrains.append(initial_terrains[i])
            all_sequences.append(sequences[i])
    # Should I just build the dataset in here?
    return all_sequences, all_initial_states, all_terrains