import multiprocessing as mp
import numpy as np
import argparse
import os

import hopper
import hopper2d
import towr_interface
import dataset_generation

# Set generate_heightmaps to False if they have
# already been generated.
def generate_gap_data_Cassie(dataset_params, generate_heightmaps=True, num_procs = 20):
    hmap_info = dataset_params["hmap_info"]
    num_heightmaps = dataset_params["num_heightmaps"]
    num_sequences = dataset_params["num_sequences"]
    hmap_folder = dataset_params["terrains_folder"]
    max_num_gaps = dataset_params["max_num_gaps"]

    num_maps_made = towr_interface.generateGapMaps(num_heightmaps, max_num_gaps, hmap_info, hmap_folder)
    all_sequences, all_states, all_terrains = towr_interface.generateFullDataset(num_maps_made, num_sequences, random_param_names, dataset_params, hmap_folder, num_procs)
    return all_sequences, all_states, all_terrains


def generate_data_SLIP(dataset_params):
    return


# TODO: rework generateRandomSequences2D to use new heightmap class...
def generate_data_SLIP2D(dataset_params, num_proc):
    # create the robot model
    robot = hopper2d.Hopper2D(hopper.Constants)
    # create the desired cost function
    def cost_function(node):
        return

    def gen_args(seed):
        return (robot,
                dataset_params["num_heightmaps_per_thread"],
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

    return all_sequences, all_initial_states, all_terrains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to dataset config file", required = True)
    parser.add_argument("--robot", type=str, help="slip, 2dslip, or cassie", required = True)
    parser.add_argument("--num_procs", type=int, help="number of processors to use", default = 10)
    args = parser.parse_args()
    config_fname = args.config
    robot = args.robot
    num_procs = args.num_procs

    with open(config_fname, 'r') as f:
        config = yaml.safe_load(f)
    
    if robot == "slip":
        print("SLIP dataset generation not yet added to this script. Goodbye!")
    if robot == "2dslip":
        all_sequences, all_states, all_terrains = generate_data_2DSLIP(config, num_procs)
    if robot == "cassie":
        all_sequences, all_states, all_terrains = generate_gap_data_Cassie(config, True, num_procs)

    dataset_dir = os.path.join("datasets/", os.path.split(config_fname)[1][:-4] + "_" + robot)
    os.mkdir(dataset_dir)
    np.save(os.path.join(dataset_dir, "all_sequences.npy"), all_sequences)
    np.save(os.path.join(dataset_dir, "all_states.npy"), all_states)
    np.save(os.path.join(dataset_dir, "all_terrains.npy"), all_terrains)
    shutil.copyfile(config_fname, os.path.join(dataset_dir, "config.yaml"))

if __name__ == "__main__":
    main()
