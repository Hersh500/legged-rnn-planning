# import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import argparse
import os
import yaml
import shutil

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
    raise NotImplementedError("Using this script for 1D SLIP hasn't been implemented yet")


def generate_data_2DSLIP(dataset_params):
    # create the robot model
    robot = hopper2d.Hopper2D(hopper.Constants(dataset_params["robot_params"]))
    cparams = dataset_params["cost_fn"]
    x_pen = cparams["x_dist"]
    y_pen = cparams["y_dist"]
    xdot_pen = cparams["xdot"]
    ydot_pen = cparams["ydot"]
    xacc = cparams["x_acc"]
    yacc = cparams["y_acc"]
    spread_pen = cparams["spread"]

    def cost_function(x_flight, neighbors, prev_flight, goal, p):
        x_pos = x_flight[0]
        y_pos = x_flight[1]

        x_vel = x_flight[3]
        y_vel = x_flight[4]

        last_x_vel = prev_flight[3]
        last_y_vel = prev_flight[4]

        spread = 0
        for n in range(len(neighbors)):
            x_dist = (neighbors[n][0] - x_pos)**2
            y_dist = (neighbors[n][1] - y_pos)**2
            spread += np.sqrt(x_dist + y_dist)/len(neighbors)

        # downweight y acceleration cost
        acc_term = xacc * np.sqrt((last_x_vel - x_vel)**2) + yacc * np.sqrt((last_y_vel - y_vel)**2)
        # should I use euclidean or manhattan?
        pos_term = np.sqrt(x_pen * np.abs(x_pos - goal[0])**2 + y_pen * np.abs(y_pos - goal[1])**2)
        vel_term = xdot_pen * np.abs(x_vel - goal[2]) + ydot_pen * np.abs(y_vel - goal[3])
        spread_term = spread_pen * spread

        return pos_term + vel_term + acc_term + spread_term

    f = lambda x: dataset_generation.generateSLIPSequences2D(*x)

    def gen_args(seed):
        return (robot,
                dataset_params["num_heightmaps_per_thread"],
                dataset_params["num_states"],
                dataset_params["num_seqs_per"],
                dataset_params["min_num_steps"],
                dataset_params["hmap_info"],
                dataset_params["init_state_lims"],
                dataset_params["goal"],
                cost_function,
                False,
                False,
                dataset_params["min_progress_per_step"],
                seed)

    all_args = [gen_args(seed) for seed in np.random.choice(1000, dataset_params["num_proc"], replace=False)]
    p = Pool(processes=dataset_params["num_proc"])
    return_val_array = p.map(f, all_args)
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
    args = parser.parse_args()
    config_fname = args.config
    robot = args.robot

    with open(config_fname, 'r') as f:
        config = yaml.safe_load(f)
    
    if robot == "slip":
        print("SLIP dataset generation not yet added to this script. Goodbye!")
    if robot == "2dslip":
        all_sequences, all_states, all_terrains = generate_data_2DSLIP(config)
    if robot == "cassie":
        all_sequences, all_states, all_terrains = generate_gap_data_Cassie(config, True, num_procs)

    dataset_dir = os.path.join("datasets/", os.path.split(config_fname)[1][:-5] + "_" + robot)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    np.save(os.path.join(dataset_dir, "all_sequences.npy"), all_sequences)
    np.save(os.path.join(dataset_dir, "all_states.npy"), all_states)
    np.save(os.path.join(dataset_dir, "all_terrains.npy"), all_terrains)
    shutil.copyfile(config_fname, os.path.join(dataset_dir, "config.yaml"))

if __name__ == "__main__":
    main()
