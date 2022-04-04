import numpy as np
import multiprocessing as mp
import subprocess as sp
import argparse
import yaml
import csv
import os
import shutil

import hopper2d
from heightmap import HeightMap, gapHeightMap, HeightMapFromCSV


def generateOptParams(lims, param_names):
    opt_params = {}
    for param in param_names:
        llim = lims[param]['min'] 
        hlim = lims[param]['max']
        if lims[param]["type"] == "int":
            opt_params[param] = np.random.randint(llim, hlim + 1) 
        else:
            opt_params[param] = np.random.uniform(llim, hlim)
    return opt_params


# this is the main function that will get called in the starmap
def runOneOptimization(config_lims, param_names, terrain_file, ofname, seed):
    np.random.seed(seed)
    opt_params = generateOptParams(config_lims, param_names)
    towr_path = config_lims["towr_path"]
    num_steps = opt_params["num_steps"]
    init_x_vel = opt_params["init_x_vel"]
    init_y_vel = opt_params["init_y_vel"]
    T = opt_params["T"]
    goal_x = opt_params["goal_x"]

    # Should add CoM height as a parameter as well.
    initial_state = [init_x_vel, init_y_vel]

    args = ["--height_csv=" + terrain_file,
            "--num_steps=" + str(num_steps),
            "--initial_x_vel=" + str(init_x_vel),
            "--initial_y_vel=" + str(init_y_vel),
            "--T=" + str(T),
            "--goal_x=" + str(goal_x),
            "--ofname=" + str(ofname)]

    # run this in a blocking way
    val = sp.call([towr_path] + args)
    step_sequence = []
    if val == 0:
        # once you get the results, read the output csv and return the sequence, terrain, initial state
        with open(ofname) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                step_sequence.append([float(row[0]), float(row[1])])
    terrain = heightmap.HeightMapFromCSV(terrain_file) 
    return step_sequence, terrain, initial_state


# generate some random dataset with some gaps.
# returns the number of heightmaps created.
def generateGapMaps(num_heightmaps, max_num_gaps, heightmap_info, path):
    if not os.path.exists(path):
        os.mkdir(path)
    num_each = num_heightmaps//(max_num_gaps+1)
    num_gaps = 0
    total_count = 0
    min_island_size = 0.5
    max_island_size = 1
    while num_gaps < max_num_gaps+1:
        for i in range(num_each):
            gap_xs = []
            gap_widths = []
            prev_x_end = 0.5
            for _ in range(num_gaps):
                gap_start = prev_x_end + np.random.uniform(min_island_size, max_island_size)
                gap_xs.append(gap_start)
                width = np.random.uniform(0.1, 0.3)
                gap_widths.append(width) 
                prev_x_end = gap_start + width
            gapHeightMap(heightmap_info, gap_xs, gap_widths).save(os.path.join(path, str(total_count) + ".csv"))
            total_count += 1
        num_gaps += 1
    return total_count


def generateStairsMaps(num_heightmaps, max_num_steps, heightmap_info, path):
    if not os.path.exists(path):
        os.mkdir(path)
    num_each = num_heightmaps//(max_num_steps+1)
    num_steps = 0
    total_count = 0
    min_stair_size = 0.3
    max_stair_size = 1
    while num_stairs < max_num_stairs+1:
        for i in range(num_each):
            stair_xs = []
            stair_heights = []
            prev_x_end = 0.2
            for _ in range(num_gaps):
                step_start = prev_x_end + np.random.uniform(min_step_size, max_step_size)
                step_xs.append(step_start)
                prev_x_end = step_start
            stairHeightMap(heightmap_info, stair_xs, stair_heights).save(os.path.join(path, str(total_count) + ".csv"))
            total_count += 1
        num_gaps += 1
    return


def generateFullDataset(num_heightmaps, num_init_states, random_params, config_lims, terrains_folder, max_num_procs = 20):
    if not os.path.isabs(terrains_folder):
        terrains_folder = os.path.join(os.getcwd(), terrains_folder)

    schedule = []
    for i in range(num_heightmaps):
        schedule += [i] * num_init_states

    counter = 0
    all_terrains = []
    all_initial_states = []
    all_sequences = []
    while counter < len(schedule):
        terrains = schedule[counter:counter+max_num_procs]
        all_params = [(config_lims, random_params, os.path.join(terrains_folder, str(t) + ".csv"), os.path.join(terrains_folder, str(t) + "_" + str(j) + "seq.csv"), np.random.randint(0, 1000)) for j,t in enumerate(terrains)]
        p = mp.Pool(processes = len(all_params))
        return_val_array = p.starmap(runOneOptimization, all_params)
        counter += max_num_procs
        print(f"finished up to {counter}/{len(schedule)}")
        for i, arr in enumerate(return_val_array):
            if len(arr[0]) > 0:
                all_sequences.append(arr[0])
                all_terrains.append(arr[1].terrain_array)
                all_initial_states.append(arr[2])
    return all_sequences, all_terrains, all_initial_states


def testHeightMapPlotting():
    disc = 0.2
    num_ditches = 2
    hmap = randomDitchHeightMap(8, 2, disc, num_ditches)
    step_sequence = [[0, 0], [1, -0.2], [2, 0], [3, -0.2]]
    hmap.plotSteps(step_sequence, lines = False)


def main():
    # read the yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", required = True)
    args = parser.parse_args()
    config_fname = args.config

    with open(config_fname, 'r') as f:
        config = yaml.safe_load(f)
    random_param_names = ["num_steps", "init_x_vel", "init_y_vel", "T", "goal_x"]

    # create a random ditch heightmap
    terrains_folder = "heightmaps"
    hmap_info = config["hmap_info"]
    num_heightmaps = config["num_heightmaps"]
    num_sequences = config["num_sequences"]
    num_maps_made = generateGapMaps(num_heightmaps, 4, hmap_info, terrains_folder)
    all_sequences, all_states, all_terrains = generateFullDataset(num_maps_made, num_sequences, random_param_names, config, "heightmaps", 15) 
    print(f"made {len(all_sequences)} sequences")

    np.save(os.path.join(terrains_folder, "all_sequences.npy"), all_sequences)
    np.save(os.path.join(terrains_folder, "all_states.npy"), all_states)
    np.save(os.path.join(terrains_folder, "all_terrains.npy"), all_terrains)
    # heightmap params is included in this!.
    shutil.copyfile(config_fname, os.path.join(terrains_folder, "config.yaml"))
    

def main2():
    terrains_folder = "heightmaps"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", required = True)
    parser.add_argument("--hmap_name", type=str)
    args = parser.parse_args()
    config_fname = args.config
    hmap_name = args.hmap_name

    hmap_fname = os.path.join(terrains_folder, hmap_name + ".csv")
    seq_fname = os.path.join(terrains_folder, hmap_name + "_test_seq.csv")

    with open(config_fname, 'r') as f:
        config = yaml.safe_load(f)

    random_param_names = ["num_steps", "init_x_vel", "init_y_vel", "T", "goal_x"]
    seq, state = runOneOptimization(config, random_param_names, hmap_fname, seq_fname, 442)
    hmap = HeightMapFromCSV(hmap_fname)
    hmap.plotSteps(seq)
    
    
if __name__ == "__main__":
    main2()
