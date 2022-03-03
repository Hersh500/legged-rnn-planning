import numpy as np
import multiprocessing as mp
import subprocess as sp
import argparse
import yaml
import csv
import os

import hopper2d


class HeightMap:
    def __init__(self, terrain_array, heightmap_info):
        self.info = heightmap_info
        self.terrain_array = terrain_array
        return

    def plotSteps(self, step_sequence_m, lines = False):
        import seaborn as sns
        import matplotlib.pyplot as plt
        labels = np.array([''] * self.terrain_array.size).reshape(self.terrain_array.shape)
        for i, step in enumerate(step_sequence_m):
            labels[self.m_to_idx(step)] = str(i)

        if lines:
            ax = sns.heatmap(self.terrain_array,
                             cmap='inferno',
                             annot=labels,
                             annot_kws = {'fontsize':8},
                             fmt='s',
                             linewidths = 0.1, linecolor='black',square = True) 
        else:
            ax = sns.heatmap(self.terrain_array,
                             cmap='inferno',
                             annot=labels,
                             annot_kws = {'fontsize':8},
                             fmt='s', square = True) 
            
        plt.show()
             

    def m_to_idx(self, step):
        x_m, y_m = step
        row_idx = np.clip(int((y_m - self.info["corner_val_m"][1])/self.info["disc"]),
                          0, self.terrain_array.shape[0])
        col_idx = np.clip(int((x_m - self.info["corner_val_m"][0])/self.info["disc"]),
                          0, self.terrain_array.shape[1])
        return row_idx, col_idx

    def save(self, fname):
        hopper2d.saveTerrainAsCSV(fname, self.terrain_array, self.info["corner_val_m"],
                                  self.info["disc"], self.info["friction"])


def randomDitchHeightMap(max_x, max_y, disc, num_ditches):
    x0 = 0
    y0 = -2
    terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(max_x - x0, max_y - y0, disc, num_ditches) 
    return HeightMap(terrain_array, (x0, y0), disc)


def collateOutput(output_path):
    return


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
def runOneOptimization(config_lims, param_names, terrain_file):
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
            "--goal_x=" + str(goal_x)]

    # run this in a blocking way
    val = sp.call([towr_path] + args)
    step_sequence = []
    if val == 0:
        # once you get the results, read the output csv and return the sequence, terrain, initial state
        output_fname = terrain_file + "output" 
        with open(output_fname) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                step_sequence.append([float(row[0]), float(row[1])])
    return step_sequence, initial_state


def gapHeightMap(heightmap_info, gap_xs, gap_widths):
    x0 = heightmap_info["corner_val_m"][0]
    y0 = heightmap_info["corner_val_m"][1]

    gap_info =  [[gap_xs[i] - x0, 0, gap_widths[i], heightmap_info["max_y"] - y0] for i in range(len(gap_xs))]
    terrain_array, terrain_func = hopper2d.generateTerrain2D(heightmap_info["max_x"] - x0, heightmap_info["max_y"] - y0,
                                                             heightmap_info["disc"], gap_info)
    return HeightMap(terrain_array, heightmap_info)


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
                width = np.random.uniform(0.3, 0.5)
                gap_widths.append(width) 
                prev_x_end = gap_start + width
            gapHeightMap(heightmap_info, gap_xs, gap_widths).save(os.path.join(path, str(total_count) + ".csv"))
            total_count += 1
        num_gaps += 1
    return total_count


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
    while counter < num_heightmaps:
        terrains = schedule[counter:counter+max_num_procs]
        all_params = [(config_lims, random_params, os.path.join(terrains_folder, str(t) + ".csv")) for t in terrains]
        p = mp.Pool(processes = len(all_params))
        return_val_array = p.starmap(runOneOptimization, all_params)
        counter += max_num_procs
        print(f"finished up to {counter}")
        for i, arr in enumerate(return_val_array):
            if len(arr[0]) > 0:
                all_terrains.append(terrains[i])
                all_initial_states.append(arr[1])
                all_sequences.append(arr[0])
    return all_sequences, all_initial_states, all_terrains


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
    hmap_info = config["hmap_info"]
    num_maps_made = generateGapMaps(4, 1, hmap_info, "heightmaps")
    all_sequences, all_states, all_terrains = generateFullDataset(num_maps_made, config["num_sequences"], random_param_names, config, "heightmaps", 10) 



if __name__ == "__main__":
    main()
