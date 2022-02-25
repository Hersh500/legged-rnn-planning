import numpy as np
import multiprocessing as mp
import subprocess as sp
import argparse
import yaml
import csv

import hopper2d


class HeightMap:
    def __init__(self, terrain_array, corner_val_m, disc, friction = 1.0):
        self.terrain_array = terrain_array
        self.corner_val_m = corner_val_m
        self.disc = disc
        self.friction = friction
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
        row_idx = np.clip(int((y_m - self.corner_val_m[1])/self.disc),
                          0, self.terrain_array.shape[0])
        col_idx = np.clip(int((x_m - self.corner_val_m[0])/self.disc),
                          0, self.terrain_array.shape[1])
        return row_idx, col_idx

    def save(self, fname):
        hopper2d.saveTerrainAsCSV(fname, self.terrain_array, self.corner_val_m,
                                  self.disc, self.friction)


def randomDitchHeightMap(max_x, max_y, disc, num_ditches):
    x0 = 0
    y0 = -2
    terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(max_x - x0, max_y - y0, disc, num_ditches) 
    return HeightMap(terrain_array, (x0, y0), disc)


def gapHeightMap(max_x, max_y, disc, gap_info):
    x0 = 0
    y0 = -2
    for gap in gap_info:
        gap[0] -= x0
        gap[1] -= y0
    terrain_array, terrain_func = hopper2d.generateTerrain2D(max_x - x0, max_y - y0, disc, gap_info)
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
    return step_sequence


def generateFullDataset():
    return


# generate some random dataset with some gaps.
def generateGapMaps(num_heightmaps, max_val_m, min_val_m, max_num_gaps, path):
    num_each = num_heightmaps//(max_num_gaps + 1)
    num_gaps = 0
    total_count = 0
    while num_gaps < max_num_gaps+1:
        for i in range(num_each):
            # generate a heightmap with the prereq number of gaps
            # save the heightmap as a csv in the folder marked by path


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
    param_names = ["num_steps", "init_x_vel", "init_y_vel", "T", "goal_x"]

    # create a random ditch heightmap
    disc = 0.1
    num_ditches = 2
    # hmap = randomDitchHeightMap(8, 2, disc, num_ditches)
    hmap = gapHeightMap(8, 2, disc, [[1, -2, 0.4, 4], [1.5, -2, 0.2, 4]])
    csvname = "gap_hmap_test.csv"
    hmap.save(csvname)

    seq = runOptimization(config, param_names, "/home/hersh/Programming/legged_planning/" + csvname)
    hmap.plotSteps(seq, lines = True)


if __name__ == "__main__":
    main()
