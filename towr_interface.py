import numpy as np
import multiprocessing as mp
import subprocess as sp
import argparse
import yaml
import csv
import matplotlib.pyplot as plt

import hopper2d


class HeightMap:
    def __init__(self, terrain_array, corner_val_m, disc):
        self.terrain_array = terrain_array
        self.corner_val_m = corner_val_m
        self.disc = disc
        return

    def plotSteps(self, step_sequence_m, lines = False):
        import seaborn as sns
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


def randomDitchHeightMap(max_x, max_y, disc, num_ditches):
    x0 = 0
    y0 = -2
    terrain_array, terrain_func = hopper2d.generateRandomTerrain2D(max_x - x0, max_y - y0, disc, num_ditches) 
    return HeightMap(terrain_array, (0, -2), disc)


def collateOutput(output_path):
    return


# this is the main function that will get called in the starmap
def runOptimization(opt_params, terrain_file):
    towr_path = opt_params["towr_path"]
    num_steps = opt_params["num_steps"]
    init_x_vel = opt_params["init_x_vel"]
    init_y_vel = opt_params["init_y_vel"]
    T = opt_params["T"]
    goal_x = opt_params["goal_x"]
    
    # run this in a blocking way
    sp.Popen([towr_path])

    # once you get the results, read the output csv and return the sequence, terrain, initial state
    output_fname = terrain_file + "output" 
    step_sequence = []
    with open(output_fname) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row) 
    return


def main():
    # read the yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", required = True)
    args = parser.parse_args()
    config_fname = args.config
    with open(config_fname, 'r') as f:
        params = yaml.safe_load(f)
    return


def testHeightMapPlotting():
    disc = 0.2
    num_ditches = 2
    hmap = randomDitchHeightMap(8, 2, disc, num_ditches)
    step_sequence = [[0, 0], [1, -0.2], [2, 0], [3, -0.2]]
    hmap.plotSteps(step_sequence, lines = False)


if __name__ == "__main__":
    testHeightMapPlotting()
