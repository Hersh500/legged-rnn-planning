import numpy as np
import csv
import hopper2d

class HeightMap:
    def __init__(self, terrain_array, heightmap_info):
        self.info = heightmap_info
        self.terrain_array = terrain_array
        return 

    def at(self, x_m, y_m):
        padding = self.info["padding"]
        if (x_m > self.info["max_x"] + padding or
           x_m < self.info["corner_val_m"][0] - padding or
           y_m > self.info["max_y"] + padding or
           y_m < self.info["corner_val_m"][1] - padding):
            return -2
        row, col = self.m_to_idx((x_m, y_m))
        return self.terrain_array[row][col]


    def normal_at(self, x_m, y_m):
        return np.pi/2  


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
             

    def plotStepsFromCSV(self, csv_fname, lines = False):
        import seaborn as sns
        import matplotlib.pyplot as plt

        step_sequence_m = []
        with open(csv_fname) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                step_sequence_m.append([float(row[0]), float(row[1])])

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
                          0, self.terrain_array.shape[0] - 1)
        col_idx = np.clip(int((x_m - self.info["corner_val_m"][0])/self.info["disc"]),
                          0, self.terrain_array.shape[1] - 1)
        return row_idx, col_idx


    def save(self, fname):
        hopper2d.saveTerrainAsCSV(fname, self.terrain_array, self.info["corner_val_m"],
                                  self.info["disc"], self.info["friction"])


def HeightMapFromCSV(fname):
    terrain_array = []
    flag = False
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if flag:
               disc, friction, corner_x, corner_y = [float(h) for h in row]
            elif row[0] == "D":
                flag = True 
            else:
                tmp = [float(h) for h in row]    
                terrain_array.append(tmp)
    terrain_array = np.array(terrain_array)
    # padding is just the default value here.
    # Need to think about a better way to approach this.
    heightmap_info = {"disc": disc, "corner_val_m": [corner_x, corner_y],
                      "max_x": terrain_array.shape[1] * disc + corner_x,
                      "max_y": terrain_array.shape[0] * disc + corner_y,
                      "friction": friction, "padding": 0.25}
    return HeightMap(terrain_array, heightmap_info)
        

def randomDitchHeightMap(max_x, max_y, disc, friction, num_ditches):
    x0 = 0
    y0 = -2
    heightmap_info = {"disc": disc, "corner_val_m": [x0, y0],
                      "max_x": max_x,
                      "max_y": max_y,
                      "friction": friction, "padding": 0.25}
    return hopper2d.generateRandomTerrain2D(heightmap_info, num_ditches) 


def stairHeightMap(heightmap_info, stair_xs, stair_heights):
    x0 = heightmap_info["corner_val_m"][0]
    y0 = heightmap_info["corner_val_m"][1]
    stair_xs = stair_xs + [heightmap_info["max_x"]]

    stair_info =  [[stair_xs[i] - x0, 0, stair_xs[i+1] - stair_xs[i], heightmap_info["max_y"] - y0, stair_heights[i]] for i in range(len(stair_heights))]
    return hopper2d.generateTerrain2D(heightmap_info, stair_info)


def gapHeightMap(heightmap_info, gap_xs, gap_widths):
    x0 = heightmap_info["corner_val_m"][0]
    y0 = heightmap_info["corner_val_m"][1]

    gap_info =  [[gap_xs[i] - x0, 0, gap_widths[i], heightmap_info["max_y"] - y0] for i in range(len(gap_xs))]
    return hopper2d.generateTerrain2D(heightmap_info, gap_info)


def main():
    stair_x = [1, 2]
    stair_height = [0.5, 1]
    info = {"max_x": 4, "max_y": 2, "padding":0.25, "corner_val_m":[0, -2], "friction":0.8, "disc":0.1}
    hmap = stairHeightMap(info, stair_x, stair_height)
    hmap.plotSteps([])

if __name__ == "__main__":
    main()
