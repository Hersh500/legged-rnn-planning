import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def plot_sphere(radius, center_x, center_y, center_z, ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center_x
    y = radius * np.outer(np.sin(u), np.sin(v)) + center_y
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z
    ax.plot_surface(x, y, z, color='b')
    return


# decide if this is an array or the state object.
def plot_robot(body_pose, foot_pose, ax):
    plot_sphere(0.25, body_pose[0], body_pose[1], body_pose[2], ax)
    plot_sphere(0.1, foot_pose[0], foot_pose[1], foot_pose[2], ax)
    ax.plot([body_pose[0], foot_pose[0]],
            [body_pose[1], foot_pose[1]],
            [body_pose[2], foot_pose[2]])
    return


def animateMoving2DHopper(body_poses, foot_poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    def animFunc(i):
        ax.clear()
        ax.set_ylim(-1, 5)
        ax.set_zlim(-1, 3)
        ax.set_xlim(-1, 5)
        plot_robot(body_poses[i], foot_poses[i], ax)
        plot_terrain(
        return

    sim = animation.FuncAnimation(fig, animFunc, frames = range(len(body_poses)))
    sim.save(filename = "test.mp4", fps = 10, dpi = 100)
    return

def plot_terrain_3d(terrain_func, ax, disc, horizon):
    return


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") 
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_zlim(-1, 3)
    body_poses = [[0, 0, z] for z in np.linspace(0.5, 1.0, 10)]
    foot_poses = [[0, 0, z - 0.5] for z in np.linspace(0.5, 1.0, 10)]
    animateMoving2DHopper(body_poses, foot_poses)
    

if __name__ == "__main__":
    main()
