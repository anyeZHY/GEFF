import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def get_frame(ax):
    side_length = 2
    ax.axes.set_xlim3d(left=-side_length / 2, right=side_length)
    ax.axes.set_ylim3d(bottom=-side_length, top=side_length)
    ax.axes.set_zlim3d(bottom=-side_length, top=side_length)

    # Prepare arrays x, y, z
    theta = np.linspace(0, np.pi, 100)
    z = np.zeros_like(theta)
    x = np.sin(theta)
    y = np.cos(theta)

    frame = 1.5
    ax.plot(
        [0, 0, 0, 0, 0],
        [-frame, -frame, frame, frame, -frame],
        [frame, -frame, -frame, frame, frame]
    )

    ax.plot(x, y, z, '--k')
    ax.plot(x, z, y, '--k')
    ax.plot(z, x, y, '--k')
    ax.plot(z, -x, -y, '--k')

def gaze_visual(gaze, show=True):
    alpha = gaze[0]
    beta = gaze[1]
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])

    get_frame(ax = ax)

    arrow = Arrow3D(
        [0, np.cos(beta)*np.cos(alpha)],
        [0, np.cos(beta)*np.sin(alpha)],
        [0, np.sin(beta)],
        mutation_scale=7, arrowstyle="-|>", color="b"
    )
    ax.add_artist(arrow)

    ax.legend()
    ax.view_init(elev=0, azim=0.)
    # ax.set_axis_off()
    # ax.grid(False)
    if show:
        plt.show()

if __name__=='__main__':
    gaze_visual([-np.pi/2, 0.5])
