import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gaze.utils.Visualization import get_frame


def convert_str_to_float(x):
    """
    convert the string to a darray
    """
    return np.array(list(map(float, x.split(','))))

def get_postion():
    df = pd.read_csv(path + '/assets/MPII_data.csv')
    df = pd.concat([df, pd.read_csv(path + '/assets/MPII_test.csv')])

    gaze_origin = df['2DGaze'].values
    comma = ','
    gaze_joined = comma.join(list(gaze_origin))
    gaze = convert_str_to_float(gaze_joined)
    gaze_x = gaze[0::2]
    gaze_y = gaze[1::2]
    return -gaze_x[0:1000], gaze_y[0:1000]

def draw_gaze_distribution_2d():
    gaze_x, gaze_y = get_postion()
    plt.scatter(gaze_x, gaze_y, s = 0.1)
    plt.xlim([-0.4,0.4])
    plt.ylim([-0.4,0.4])
    # plt.show()

def draw_gaze_distribution_3d():
    alpha, beta = get_postion()
    print(alpha,beta)
    x= np.cos(beta)*np.cos(alpha)
    y = np.cos(beta)*np.sin(alpha)
    z = np.sin(beta)
    # x, y, z = x[0:10], y[0:10], z[0:10]

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    get_frame(ax=ax)
    print(x,y,z)
    ax.scatter(x, y, z, s = 0.001)

    ax.legend()
    ax.view_init(elev=8., azim=0.)
    # ax.set_axis_off()
    # ax.grid(False)
    # plt.show()

if __name__ == '__main__':
    # plt.subplot(1,2,1)
    draw_gaze_distribution_2d()
    # plt.subplot(1,2,2)
    draw_gaze_distribution_3d()
    plt.show()
