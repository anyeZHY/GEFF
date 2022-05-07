import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os


def generate_loss_logs(files):
    """
        The file tree should be the same as project GE
    """

    path = str(Path.cwd().parent.parent)
    logs = []
    for file in files:
        loss_log = []
        df = pd.read_table(path + file, header=None, skiprows=0)
        for i in range(len(df)):
            if str(df[0][i])[0:4] == "Test":
                loss_log.append(float(str(df[0][i])[-5:]))
        logs.append(loss_log)
    return logs

def smooth_log(logs):
    logs = np.array(logs)
    result = []
    for log in logs:
        total = 0
        subresult = []
        for i, cur in enumerate(log, 1):
            total += cur
            subresult.append(total/i)
        result.append(subresult)
    return result

def plot_loss(files, logs):
    plt.title("Test losses with various models")
    for i in range(len(files)):
        plt.plot(logs[i], label=files[i][5:-4], linewidth=1.1, )
    plt.legend()
    plt.xlabel('Train epoches')
    plt.ylabel('Test loss')
    # plt.show()
    plt.savefig('../../figs/Test_losses.pdf')


def generate_log_files(file_dir):
    pass


if __name__ == '__main__':
    log_files = ['/log/baseline.txt', '/log/naive fuse.txt',
                 '/log/fuse lr=1e-4.txt', '/log/fuse lr=5e-4.txt']
    loss_logs = generate_loss_logs(log_files)
    loss_logs = smooth_log(loss_logs)
    plot_loss(log_files, loss_logs)



