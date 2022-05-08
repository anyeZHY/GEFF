import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os


def generate_loss_logs():
    """
        The file tree should be the same as project GE
    """
    path = str(Path.cwd().parent.parent)+'/log/'
    files = os.listdir(path)
    logs = []
    file_names = []
    for file in files:
        if file.endswith(".txt"):
            loss_log = []
            df = pd.read_table(path + file, header=None, skiprows=0)
            for i in range(len(df)):
                if str(df[0][i])[0:4] == "Test":
                    loss_log.append(float(str(df[0][i])[16:]))
            logs.append(loss_log)
            file_names.append(file[:-4])
    return logs, file_names

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

def plot_loss_one_graph(files, logs):
    plt.title("Validation losses with various models")
    for i in range(len(files)):
        plt.plot(logs[i], label=files[i], linewidth=1.1, )
    plt.legend()
    plt.xlabel('Train epoches')
    plt.ylabel('Validation loss')
    # plt.show()
    plt.savefig('../../figs/Validation_losses.pdf')


def generate_log_files(file_dir):

    pass


if __name__ == '__main__':
    loss_logs, file_names = generate_loss_logs()
    loss_logs = np.array(loss_logs)
    # start = 34
    # loss_logs = smooth_log(loss_logs[:, start:])
    plot_loss_one_graph(file_names, loss_logs)



