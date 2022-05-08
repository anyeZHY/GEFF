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
    names = []
    for file in files:
        if file.endswith(".txt"):
            loss_log = []
            df = pd.read_table(path + file, header=None, skiprows=0)
            for i in range(len(df)):
                if str(df[0][i])[0:4] == "Test":
                    loss_log.append(float(str(df[0][i])[16:]))
            logs.append(loss_log)
            names.append(file[:-4])
    return logs, names


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
        plt.plot(logs[i], label=files[i], linewidth=0.5,)
    plt.legend()
    plt.xlabel('Train epoches')
    plt.ylabel('Validation loss')
    plt.ylim([0, 10])
    # plt.show()
    plt.savefig('../../figs/loss/Validation_losses.pdf')


def plot_loss_multi_graph(files, logs):
    for i in range(len(files)):
        plt.title("Validation losses with various models")
        plt.cla()
        plt.plot(logs[i], label=files[i], linewidth=1.1, )
        plt.legend()
        plt.xlabel('Train epoches')
        plt.ylabel('Validation loss')
        # plt.show()
        plt.savefig(f'../../figs/loss/{files[i]}.pdf')


def show_mean_loss(files, loss, position):
    path = str(Path.cwd().parent.parent) + '/log/'
    mean = np.mean(loss[:, position:], axis=1)
    print(f'mean validation loss on epoches {position} to 100')
    for i in range(len(files)):
        df = pd.read_table(path + files[i] + '.txt', header=None, skiprows=0)
        print(round(mean[i], 4), '\n', df[0][1], '\n')


def show_mean_loss_2(files, loss, position1, position2):
    path = str(Path.cwd().parent.parent) + '/log/'
    mean1 = np.mean(loss[:, position1:], axis=1)
    mean2 = np.mean(loss[:, position2:], axis=1)
    print(f'mean validation loss on epoches {position1} to 100 and {position2} to 100')
    for i in range(len(files)):
        df = pd.read_table(path + files[i] + '.txt', header=None, skiprows=0)
        print('(40,100)', round(mean1[i], 4), '(80,100)', round(mean2[i], 4), '\n', df[0][1], '\n')


if __name__ == '__main__':
    loss_logs, file_names = generate_loss_logs()
    loss_logs = np.array(loss_logs)
    # start = 34
    # loss_logs = smooth_log(loss_logs[:, start:])
    # plot_loss_one_graph(file_names, loss_logs)
    # plot_loss_multi_graph(file_names, loss_logs)
    show_mean_loss_2(file_names, loss_logs, 40, 80)
