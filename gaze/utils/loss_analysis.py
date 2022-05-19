import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os


def generate_loss_logs():
    logs1 = []
    names1 = []  # store 100 training epoch
    logs2 = []
    names2 = []  # store 200 training epoch
    path = str(Path.cwd()) + '/log/base/'
    files = os.listdir(path)
    for file in files:
        if file.endswith(".txt"):
            loss_log = []
            df = pd.read_table(path + file, header=None, skiprows=0)
            for i in range(len(df)):
                if str(df[0][i])[0:4] == "Test":
                    loss_log.append(float(str(df[0][i])[16:]))
            if len(loss_log) == 100:
                logs1.append(loss_log)
                names1.append(file[:-4])
            if len(loss_log) == 200:
                logs2.append(loss_log)
                names2.append(file[:-4])
    path = str(Path.cwd()) + '/log/fuse/'
    files = os.listdir(path)
    for file in files:
        if file.endswith(".txt"):
            loss_log = []
            df = pd.read_table(path + file, header=None, skiprows=0)
            for i in range(len(df)):
                if str(df[0][i])[0:4] == "Test":
                    loss_log.append(float(str(df[0][i])[16:]))
            if len(loss_log) == 100:
                logs1.append(loss_log)
                names1.append(file[:-4])
            if len(loss_log) == 200:
                logs2.append(loss_log)
                names2.append(file[:-4])
    path = str(Path.cwd()) + '/log/geff/'
    files = os.listdir(path)
    for file in files:
        if file.endswith(".txt"):
            loss_log = []
            df = pd.read_table(path + file, header=None, skiprows=0)
            for i in range(len(df)):
                if str(df[0][i])[0:4] == "Test":
                    loss_log.append(float(str(df[0][i])[16:]))
            if len(loss_log) == 100:
                logs1.append(loss_log)
                names1.append(file[:-4])
            if len(loss_log) == 200:
                logs2.append(loss_log)
                names2.append(file[:-4])
    return logs1, names1, logs2, names2


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


def show_mean_loss_archive(files, loss, position):
    path = str(Path.cwd().parent.parent) + '/log/'
    mean = np.mean(loss[:, position:], axis=1)
    print(f'mean validation loss on epoches {position} to 100')
    for i in range(len(files)):
        df = pd.read_table(path + files[i] + '.txt', header=None, skiprows=0)
        print(round(mean[i], 4), '\n', df[0][1], '\n')


def show_mean_loss(files, loss, length):
    path = str(Path.cwd()) + '/log/'
    pos = np.argmin(loss, axis=1)
    # pos60 = min(pos, length-60)
    # pos20 = min(pos, length-20)
    min = np.min(loss, axis=1)
    pos60 = length-60
    pos20 = length-20
    mean60 = np.mean(loss[:, pos60:], axis=1)
    mean20 = np.mean(loss[:, pos20:], axis=1)
    print(f'mean loss on epoches {pos60} to {length} and {pos20} to {length} and global min loss')
    for i in range(len(files)):
        df = pd.read_table(path + files[i][:4] +'/' + files[i] + '.txt', header=None, skiprows=0)
        print('min', round(min[i], 4), round(mean60[i], 4), round(mean20[i], 4), files[i], '\n', df[0][1], '\n')


if __name__ == '__main__':
    loss100, file100, loss200, file200 = generate_loss_logs()
    loss100 = np.array(loss100)
    loss200 = np.array(loss200)
    pos1 = np.argmin(loss100, axis=1)
    pos2 = np.argmin(loss200, axis=1)
    # print(pos1, pos2)
    # start = 34
    # loss_logs = smooth_log(loss_logs[:, start:])
    # plot_loss_one_graph(file_names, loss_logs)
    # plot_loss_multi_graph(file_names, loss_logs)
    # show_mean_loss(file100, loss100, 100)
    # show_mean_loss(file200, loss200, 200)
