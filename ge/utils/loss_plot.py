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


def plot_loss(files, logs):
    plt.title("Test losses with various models")
    for i in range(len(files)):
        plt.plot(logs[i], label=files[i][4:], linewidth=1.1, )
    plt.legend()
    plt.xlabel('Train epoches')
    plt.ylabel('Test loss')
    plt.show()
    ##plt.savefig('Test_losses.jpg')


def generate_log_files(file_dir):
    pass


if __name__ == '__main__':
    log_files = ['/log1528896.out', '/log1528916.out']
    loss_logs = generate_loss_logs(log_files)
    plot_loss(log_files, loss_logs)



