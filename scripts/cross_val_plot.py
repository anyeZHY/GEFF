import numpy as np
import matplotlib.pyplot as plt
RGB = [
    [142, 207, 201],
    [255, 190, 122],
    [250, 127, 111],
    [130, 176, 210],
    [190, 184, 220],
    [231, 218, 210],
    [40,  120, 181],
    [200, 36,  35],
    [84,  179, 69],
    [199, 109, 162]
]
color = np.array(RGB)/255
index = np.arange(1,11)
losses = {
    'Baseline': [
        3.1242787933349607, 4.86889510345459, 3.8676000277201337, 4.143331093470255, 4.171029663085937,
        5.045097078959147, 4.807037024180095, 6.198452950795492, 5.420211437225341, 6.1758616994222
    ],
    'GEFF-MLP': [
        2.928028965632121, 4.0780718015035, 3.08648743947347, 4.187667137781779, 3.904286936442057,
        4.106054110209147, 4.161072677612305, 5.401485674540202, 5.148667079925537, 5.4365968475341795
    ],
    'GEFF (ResNet)':[
        2.9471873995463054, 4.102914237976075, 3.2502939828236896, 4.126466691335042, 3.7709080130259194,
        4.498519364674886, 4.386260398864746, 5.601324507395426, 4.875124418258667, 5.308706303914388
    ],
    'Vanilla Fusion': [
        3.0573593254089357, 2.9676180566151937, 4.203337666829427, 3.8255641911824543, 3.740517753601074,
        4.069397972742716, 4.306864625295003, 5.626141728719076, 5.401134460449219, 5.004379927317301
    ],
    'FE-Baseline': [
        2.8354159075419108, 3.4598169809977213, 3.343762866338094, 4.282653340021769, 4.092017140070597,
        4.120004873911539, 3.7727598075866697, 5.359767752329509, 4.932834992726644, 4.839421946207683
    ],
    'GEFF-MLP (mask off)':[
        3.0436652647654214, 3.972334520339966, 3.2451619459788006, 3.5805965251922607, 3.6481445287068683,
        4.330997540791829, 3.8420042432149253, 5.47896527226766, 4.990495882670085, 5.030974634806315
    ],
    'GEFF-MLP (60 epoch)':[
        2.539538073857625, 3.8029450600941974, 3.0754419956207277, 3.7631584765116375, 3.8390845959981283,
        4.2228104184468584, 4.15600775273641, 5.294918661753337, 5.042807373046875, 5.1396954752604165
    ],
    'GEFF-MLP (100 epoch)':[
        2.5233322798411053, 3.869793039957682, 3.079336062113444, 3.931205847422282, 3.364947898228963,
        4.2196826725006105, 3.9490752258300783, 5.207069086710612, 4.966465431213379, 5.019186856587728
    ],
    'GEFF-RF (heavier data-Aug)':[
        2.5276485888163247, 3.4206851921081545, 3.045332301457723, 3.63777263323466, 3.4291863441467285,
        4.1741627604166665, 3.839682412465413, 5.325767004648845, 5.113183091481527, 5.301839869181315
    ]
}

n = 3
plt.figure(figsize=(6/n * len(losses), n*5))
plt.subplots_adjust(hspace=0.3)
losses = sorted(losses.items(), key=lambda x : np.mean(x[1]), reverse=True)
print(losses)
for i in range(len(losses)):
    plt.subplot(n, (len(losses)+1)//n, i+1)
    loss = losses[i][1]
    label = losses[i][0]
    plt.bar(index, loss, color=color)
    mean = np.mean(loss)
    plt.plot([-1,11], [mean, mean], '--')
    plt.text(4.5, mean + 0.4, 'Mean: %0.3f' % mean, size=14)
    plt.xlim([0.5,10.5])
    plt.ylim([0,6.5])
    plt.ylabel(r'angular loss (degree)')
    plt.xlabel('personal ID')
    plt.title(label, size=18)
plt.savefig('figs/result.pdf')
plt.savefig('figs/result.png')
