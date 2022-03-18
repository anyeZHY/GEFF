# Gaze_Estimation

 Project of AI2611, Shanghai Jiao Tong University.

## Installation

### getting start

```shell
$ git clone --recursive https://github.com/anyeZHY/Gaze_Estimation.git
$ cd Gaze_Estimation
```

### Set up new environment:

```shell
$ conda env create -f environment.yaml
$ conda activate GE
```

### Download assets

- For **SJTUers**, download datasets from [Gaze2022](https://jbox.sjtu.edu.cn/v/link/view/d7dad40649094e1fb6c6a93678ef9512) whose access code is `mrte`.

- Also you could download them from [MPIIFaceGaze](https://github.com/hysts/pytorch_mpiigaze) and [ColumbiaGazeDataSet](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/).

Put them into `assests`. Now your `assets/` folder should look like:

```
assets
├── MPIIFaceGaze/
│   └── ...
└── ColumbiaGazeDataSet/
    └── ...
```
