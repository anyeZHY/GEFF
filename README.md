# Gaze_Estimation

 Project of AI2611, Shanghai Jiao Tong University.

## Introduction

The "contributions" of this project are as follows:

- Transfer PIXIE model from human body reconstruction to gaze estimation. Now the features of head fuse with that of eyes.
- Implement SimCLR framework for training deep and complicated network.
- Transform datas: Flip the images horizontally! Swap the left eyes and the right eyes. 👀

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

### Download assets and process datas

- For **SJTUers**, download datasets from [Gaze2022](https://jbox.sjtu.edu.cn/v/link/view/d7dad40649094e1fb6c6a93678ef9512) whose access code is `mrte`.
- Also you could download them from [MPIIFaceGaze](https://github.com/hysts/pytorch_mpiigaze) and [ColumbiaGazeDataSet](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/).

Put them into `assets`. Now the folder `assets/` should look like:

```
assets
├── MPIIFaceGaze/
│   └── ...
└── ColumbiaGazeDataSet/
    └── ...
```

Also you could preprocess your by following command:

```shell
$ python ge/utils/dataloader.py
```

```shell
assets
├── MPIIFaceGaze/
│   └── ...
├── ColumbiaGazeDataSet/
│   └── ...
├── MPII_train.csv
├── MPII_val.csv
├── MPII_test.csv
└── ...
```

## Demo

We save our model at `assets/model_saved/`. You could run the demo to see the results on validation set.

```shell
$ python scripts/naive_res_demo.py
```

## Training Part

Note: the following scripts may take a while ( ~ 5 hours ).

### MPIIGaze

Get our baseline.

```shell
$ python res_train.py --model 'baseline' --lr_step 20 --lr_gamma 0.7 --data_aug
```

Train the GEFF architecture.

```shell
$ python res_train.py --model 'geff' --t 1\ 
		--lr_step 20 --lr_gamma 0.5 \
		--data_aug \
		--useres \
		--epoch 100 --lr 0.001
```

## Development Team

- Haoyu Zhen [@**anyeZHY**](https://github.com/anyeZHY)
- YiLin Sun [@**SylvanSun**](https://github.com/SylvanSun)
