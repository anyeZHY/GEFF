# GEFF: Gaze Estimation with Fused Features

Project of AI2611 Machine Learning, Shanghai Jiao Tong University.

## Introduction

The report of this project is available at: [GEFF](somwhere).

The "contributions" of this project are as follows:

- Transfer [PIXIE](https://github.com/YadiraF/PIXIE) model from human body reconstruction to gaze estimation. Now the features of head fuse with that of eyes. 
- Implement [SimCLR](https://github.com/google-research/simclr) framework for training deep and complicated network.
- Augment datas: flip the images horizontally! Swap the left eyes and the right eyes.

## Installation

### getting start

```shell
$ git clone --recursive https://github.com/anyeZHY/Gaze_Estimation.git
$ cd Gaze_Estimation
```

### Set up new environment:

```shell
$ conda env create -f environment.yaml
$ conda activate geff
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

Then you need to crop figures for Columbia Data Set:

```shell
$ python .py
```

## Demo

We save our model at `assets/model_saved/`. You could run the demo to see the results on validation set.

```shell
$ python scripts/naive_res_demo.py
```

## Training Part

### MPIIGaze

The following scripts may take a while ( ~ 10 hours ).

First you need to get our baseline.

```shell
$ python res_train.py --epoch 100 --lr 0.001 --print_every 10 \
		--model 'baseline' --lr_step 20 --lr_gamma 0.7 --data_aug
```

**Nota Bene:** it is necessary to get our baseline model first, because we will use it as the face encoder when training GEFF and Vanilla Fusion.

Train the Vanilla Fusion

```shell
$ python res_train.py --lr 0.001 --epoch 100 --print_every 10 --name 'MFP'
		--model 'fuse' --wight 0.2\ 
		--data_aug --flip 0.5 \
		--lr_step 20 --lr_gamma 0.5 \
		--pretrain
```

Train the GEFF architecture

```shell
$ python res_train.py --lr 0.0005 --epoch 200 --print_every 10 --name 'MFP'
		--model 'geff' --t 1\ 
		--data_aug --flip 0.5 \
		--lr_step 20 --lr_gamma 0.5 \
		--pretrain
```

You could add command `--useres` to use ResNet as Eyes' encoder. The best model will be saved in path `assets/model_saved/`.

### ColumbiaGaze

To be continued.

### SimCLR

To be continued.

## Development Team

- Haoyu Zhen: [@**anyeZHY**](https://github.com/anyeZHY)
- YiLin Sun: [@**SylvanSun**](https://github.com/SylvanSun)
