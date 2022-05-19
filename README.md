# GEFF: Gaze Estimation with Fused Features

Project of AI2611 Machine Learning, Shanghai Jiao Tong University.

## Introduction

The report of this project is available at: [GEFF (PDF)](somewhere).

The "contributions" of this project are as follows:

- Transfer [PIXIE](https://github.com/YadiraF/PIXIE) [CVPR 2021] model from 3D human body reconstruction to gaze estimation. Now the features of head fuse with that of eyes (we call our model as GEFF).
- Implement [SimCLR](https://github.com/google-research/simclr) [ICML 2020] framework for training deep and complicated network (Currently the SimCLR framework was adapted for GEFF).
- Augment datas. Flip the images horizontally. Swap the left eyes and the right eyes. Use masks to generalize our model.

## Installation

### Getting start

Clone our repository:

```shell
$ git clone --recursive https://github.com/anyeZHY/GEFF.git
$ cd GEFF
```

Set up a new environment:

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
$ python gaze/utils/dataloader.py
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

Then you need to crop figures for Columbia Data Set: (~[ time ] hours)

```shell
$ python [ name ].py
```

## Demo (update soon)

We save our model at `assets/model_saved/`. You could run the demo to see the results on validation set.

```shell
$ python scripts/naive_res_demo.py
```

[ gif ]

## Training Part

Our GEFF architecture is similar to PIXE architecture:

[ image ]

The best model will be saved in path `assets/model_saved/` after running the following commands.

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

You could add command `--useres` to use ResNet as Eyes' encoder.

### ColumbiaGaze (update soon)

Similarly, just add `--Columbia` behind the commands above.

### SimCLR

We design the framework of SimCLR in gaze estimation task:

<p align="center">
  <img src="figs/SimCLR.png", width="40%"/></br>
	SimCLR framework
</p>


We implement a stronger augmentation for datas, especially for the images of faces. Here's some examples:

<p align="center">
  <img src="figs/simclr_tran.png", width="30%"/></br>
	Data Augmentation
</p>


Run the following command to get our pre-trained model:

```shell
$ python simclr_train.py --batch 256 --tau 0.5 --epoch 500 --multi_gpu
or
$ python simclr_train.py --batch 128 --tau 0.5 --epoch 500
```

**Note:** since large batch size and long training time matters, we use **3** GPUs when training. It may take ~**100** hours.

Then run the following script for a quick test

```shell
$ python res_train.py --lr 0.0005 --epoch 100 --print_every 10
		--model 'simclr' --t 1\ 
		--data_aug --flip 0.5 \
		--lr_step 20 --lr_gamma 0.5
```



## Results

### Ablation study (update soon)

[ image ]

### For TAs (update soon)

We provide a python file to test on the datas which are not access to students.

```shell
$ python scipts/test.py --MPII --folders 10-16
```

## Development Team

- Haoyu Zhen: [@**anyeZHY**](https://github.com/anyeZHY)
- YiLin Sun: [@**SylvanSun**](https://github.com/SylvanSun)
