# Learning Individual Intrinsic Reward in MARL

This repository is an implementation of **LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning.** The framework for LIIR is inherited from [PyMARL](https://github.com/oxwhirl/pymarl).  LIIR is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

```
@inproceedings{
du2019learning,
title={LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning.},
author={Yali Du and Lei Han and Meng Fang and Tianhong Dai and Ji Liu and Dacheng Tao},
booktitle={Advances in Neural Information Processing Systems},
year={2019},
}
```



## Setup

Set up the working environment:

```shell
pip install -r requirements.txt 
```

Set up the StarCraftII game core

```shell
bash install_sc2.sh  
```

## Run an experiment 

To train `LIIR`  on the map with `3 marine`, 

```shell
python3 src/main.py --config=liir_smac --env-config=sc2 --map=3m  
```

All results will be stored in the `Results` folder.



## Licence

The MIT License


