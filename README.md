# Cross-Domain Generalization with Reverse Dynamics Models in Offline Model-Based Reinforcement Learning

Code for the paper "Cross-Domain Generalization with Reverse Dynamics Models in Offline Model-Based Reinforcement Learning".
Each folder contains the code for the baselines BMIL, CABI, and ROMI, as well as the newly proposed method, CARI.
Each folder contains the instructions how to run the respective method in the README file.

## Installation
Install the requirements using the following commands:
```bash
pip install -r requirements.txt
pip install -e .
```
d3rlpy==1.0.0

mkl-service==2.4.0

### D4RL
```shell
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```
For more details on use, please see [d4rl](https://github.com/rail-berkeley/d4rl).

### Data
The data used in this setup is collected by Liu et al., 2022, in 
[DARA: Dynamics-Aware Reward Augmentation in Offline Reinforcement Learning](https://openreview.net/forum?id=9SDQB3b68K).
Please download the data from https://drive.google.com/drive/folders/11kCzonIXzpkXfjyqM71-jv6ULeGuJRJ6?usp=share_link
and put it in the `data` directory of this repository. The directory structure should be
data/Hopper/body_mass/body_<version>.hdf5 and data/Walker2d/body_mass/body_<version>.hdf5, where version is random,
medium, medium_replay, medium_expert.


## How to run
To run the code for each method, follow the sinstructions provided in each README file in the subfolders. Currently, the 
environment in the bash files is set to `walker2d-random-v0`. The list of available environments is the following:

- `walker2d-random-v0`
- `walker2d-medium-v0`
- `walker2d-medium-replay-v0`
- `walker2d-medium-expert-v0`
- `hopper-random-v0`
- `hopper-medium-v0`
- `hopper-medium-replay-v0`
- `hopper-medium-expert-v0`


## References
<a id="1">[1]</a>
[DARA: Dynamics-Aware Reward Augmentation in Offline Reinforcement Learning](https://openreview.net/forum?id=9SDQB3b68K), Liu et al., 2023

<a id="2">[2]</a>
[Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337), Park & Wong, 2022

<a id="3">[3]</a>
[Confidence-Aware Bidirectional Offline Model-Based Imagination (CABI)](https://proceedings.neurips.cc/paper_files/paper/2022/file/f9e2800a251fa9107a008104f47c45d1-Paper-Conference.pdf), Lyu et al, 2022

<a id="4">[4]</a>
[Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/pdf/2110.00188), Wang et al, 2022