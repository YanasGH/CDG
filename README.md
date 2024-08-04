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
[Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337), Park & Wong, 2022

<a id="1">[1]</a>
[Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/pdf/2110.00188), Wang et al, 2022
