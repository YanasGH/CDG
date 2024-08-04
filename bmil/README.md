# Robust Imitation of a Few Demonstrations with a Backwards Model

This repository implements code for the paper [Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337).
The code is mainly based on the original implementation of the authors!

## Experiments
To train and evaluate BMIL, use the provided bash file by including the directory of the bmil folder (which is the same directory as the data folder) in `work_dir`. Run the following command in Terminal/Conda prompt:
```bash
bash train_bmil.sh
```
Currently, the environment in the bash file is set to `walker2d-random-v0`. If you wish to train BMIL for a different environment, choose from the following list:
- `walker2d-random-v0`
- `walker2d-medium-v0`
- `walker2d-medium-replay-v0`
- `walker2d-medium-expert-v0`
- `hopper-random-v0`
- `hopper-medium-v0`
- `hopper-medium-replay-v0`
- `hopper-medium-expert-v0`

Make sure to change the `TASK` and `TASK_d4rlname` using the correct formatting (as shown in the bash file). Finally, if you run either of the two medium-expert environments, make sure to set `isMediumExpert` to True.

## References
<a id="1">[1]</a>
[Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337), Park & Wong, 2022

<a id="2">[2]</a>
[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018.

<a id="3">[3]</a>
[Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://arxiv.org/abs/1709.10087), Rajeswaran et al, 2018.
