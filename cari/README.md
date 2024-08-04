# Context-Aware Reverse Imagination (CARI)

Code for Context-Aware Reverse Imagination (CARI). This code is mainly based on the code of [Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/pdf/2110.00188).

## How to run
To run the code, you can configure the environment in bash examples in `bash/`. Before running the code, set current directory to `cari`.
To train and evaluate CARI, use the provided bash file by including the directory of the cari folder (which is the same directory as the data folder)  in `work_dir`.
Currently, the environment in the bash files is set to `walker2d-random-v0`. If you wish to train CARI for a different environment, choose from the following list:
- `walker2d-random-v0`
- `walker2d-medium-v0`
- `walker2d-medium-replay-v0`
- `walker2d-medium-expert-v0`
- `hopper-random-v0`
- `hopper-medium-v0`
- `hopper-medium-replay-v0`
- `hopper-medium-expert-v0`

Make sure to change the `TASK`, `env` and `version` using the correct formatting (as shown in the bash file).

### To learn reverse models
```bash
bash train_reverse_model_cari.sh
```

### To train diverse rollout policy
```bash
bash train_reverse_bc_cari.sh
```

### To train CARI-BCQ
```bash
bash train_cari_bcq.sh
```
## References
<a id="1">[1]</a>
[Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/pdf/2110.00188), Wang et al, 2022
