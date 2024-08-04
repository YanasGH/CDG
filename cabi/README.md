# Confidence-Aware Bidirectional Offline Model-Based Imagination (CABI)

Code for "Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination" (NeurIPS 2022). 
This code is mainly based on the original code of CABI, which is highly based on the [offlineRL](https://agit.ai/Polixir/OfflineRL) repository.

## Experiments
To train and evaluate CABI, use the provided bash file by including the directory of the cabi folder (which is the same directory as the data folder)  in `work_dir`. Run the following command in Terminal/Conda prompt:
```bash
bash train_cabi_bcq.sh
```
Currently, the environment in the bash file is set to `walker2d-random-v0`. If you wish to train CABI for a different environment, choose from the following list:
- `walker2d-random-v0`
- `walker2d-medium-v0`
- `walker2d-medium-replay-v0`
- `walker2d-medium-expert-v0`
- `hopper-random-v0`
- `hopper-medium-v0`
- `hopper-medium-replay-v0`
- `hopper-medium-expert-v0`

Make sure to change the `TASK`, `env` and `version` using the correct formatting (as shown in the bash file). Finally, if you run either of the two medium-expert environments, make sure to set `isMediumExpert` to True.

## References
<a id="1">[1]</a>
[Confidence-Aware Bidirectional Offline Model-Based Imagination (CABI)](https://proceedings.neurips.cc/paper_files/paper/2022/file/f9e2800a251fa9107a008104f47c45d1-Paper-Conference.pdf), Lyu et al, 2022
