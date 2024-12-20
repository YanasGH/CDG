#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./.local/lib/python3.10/site-packages"

cd
cd ...

DEVICE="cpu"
METHOD="bmil"
TASK="walker2d_random_v0"
TASK_d4rlname="walker2d-random-v0"
isMediumExpert=False

WANDB_API_KEY="..."
wandb login "$WANDB_API_KEY"

CMD=(python -u experiments/"${METHOD}".py
  +experiment="${METHOD}"/"${TASK}"
  policy.train.n_epoch=1  # for demonstration
  device="$DEVICE"
  work_dir="<directory_where_the_data_is>"
  +env_name_str="$TASK_d4rlname"
  +isMediumExpert=$isMediumExpert
)


echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"