#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./.local/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="./.local/lib/python3.9/site-packages:$LD_LIBRARY_PATH"

cd
cd /Users/yanasotirova/Desktop/FINAL_CODE/cabi

TASK="walker2d-random-v0"
env="Walker2d"
version="random"
work_dir="<directory_where_the_data_is>"
isMediumExpert=False

python examples/train_d4rl.py --algo_name="bcq" --task "d4rl-${TASK}" --seed 45 --real-data-ratio 0.7 --horizon 5 --isMediumExpert ${isMediumExpert} --data_proportion 10 --data_path "${work_dir}/data/${env}/body_mass/body_${version}.hdf5"