#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./.local/lib/python3.10/site-packages"

cd
cd ...

TASK="walker2d-random-v0"
env="Walker2d"
version="random"
work_dir="<directory_where_the_data_is>"
seed="24"

python scripts/train_model_cari.py --env_name=${TASK} --seed=${seed} --train_reverse_model --data_path="${work_dir}/data/${env}/body_mass/body_${version}.hdf5"