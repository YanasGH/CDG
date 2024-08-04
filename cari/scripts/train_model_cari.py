import gym
import d4rl
import numpy as np
import tensorflow as tf
import os
import h5py

from mopo.mopo.models.constructor import construct_model, format_samples_for_training, \
    format_reverse_samples_for_training
from utils.dataset_utils import processed_qlearning_dataset
from utils.filesystem import mkdir

def get_data(dataset, data_path, isMediumExpert, data_proportion):
    num = int(len(dataset["observations"])/data_proportion)
    if not isMediumExpert:
        print("---------------------------------")
        print("Not isMediumExpert.")
        print("---------------------------------")
        dataset['observations'] = dataset['observations'][:num]
        dataset['actions'] = dataset['actions'][:num]
        dataset['rewards'] = dataset['rewards'][:num]
        dataset['terminals'] = dataset['terminals'][:num]
        if 'timeouts' in dataset:
            dataset['timeouts'] = dataset['timeouts'][:num]

    else:
        print("---------------------------------")
        print("IsMediumExpert.")
        print("---------------------------------")
        num=num//2
        dataset['observations'] = np.concatenate((dataset['observations'][:num], dataset['observations'][-num:]),
                                                 axis=0)
        dataset['actions'] = np.concatenate((dataset['actions'][:num], dataset['actions'][-num:]), axis=0)
        dataset['rewards'] = np.concatenate((dataset['rewards'][:num], dataset['rewards'][-num:]), axis=0)
        dataset['terminals'] = np.concatenate((dataset['terminals'][:num], dataset['terminals'][-num:]), axis=0)
        if 'timeouts' in dataset:
            dataset['timeouts'] = np.concatenate((dataset['timeouts'][:num], dataset['timeouts'][-num:]), axis=0)

    print("num:", num)
    if data_path:
        data = h5py.File(data_path, 'r')
        dataset['observations'] = np.concatenate((dataset['observations'], data['observations']), axis=0)
        dataset['actions'] = np.concatenate((dataset['actions'], data['actions']), axis=0)
        dataset['rewards'] = np.concatenate((dataset['rewards'], np.squeeze(data['rewards'])), axis=0)
        dataset['terminals'] = np.concatenate((dataset['terminals'], data['terminals']), axis=0)
        if 'timeouts' in dataset:
            dataset['timeouts'] = np.concatenate((dataset['timeouts'], data['timeouts']), axis=0)
    return dataset


def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    mkdir(args.save_dir)

    env = gym.make(args.env_name)
    
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    hidden_dim = 200
    num_networks = 7
    num_elites = 5
    model_type = "mlp"
    separate_mean_var = True
    model_name = None
    deterministic = False

    model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim, num_networks=num_networks,
                            num_elites=num_elites, model_type=model_type, separate_mean_var=separate_mean_var,
                            name=model_name, deterministic=deterministic)

    if 'antmaze' in args.env_name:
        samples = processed_qlearning_dataset(args, env, args.env_name, timeout_frame=False, done_frame=True)
    else:
        samples = processed_qlearning_dataset(args, env, args.env_name, timeout_frame=False, done_frame=False)
    samples['rewards'] = np.expand_dims(samples['rewards'], axis=1)

    if args.train_forward_model and not args.train_reverse_model:
        print("---------------------------------")
        print("Training forward model.")
        print("---------------------------------")
        save_dir = os.path.join(args.save_dir, args.env_name + '_CARI_forward_{}'.format(args.seed))
        inputs, outputs = format_samples_for_training(samples)
    elif args.train_reverse_model and not args.train_forward_model:
        print("---------------------------------")
        print("Training reverse model.")
        print("---------------------------------")
        save_dir = os.path.join(args.save_dir, args.env_name + '_CARI_reverse_{}'.format(args.seed))
        inputs, outputs = format_reverse_samples_for_training(samples)
    else:
        raise NotImplementedError
    
    batch_size = 256
    max_epochs = 1  # for demonstration, else set to None
    holdout_ratio = 0.2
    max_t = 1  # for demonstration, else set to None

    model.train(inputs, outputs, batch_size=batch_size, max_epochs=max_epochs, holdout_ratio=holdout_ratio, max_t=max_t)

    mkdir(save_dir)
    model.save(save_dir, 0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--train_forward_model', action='store_true')
    parser.add_argument('--train_reverse_model', action='store_true')
    parser.add_argument('--save_dir', default='mopo_models')
    parser.add_argument('--data_path', default='d4rl/dataset/Walker2d/body_mass/body_random.hdf5')
    parser.add_argument('--data_proportion', default=10, type=int)
    parser.add_argument('--isMediumExpert', default=False, type=bool)

    args = parser.parse_args()
    main(args)
