import pickle
import gym
import mujoco_py
import d4rl
import h5py

import numpy as np

def get_data(dataset, data_path, isMediumExpert, data_proportion=10):
    #num = int(1e5)
    num = int(len(dataset["observations"])/data_proportion)
    # print(num)
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
            if 'timeouts' in data:
                dataset['timeouts'] = np.concatenate((dataset['timeouts'], data['timeouts']), axis=0)
            else:
                del dataset['timeouts']
    return dataset

def processed_qlearning_dataset(env, data_path, isMediumExpert, timeout_frame=False, done_frame=False):
    dataset = env.get_dataset()
    dataset = get_data(dataset, data_path, isMediumExpert)

    N = dataset['rewards'].shape[0]

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not timeout_frame) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if (not done_frame) and done_bool:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'obs': np.array(obs_),
        'act': np.array(action_),
        'obs_next': np.array(next_obs_),
        'rew': np.array(reward_),
        'done': np.array(done_),
    }

def load_demos(filename, env_id):
    with open(filename, "rb") as f:
        demo = pickle.load(f)

    # Post-processing for Fetch envs
    if any(env_id.startswith(prefix) for prefix in ["Push", "Pick"]):
        demo["obs_achieved_goal"] = []
        demo["obs_desired_goal"] = []
        demo["obs_next_achieved_goal"] = []
        demo["obs_next_desired_goal"] = []
        for e in range(len(demo["obs"])):
            demo["obs_achieved_goal"].append([])
            demo["obs_desired_goal"].append([])
            demo["obs_next_achieved_goal"].append([])
            demo["obs_next_desired_goal"].append([])
            for t in range(len(demo["obs"][e])):
                # Observation is an OrderedDict
                # Split keys and flatten trajectories
                obs = demo["obs"][e][t].copy()
                demo["obs"][e][t] = obs["observation"]
                demo["obs_achieved_goal"][-1].append(obs["achieved_goal"])
                demo["obs_desired_goal"][-1].append(obs["desired_goal"])

                obs_next = demo["obs_next"][e][t].copy()
                demo["obs_next"][e][t] = obs_next["observation"]
                demo["obs_next_achieved_goal"][-1].append(obs_next["achieved_goal"])
                demo["obs_next_desired_goal"][-1].append(obs_next["desired_goal"])

            for key in [
                "obs",
                "obs_achieved_goal",
                "obs_desired_goal",
                "obs_next",
                "obs_next_achieved_goal",
                "obs_next_desired_goal",
            ]:
                demo[key][e] = np.vstack(demo[key][e])

    return demo
