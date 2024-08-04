from pathlib import Path

import hydra
import mujoco_maze
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Batch
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from src.demo_utils import processed_qlearning_dataset #load_demos
from src.envs.utils import make_env
from src.evaluate import d4rl_score #evaluate_adroit, evaluate_fetch, evaluate_maze
from src.logger import get_logger
from src.plot.maze import render_demonstrations as render_maze_demos
from src.trainer.bc import BCTrainer
from src.trainer.utils import evaluate_policy
from src.utils import seed_all, trunc_normal_init

import gym
import mujoco_py
import d4rl
import h5py


@hydra.main(config_path="../conf", config_name="config")
# @hydra.main(config_path="../conf/experiment/bmil", config_name="maze_point5x11")
def main(cfg: DictConfig):
    log = get_logger(__name__)

    # Seed
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # Init WandBLogger
    tags = None if cfg.logger.wandb.get("id") else cfg.logger.wandb.tags
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    log.info(f"Instantiating logger <{cfg.logger.wandb._target_}>")
    logger = instantiate(cfg.logger.wandb, wandb_config, tags=tags)
    
    # env = test_envs.workers[0].env
    env_name = cfg.demonstration.env_name #'walker2d-random-v0' #'walker2d-medium-v0' #'walker2d-random-v0'  #'walker2d-medium-expert-v0'
    env = gym.make(env_name)
    test_envs = gym.make(env_name)

    obs_dim = int(np.prod(env.observation_space.shape))
    if env.action_space.shape:
        act_dim = int(np.prod(env.action_space.shape))
    elif env.action_space.n:
        act_dim = env.action_space.n
    # Add info about obs/act space into cfg and wandb.config
    obs_info = {"dim": obs_dim}
    act_info = {"dim": act_dim}
    logger.run.config.env["obs"] = obs_info
    logger.run.config.env["act"] = act_info

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(logger.run.config)

    # Demonstration
    demonstration = processed_qlearning_dataset(env, cfg.demonstration.path, cfg.demonstration.isMediumExpert)
    demonstration['rew'] = np.expand_dims(demonstration['rew'], axis=1)

    # Replay Buffer
    log.info(f"Buffer: instantiating buffers")
    demos = []
    demo_buffer = instantiate(cfg.demonstration.buffer)
    for e in range(len(demonstration["obs"])):
        # for t in range(len(demonstration["obs"][e])):

        demo_batch = Batch(
            obs=demonstration["obs"][e].astype(np.float32), #[e][t]
            act=demonstration["act"][e].astype(np.float32), #[e][t]
            rew=demonstration["rew"][e].astype(np.float32).squeeze(), #[e][t]
            done=demonstration["done"][e].astype(bool).squeeze(), #[e][t]
            obs_next=demonstration["obs_next"][e].astype(np.float32), #[e][t]
        )

        demos.append(demo_batch)

        # Add demonstration to buffer
        for _ in range(cfg.demonstration.repeat):
            demo_buffer.add(demo_batch)
            
    demos = Batch.stack(demos)
    act_scale = demos.act.std(axis=0)


    # Policy
    net = instantiate(
        cfg.policy.net,
        state_shape=obs_dim,
        activation=torch.nn.ReLU,
    )
    actor = instantiate(cfg.policy.actor, preprocess_net=net, action_shape=act_dim).to(cfg.device)
    optim = instantiate(cfg.policy.optimizer, params=actor.parameters())

    log.info(f"Policy: instantiating policy <{cfg.policy.bc._target_}>")
    policy = instantiate(
        cfg.policy.bc, model=actor, optim=optim, action_space=env.action_space
    )
    logger.watch_model(policy)

    # Dynamics model
    log.info(f"Instantiating dynamics model <{cfg.dynamics.model._target_}>")
    if cfg.dynamics.mode == "backward":
        # In: next_obs
        # Out: act
        act_net = instantiate(
            cfg.dynamics.act_net,
            in_dim=obs_dim,
            out_dim=act_dim,
            activation=torch.nn.ReLU,
        ).to(cfg.device)
        act_net.apply(trunc_normal_init)

        # In: next_obs, act
        # Out: obs
        obs_net = instantiate(
            cfg.dynamics.obs_net,
            in_dim=obs_dim + act_dim,
            out_dim=obs_dim,
            activation=torch.nn.ReLU,
        ).to(cfg.device)
        obs_net.apply(trunc_normal_init)

        # Optimizer
        dynamics_optim = instantiate(
            cfg.dynamics.optimizer,
            params=list(act_net.parameters()) + list(obs_net.parameters()),
        )
        dynamics_lr_scheduler = None
        if cfg.dynamics.lr_decay:
            # decay learning rate to 0 linearly
            dynamics_lr_scheduler = LambdaLR(
                dynamics_optim,
                lr_lambda=lambda epoch: 1 - epoch / cfg.policy.train.n_epoch,
            )

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        dynamics_model = instantiate(
            cfg.dynamics.model,
            act_net=act_net,
            obs_net=obs_net,
            optim=dynamics_optim,
            dist_fn=dist,
            action_space=env.action_space,
            observation_space=env.observation_space,
            lr_scheduler=dynamics_lr_scheduler,
            act_scale=act_scale,
        ).to(cfg.device)

    elif cfg.dynamics.mode == "forward":
        # In: obs, act
        # Out: obs_next
        obs_net = instantiate(
            cfg.dynamics.obs_net,
            in_dim=obs_dim + act_dim,
            out_dim=obs_dim,
            activation=torch.nn.ReLU,
        ).to(cfg.device)
        obs_net.apply(trunc_normal_init)

        # Optimizer
        dynamics_optim = instantiate(
            cfg.dynamics.optimizer,
            params=obs_net.parameters(),
        )
        dynamics_lr_scheduler = None
        if cfg.dynamics_decay:
            # decay learning rate to 0 linearly
            dynamics_lr_scheduler = LambdaLR(
                dynamics_optim,
                lr_lambda=lambda epoch: 1 - epoch / cfg.policy.train.n_epoch,
            )

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        dynamics_model = instantiate(
            cfg.dynamics.model,
            obs_net=obs_net,
            optim=dynamics_optim,
            dist_fn=dist,
            observation_space=env.observation_space,
            lr_scheduler=dynamics_lr_scheduler,
        ).to(cfg.device)

    logger.watch_model(dynamics_model)

    # Checkpoint fn
    def checkpoint_fn(save_path, **kwargs):
        # Policy
        policy_filename = "policy.pt"
        torch.save(
            {"policy": policy.state_dict(), "optim": optim.state_dict()},
            Path(save_path) / policy_filename,
        )

        # Dynamics Model
        dynamics_model_filename = "dynamics_model.pt"
        if cfg.dynamics.mode == "backward":
            torch.save(
                {
                    "model": dynamics_model.state_dict(),
                    "optim": dynamics_optim.state_dict(),
                },
                Path(save_path) / dynamics_model_filename,
            )

    timestamp = BCTrainer(
        policy=policy,
        demo_buffer=demo_buffer,
        cfg=cfg,
        dynamics_model=dynamics_model,
        demos=demos,
        logger=logger,
        checkpoint_fn=checkpoint_fn,
    )

    # Evaluate on test_envs
    log.info("[Test]")
    policy_test_collector = instantiate(
        cfg.test.collector, policy=policy, env=test_envs
    )
    test_result = evaluate_policy(
        policy=policy,
        collector=policy_test_collector,
        n_episode=cfg.test.n_ep,
        collect_kwargs={"disable_tqdm": False},
    )
    print(test_result.keys())
    logger.write({"test/success_ratio": test_result["success_ratio"]}, timestamp)
    log.info(f"[Test] success_ratio: {test_result['success_ratio']:.3f}")
    log.info(
        f"[Test] Reward: {test_result['rew']:.2f} +/- {test_result['rew_std']:.2f}"
    )
    log.info(
        f"[Test] Length: {test_result['len']:.2f} +/- {test_result['len_std']:.2f}"
    )
    test_envs.close()

    # Evaluate on eval_envs
    # if any(cfg.env.id.startswith(prefix) for prefix in ["Push", "Pick"]):
    #     evaluate_fetch(policy, cfg, env, logger, timestamp)
    # elif any(cfg.env.id.startswith(prefix) for prefix in ["Point", "Ant"]):
    #     evaluate_maze(policy, cfg, env, logger, timestamp)
    # elif cfg.env.id.startswith("Adroit"):
    #     evaluate_adroit(policy, cfg, logger, timestamp)

    # Get d4rl results
    score = d4rl_score(env_name, test_result['rew'], logger, timestamp)
    print("####### D4RL SCORE: ", score)

    logger.close()


if __name__ == "__main__":
    main()
    wandb.finish()
