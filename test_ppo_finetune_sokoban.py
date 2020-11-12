from ppo_spinningup.ppo import *
from ppo_spinningup.core import *
from ppo_spinningup.model import *

from ppo_spinningup.ppo import PPO
import gym
import gym_sokoban

import time

import wandb

from tester_ppo import *

import numpy as np

import torch

from utils.wrappers import make_atari, wrap_deepmind, wrap_pytorch


# if __name__ == '__main__':
import argparse
parser = argparse.ArgumentParser()

# parser.add_argument('--l', type=int, default=2)
parser.add_argument('--env', type=str, default='Sokoban-v0')
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--steps', type=int, default=2048)
parser.add_argument('--epochs', type=int, default=10001)

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_dim', type=int, default=-1)
parser.add_argument('--nb_hidden', type=int, default=-1)
parser.add_argument('--lam', type=float, default=0.97)
parser.add_argument('--clip_ratio', type=float, default=0.2)
parser.add_argument('--pi_lr', type=float, default=3e-4)
parser.add_argument('--vf_lr', type=float, default=1e-3)
parser.add_argument('--target_kl', type=float, default=0.01)
parser.add_argument('--train_pi_iters', type=int, default=80)
parser.add_argument('--train_v_iters', type=int, default=80)
parser.add_argument('--normalize_reward', type=int,default=-1)
parser.add_argument('--clip_reward', type=float, default=-1)
parser.add_argument('--clip_grad', type=float, default=-1)
parser.add_argument('--recurrent_seq_length_policy', type=int, default=-1) #-1=no rec

parser.add_argument('--save_blueprint_frequency', type=int, default=500)

parser.add_argument('--max_ep_len', type=int, default=120)

parser.add_argument('--nb_test_episodes', type=int, default=3)

parser.add_argument('--rollouts_online_search', type=int, default=64)
parser.add_argument('--test_frequency', type=int, default=200)
parser.add_argument('--only_finetune', type=int, default=-1)

parser.add_argument('--horizon_online_search', type=int, default=120)
parser.add_argument('--nb_gradient_finetune', type=int, default=80)
parser.add_argument('--steps_finetune', type=int, default=128)

parser.add_argument('--render_mode', type=str, default='rgb_array') #rgb_array or tiny_rgb_array
parser.add_argument('--num_boxes', type=int, default=4)
parser.add_argument('--fixed_env', type=int, default=1)
parser.add_argument('--randomized_init_position', type=int, default=1)

parser.add_argument('--epoch_blueprint', type=int, default=200)
parser.add_argument('--test_horizon', type=int, default=-1)
parser.add_argument('--test_nb_gradient_finetune', type=int, default=-1)
parser.add_argument('--test_steps_finetune', type=int, default=-1)

parser.add_argument('--index_level', type=int, default=1)

args = parser.parse_args()
test_horizon = args.test_horizon>0
test_nb_gradient_finetune = args.test_nb_gradient_finetune>0
test_steps_finetune = args.test_steps_finetune>0
str_run = "horizon" if test_horizon else "gradient" if test_nb_gradient_finetune else "steps"
str_plot = "Online planning horizon" if test_horizon else "Number of online gradient descent per step" if test_nb_gradient_finetune else "Online batch size"

fixed_env = args.fixed_env >0
randomized_init_position = args.randomized_init_position >0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(project="test-sokoban", name="ppo_finetune_{}_{}_{}_{}".format(str_run, args.render_mode, args.env, args.seed).replace('-', '_'))
wandb.config.update(args)

only_finetune = args.only_finetune > 0

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

env_id = args.env
# import pdb; pdb.set_trace()
env = gym.make(env_id, num_boxes= args.num_boxes, max_steps=args.max_ep_len, seed=args.seed, fixed_env=fixed_env, randomized_init_position=randomized_init_position)

params_env = {"max_steps":args.max_ep_len, "seed":args.seed, "num_boxes":args.num_boxes, "fixed_env":fixed_env, "randomized_init_position":randomized_init_position}

params_actor = {"render_mode":args.render_mode,"hidden_dim": args.hidden_dim, "nb_hidden": args.nb_hidden}

def vanilla(trainer, env_id, params_env,nb_episode, max_ep_len, path_ac, path_piot, path_vopt):
    trainer.load_state_dict(path_ac, path_piot, path_vopt)
    test_env = gym.make(env_id, **params_env)
    cumulative_rewards = []
    for episode in range(nb_episode):
        # print(episode)
        ep_len = 0
        episode_reward = 0
        obs = test_env.reset()
        while True:
            ep_len+=1
            action, v, logp = trainer.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
            obs, reward, done, _ = test_env.step(action.item())
            episode_reward += reward
            if done or ep_len>max_ep_len:
                cumulative_rewards.append(episode_reward)
                break
    wandb.log({'Vanilla Test Cumulative Reward': np.mean(cumulative_rewards), 'Finetune Test Cumulative Reward': np.mean(cumulative_rewards), str_plot:0, "Online Samples": 0, "Time": 0, "Online planning horizon":0, "Number of online gradient descent per step":0, "Online batch size":0})

def finetune(trainer, horizon, env_id, params_env, nb_episode, max_ep_len, path_ac, path_piot, path_vopt):
    horizon = horizon
    # blueprint_trainer=trainer
    # trainer=trainer_class(**params_trainer)
    test_env = gym.make(env_id, **params_env)
    planning_env = gym.make(env_id, **params_env)

    def select_action(state, init_obs):
        trainer.load_state_dict(path_ac, path_piot, path_vopt)
        planning_env.reset()
        sample = 0
        while trainer.buf.ptr<trainer.buf.max_size:
            planning_env.init_state(state)
            obs = init_obs
            for t in range(horizon):
                path_already_finished = False
                action, v, logp = trainer.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
                next_obs, reward, done, _ = planning_env.step(action.item())
                sample+=1
                if trainer.buf.ptr < trainer.buf.max_size:
                    trainer.buf.store(obs, action, reward, v, logp)
                else:
                    break
                if done:
                    trainer.buf.finish_path(0)
                    path_already_finished = True
                    break
                obs = next_obs
        if done:
            if not path_already_finished:
                v=0
                trainer.buf.finish_path(v)
        else:
            _, v, _ = trainer.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
            trainer.buf.finish_path(v)
        trainer.update()
        trainer.buf.after_update()
        a, v, logp = trainer.ac.step(torch.as_tensor(init_obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
        return a, sample

    cumulative_rewards=[]
    samples = []
    times = []
    for episode in range(nb_episode):
        ep_len = 0
        episode_reward = 0
        obs = test_env.reset()
        state = test_env.get_state()
        sample = 0
        t0 = time.time()
        while True:
            ep_len+=1
            action, sampl = select_action(state, obs)
            sample+=sampl
            obs, reward, done, _ = test_env.step(action.item())
            episode_reward+=reward
            if done or ep_len>max_ep_len:
                cumulative_rewards.append(episode_reward)
                samples.append(sample)
                times.append(time.time() - t0)
                break
            state = test_env.get_state()
    wandb.log({'Finetune Test Cumulative Reward': np.mean(cumulative_rewards), "Online Samples": np.mean(samples), "Time": np.mean(times), "Online planning horizon":horizon, "Number of online gradient descent per step":trainer.train_pi_iters, "Online batch size":trainer.local_steps_per_epoch})

nb_episode = 5
max_ep_len = 120

if test_horizon:
    horizons = [8, 64, 128]
    steps_finetune = 512
    nb_gradient = 32
    vanilla_done=False
    for horizon in horizons:
        params_finetune_trainer = {"env":env, "ac_kwargs":params_actor, "gamma":args.gamma,
                "seed":args.seed, "steps_per_epoch":steps_finetune, "epochs":args.epochs,
                 "clip_ratio":args.clip_ratio, "pi_lr":args.pi_lr,
                "vf_lr":args.vf_lr, "train_pi_iters":nb_gradient, "train_v_iters":nb_gradient, "lam":args.lam,
                "target_kl":args.target_kl}
        trainer=PPO(**params_finetune_trainer)
        path_ac = "ppo_ac_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), "fix" if fixed_env else "nofix")
        path_piot = "ppo_piopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), "fix" if fixed_env else "nofix")
        path_vopt = "ppo_vopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), "fix" if fixed_env else "nofix")
        if not vanilla_done:
            vanilla(trainer, env_id, params_env, nb_episode, max_ep_len, path_ac, path_piot, path_vopt)
            vanilla_done=True
        finetune(trainer, horizon, env_id, params_env, nb_episode, max_ep_len, path_ac, path_piot, path_vopt)

elif test_nb_gradient_finetune:
    horizon = 128
    steps_finetune = 512
    nb_gradients = [80,1,32]
    vanilla_done=False

    for gradient in nb_gradients:
        params_finetune_trainer = {"env": env, "ac_kwargs": params_actor, "gamma": args.gamma,
                                   "seed": args.seed, "steps_per_epoch": steps_finetune, "epochs": args.epochs,
                                   "clip_ratio": args.clip_ratio, "pi_lr": args.pi_lr,
                                   "vf_lr": args.vf_lr, "train_pi_iters": gradient, "train_v_iters": gradient,
                                   "lam": args.lam,
                                   "target_kl": args.target_kl}
        trainer = PPO(**params_finetune_trainer)
        path_ac = "ppo_ac_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                   "fix" if fixed_env else "nofix")
        path_piot = "ppo_piopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                        "fix" if fixed_env else "nofix")
        path_vopt = "ppo_vopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                       "fix" if fixed_env else "nofix")
        if not vanilla_done:
            vanilla(trainer, env_id, params_env, nb_episode, max_ep_len)
            vanilla_done=True
        finetune(trainer, horizon, env_id, params_env, nb_episode, max_ep_len)

elif test_steps_finetune:
    horizon = 128
    steps_finetunes = [2048, 64, 512]
    nb_gradient = 32
    vanilla_done=False

    for step_finetun in steps_finetunes:
        params_finetune_trainer = {"env": env, "ac_kwargs": params_actor, "gamma": args.gamma,
                                   "seed": args.seed, "steps_per_epoch": step_finetun, "epochs": args.epochs,
                                   "clip_ratio": args.clip_ratio, "pi_lr": args.pi_lr,
                                   "vf_lr": args.vf_lr, "train_pi_iters": nb_gradient, "train_v_iters": nb_gradient,
                                   "lam": args.lam,
                                   "target_kl": args.target_kl}
        trainer = PPO(**params_finetune_trainer)
        path_ac = "ppo_ac_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                   "fix" if fixed_env else "nofix")
        path_piot = "ppo_piopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                        "fix" if fixed_env else "nofix")
        path_vopt = "ppo_vopt_{}epch_{}_{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''),
                                                       "fix" if fixed_env else "nofix")
        if not vanilla_done:
            vanilla(trainer, env_id, params_env, nb_episode, max_ep_len)
            vanilla_done=True
        finetune(trainer, horizon, env_id, params_env, nb_episode, max_ep_len)

else:
    assert False