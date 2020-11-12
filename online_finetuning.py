from ppo_spinningup.ppo import *
from ppo_spinningup.core import *
from ppo_spinningup.model import *

from ppo_spinningup.ppo import PPO
import gym
import gym_sokoban

import time

import wandb

import numpy as np

import torch

# if __name__ == '__main__':
import argparse
parser = argparse.ArgumentParser()

# parser.add_argument('--l', type=int, default=2)
parser.add_argument('--env', type=str, default='Sokoban-v0')
parser.add_argument('--seed', '-s', type=int, default=0)

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

parser.add_argument('--render_mode', type=str, default='rgb_array') #rgb_array or tiny_rgb_array
parser.add_argument('--num_boxes', type=int, default=4)
parser.add_argument('--fixed_env', type=int, default=1)
parser.add_argument('--randomized_init_position', type=int, default=1)

parser.add_argument('--epoch_blueprint', type=int, default=1500)

parser.add_argument('--index_level', type=int, default=1)

parser.add_argument('--nb_test_episodes', type=int, default=5)
parser.add_argument('--test_ep_length', type=int, default=120)
parser.add_argument('--steps_finetune', type=int, default=256)
parser.add_argument('--nb_gradient_finetune', type=int, default=64)
parser.add_argument('--min_horizon_online_search', type=int, default=30)
parser.add_argument('--max_horizon_online_search', type=int, default=120)

args = parser.parse_args()

fixed_env = args.fixed_env >0
randomized_init_position = args.randomized_init_position >0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

wandb.init(project="test-sokoban", name="ppo_finetune_{}_{}_{}".format(args.render_mode, args.env, args.seed).replace('-', '_'))
wandb.config.update(args)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

env_id = args.env
env_ = gym.make(env_id, num_boxes= args.num_boxes, index = args.index_level, max_steps=args.test_ep_length, seed=args.seed, fixed_env=fixed_env, randomized_init_position=randomized_init_position)

params_env = {"max_steps":args.test_ep_length, "index":args.index_level, "seed":args.seed, "num_boxes":args.num_boxes, "fixed_env":fixed_env, "randomized_init_position":randomized_init_position}

params_actor = {"render_mode":args.render_mode,"hidden_dim": args.hidden_dim, "nb_hidden": args.nb_hidden}

path_ac = "ppo_ac_{}epch_{}{}_{}_seed{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed)
path_piot = "ppo_piopt_{}epch_{}{}_{}_seed{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed)
path_vopt = "ppo_vopt_{}epch_{}{}_{}_seed{}.pth".format(args.epoch_blueprint, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed)

params_finetune_trainer = {"env":env_, "ac_kwargs":params_actor, "gamma":args.gamma,
                "seed":args.seed, "steps_per_epoch":args.steps_finetune,
                 "clip_ratio":args.clip_ratio, "pi_lr":args.pi_lr,
                "vf_lr":args.vf_lr, "train_pi_iters":args.nb_gradient_finetune, "train_v_iters":args.nb_gradient_finetune, "lam":args.lam,
                "target_kl":args.target_kl}

def vanilla():
    global_trainer = PPO(**params_finetune_trainer)
    global_trainer.load_state_dict(path_ac, path_piot, path_vopt)
    test_env = gym.make(env_id, **params_env)
    cumulative_rewards = []
    gif = []
    for episode in range(args.nb_test_episodes):
        ep_len = 0
        episode_reward = 0
        obs = test_env.reset()
        while True:
            ep_len+=1
            gif.append(test_env.render(mode='rgb_array').transpose(2, 0, 1))
            action, v, logp = global_trainer.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
            obs, reward, done, _ = test_env.step(action.item())
            episode_reward += reward
            if done or ep_len>args.test_ep_length:
                cumulative_rewards.append(episode_reward)
                break
    wandb.log({"Blueprint Policy": wandb.Video(np.stack(gif, axis=0), fps=4,
                                         format="gif"), 'Vanilla Test Cumulative Reward': np.mean(cumulative_rewards), 'Finetune Test Cumulative Reward': np.mean(cumulative_rewards), "Online Samples": 0, "Time": 0, "Number of online gradient descent per step":0, "Online batch size":0})

def finetune(nb_online_sample, cupt=5):
    # import pdb; pdb.set_trace()
    test_env = gym.make(env_id, **params_env)
    planning_env = gym.make(env_id, **params_env)
    global_trainer = PPO(**params_finetune_trainer)
    simulation_trainer = PPO(**params_finetune_trainer)
    global_trainer.load_state_dict(path_ac, path_piot, path_vopt)
    simulation_trainer.load_state_dict(path_ac, path_piot, path_vopt)

    gif = []

    def select_action(state, init_obs):
        global_probs = global_trainer.get_prob(torch.as_tensor(init_obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)).squeeze().cpu().numpy()

        nb_rollouts = 0
        nb_rollouts_action = np.zeros(planning_env.action_space.n)
        rollouts_qvalues = {key: [] for key in range(planning_env.action_space.n)}

        action_seq_seen_so_far = set()
        sample = 0

        nb_iteration = int(nb_online_sample/args.steps_finetune)
        delta_horizon = int((args.max_horizon_online_search-args.min_horizon_online_search)/nb_iteration)

        iteration=0

        # print(nb_iteration)
        # print(delta_horizon)

        while sample<nb_online_sample:
            horizon = args.min_horizon_online_search+iteration*delta_horizon
            iteration+=1
            # print(horizon)
            while simulation_trainer.buf.ptr < simulation_trainer.buf.max_size:
                for first_action in range(planning_env.action_space.n):
                    planning_env.init_state(state)
                    if simulation_trainer.buf.ptr == simulation_trainer.buf.max_size:
                        break
                    length_seq = 0
                    rollout_reward = 0
                    obs, reward, done, _ = planning_env.step(first_action)
                    sample+=1
                    nb_rollouts+=1
                    nb_rollouts_action[first_action]+=1
                    rollout_reward+=(args.gamma**length_seq)*reward
                    action_seq = [first_action]
                    length_seq+=1
                    while length_seq<horizon or tuple(action_seq) in action_seq_seen_so_far:
                        path_already_finished = False
                        try:
                            action_seq_seen_so_far.add(tuple(action_seq))
                        except:
                            import pdb; pdb.set_trace()
                        action, v, logp = simulation_trainer.ac.step(
                            torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
                        action_seq.append(action.item())
                        length_seq+=1
                        next_obs, reward, done, _ = planning_env.step(action.item())
                        sample+=1
                        rollout_reward+=(args.gamma**length_seq)*reward
                        if simulation_trainer.buf.ptr < simulation_trainer.buf.max_size:
                            simulation_trainer.buf.store(obs, action, reward, v, logp)
                        if done:
                            simulation_trainer.buf.finish_path(0)
                            path_already_finished = True
                            break
                        if simulation_trainer.buf.ptr == simulation_trainer.buf.max_size:
                            break
                        obs=next_obs
                    if done:
                        if not path_already_finished:
                            v = 0
                            simulation_trainer.buf.finish_path(v)
                    else:
                        _, v, _ = simulation_trainer.ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
                        simulation_trainer.buf.finish_path(v)
                        rollout_reward+=(args.gamma**length_seq)*v
                    # print(length_seq)
                    # print(action_seq_seen_so_far)
                    rollouts_qvalues[first_action].append(rollout_reward)
            simulation_trainer.update()
            simulation_trainer.buf.after_update()
        mean_rollouts_qvalues = {key: np.mean(rollouts_qvalues[key]) for key in range(planning_env.action_space.n)}
        def puct(k):
            return mean_rollouts_qvalues[k]+cupt*global_probs[k]*np.sqrt(nb_rollouts)/(1+nb_rollouts_action[k])
        # import pdb; pdb.set_trace()
        return max(np.arange(planning_env.action_space.n), key=puct), sample

    cumulative_rewards=[]
    samples = []
    times = []
    for episode in range(args.nb_test_episodes):
        ep_len = 0
        episode_reward = 0
        obs = test_env.reset()
        state = test_env.get_state()
        sample = 0
        t0 = time.time()
        while True:
            # print(ep_len)
            ep_len+=1
            gif.append(test_env.render(mode='rgb_array').transpose(2, 0, 1))

            action, sampl = select_action(state, obs)
            sample+=sampl
            obs, reward, done, _ = test_env.step(action.item())
            episode_reward+=reward
            if done or ep_len>args.test_ep_length:
                cumulative_rewards.append(episode_reward)
                samples.append(sample)
                times.append(time.time() - t0)
                break
            state = test_env.get_state()
    wandb.log({f"Finetuned Policy, {nb_online_sample} samples, {args.steps_finetune} samples per update, {args.nb_gradient_finetune} descent per update": wandb.Video(np.stack(gif, axis=0), fps=4,
                                         format="gif"), 'Finetune Test Cumulative Reward': np.mean(cumulative_rewards), "Online Samples": np.mean(samples), "Time": np.mean(times), "Number of online gradient descent per step":simulation_trainer.train_pi_iters, "Online batch size":simulation_trainer.local_steps_per_epoch})

nb_online_samples = [2048, 4096]

vanilla()

for nb_online_sample in nb_online_samples:
    finetune(nb_online_sample)
