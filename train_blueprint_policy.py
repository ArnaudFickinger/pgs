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

parser.add_argument('--save_blueprint_frequency', type=int, default=100)

parser.add_argument('--max_ep_len', type=int, default=120)

parser.add_argument('--nb_test_episodes', type=int, default=3)
parser.add_argument('--horizon_online_search', type=int, default=120)
parser.add_argument('--rollouts_online_search', type=int, default=64)
parser.add_argument('--test_frequency', type=int, default=200)
parser.add_argument('--only_finetune', type=int, default=-1)

parser.add_argument('--gif_frequency', type=int, default=500)
parser.add_argument('--gif_episodes', type=int, default=1)
parser.add_argument('--gif_length', type=int, default=120)

parser.add_argument('--nb_gradient_finetune', type=int, default=80)
parser.add_argument('--steps_finetune', type=int, default=128)

parser.add_argument('--render_mode', type=str, default='rgb_array') #rgb_array or tiny_rgb_array
parser.add_argument('--num_boxes', type=int, default=4)
parser.add_argument('--fixed_env', type=int, default=1)
parser.add_argument('--randomized_init_position', type=int, default=1)

parser.add_argument('--index_level', type=int, default=1)

args = parser.parse_args()
fixed_env = args.fixed_env >0
randomized_init_position = args.randomized_init_position >0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.init(project="test-sokoban", name="ppo_{}_{}{}_{}".format(args.render_mode, args.env, args.index_level, args.seed).replace('-', '_'))
wandb.config.update(args)

only_finetune = args.only_finetune > 0

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

env_id = args.env
# import pdb; pdb.set_trace()
env = gym.make(env_id, num_boxes= args.num_boxes, index=args.index_level,max_steps=args.max_ep_len, seed=args.seed, fixed_env=fixed_env, randomized_init_position=randomized_init_position)

params_env = {"max_steps":args.max_ep_len, "seed":args.seed,"index":args.index_level, "num_boxes":args.num_boxes, "fixed_env":fixed_env, "randomized_init_position":randomized_init_position}

params_actor = {"render_mode":args.render_mode,"hidden_dim": args.hidden_dim, "nb_hidden": args.nb_hidden}

ppo_trainer = PPO(env=env, ac_kwargs=params_actor, gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         clip_ratio=args.clip_ratio, pi_lr=args.pi_lr,
        vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters, lam=args.lam,
        target_kl=args.target_kl)

params_finetune_trainer = {"env":env, "ac_kwargs":params_actor, "gamma":args.gamma,
        "seed":args.seed, "steps_per_epoch":args.steps_finetune, "epochs":args.epochs,
         "clip_ratio":args.clip_ratio, "pi_lr":args.pi_lr,
        "vf_lr":args.vf_lr, "train_pi_iters":args.nb_gradient_finetune, "train_v_iters":args.nb_gradient_finetune, "lam":args.lam,
        "target_kl":args.target_kl}

t0 = time.time()

def test(nb_test_episodes, length_test_episode):
    o  = env.reset()
    gif = []
    for _ in range(nb_test_episodes):
        for t in range(length_test_episode):
            gif.append(env.render(mode='rgb_array').transpose(2, 0, 1))
            a, _, _ = ppo_trainer.ac.step(
                torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))

            next_o, r, d, info = env.step(a.item(), observation_mode=args.render_mode)

            o = next_o

            timeout = ep_len == args.max_ep_len
            terminal = d or timeout
            if terminal:
                break
    return gif

o, ep_ext_ret, ep_len = env.reset(render_mode=args.render_mode),  0, 0
for epoch in range(args.epochs):
    print(epoch)
    for t in range(args.steps):
        a, v, logp = ppo_trainer.ac.step(torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))

        next_o, r, d, _ = env.step(a.item(), observation_mode=args.render_mode)
        ep_ext_ret += r
        ep_len += 1

        # save and log

        ppo_trainer.buf.store(o, a, r, v, logp)

        ppo_trainer.logger.store(VVals=v)

        # Update obs (critical!)
        o = next_o

        timeout = ep_len == args.max_ep_len
        terminal = d or timeout
        epoch_ended = t == args.steps - 1

        if terminal or epoch_ended:
            if epoch_ended and not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
            if timeout or epoch_ended:
                _, v, _ = ppo_trainer.ac.step(torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
            else:
                v = 0
            ppo_trainer.buf.finish_path(v)
            if terminal:
                ppo_trainer.logger.store(EpExternalRet=ep_ext_ret, EpLen=ep_len)
            o, ep_ext_ret, ep_len = env.reset(render_mode=args.render_mode), 0, 0

    # if args.test_frequency>0 and epoch % args.test_frequency == 0:
    #     for tester in testers:
    #         tester.test(args.nb_test_episodes, args.max_ep_len)

    ppo_trainer.update()
    keys = ['EpExternalRet', 'VVals', 'LossPi', 'LossV', 'Entropy', 'KL', 'StopIter']
    vals={}
    for key in keys:
        v = ppo_trainer.logger.epoch_dict[key]
        val = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        vals[key] = np.mean(val)
    vals['Epoch']=epoch
    vals['Time'] = time.time()-t0
    wandb.log(vals)
    ppo_trainer.buf.after_update()

    ppo_trainer.logger.log_tabular('Epoch', epoch)
    ppo_trainer.logger.log_tabular('EpExternalRet', with_min_and_max=True)
    ppo_trainer.logger.log_tabular('EpLen', average_only=True)
    ppo_trainer.logger.log_tabular('VVals', with_min_and_max=True)
    ppo_trainer.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * args.steps)
    ppo_trainer.logger.log_tabular('LossPi', average_only=True)
    ppo_trainer.logger.log_tabular('LossV', average_only=True)
    ppo_trainer.logger.log_tabular('DeltaLossPi', average_only=True)
    ppo_trainer.logger.log_tabular('DeltaLossV', average_only=True)
    ppo_trainer.logger.log_tabular('Entropy', average_only=True)
    ppo_trainer.logger.log_tabular('KL', average_only=True)
    ppo_trainer.logger.log_tabular('ClipFrac', average_only=True)
    ppo_trainer.logger.log_tabular('StopIter', average_only=True)
    ppo_trainer.logger.dump_tabular()

    if args.gif_frequency>0 and epoch % args.gif_frequency == 0:
        ppo_trainer.ac.eval()
        gif_ppo = test(args.gif_episodes, args.gif_length)
        ppo_trainer.ac.train()
        o, ep_ext_ret, ep_len = env.reset(render_mode=args.render_mode), 0, 0
        wandb.log({"Policy Epoch {}".format(epoch): wandb.Video(np.stack(gif_ppo, axis=0), fps=4,
                                                                              format="gif"), 'Epoch':epoch})

    if args.save_blueprint_frequency > 0 and epoch % args.save_blueprint_frequency == 0:
        torch.save(ppo_trainer.ac.state_dict(), "ppo_ac_{}epch_{}{}_{}_seed{}.pth".format(epoch, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed))
        torch.save(ppo_trainer.pi_optimizer.state_dict(), "ppo_piopt_{}epch_{}{}_{}_seed{}.pth".format(epoch, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed))
        torch.save(ppo_trainer.vf_optimizer.state_dict(), "ppo_vopt_{}epch_{}{}_{}_seed{}.pth".format(epoch, args.env.replace('-', ''), args.index_level,"fix" if fixed_env else "nofix", args.seed))

