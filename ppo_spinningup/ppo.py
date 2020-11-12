import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from .core import *
from .model import *
from .logx import *
# from .core import core
import wandb

# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.size = size
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def after_update(self):
        self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get

        # the next two lines implement the advantage normalization trick
        # import pdb; pdb.set_trace()
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std+1e-5)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in data.items()}

class PPO:
    def __init__(self, env, actor_critic=CNNActorCritic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.01, logger_kwargs=dict(), save_freq=100, test_freq=100, max_grad_norm=-1, clip_grad=-1):
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.clip_grad = clip_grad

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

        # Random seed
        # seed += 10000 * proc_id()
        seed += 10000 * 0
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = env
        # import pdb; pdb.set_trace()
        if ac_kwargs["render_mode"]=="rgb_array":
            obs_dim = self.env.observation_space.shape
        else:
            obs_dim = (10, 10, 3)
        act_dim = self.env.action_space.shape
        # import pdb; pdb.set_trace()

        # Create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, ac_kwargs).to(device)

        # wandb.watch(self.ac.pi.logits_net)
        # wandb.watch(self.ac.v.v_net)

        # Sync params across processes
        # sync_params(ac)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Set up experience buffer
        # local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.local_steps_per_epoch = int(steps_per_epoch / 1)
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)
        self.obs_dim=obs_dim
        self.act_dim=act_dim

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        # import pdb; pdb.set_trace()
        pi, logp = self.ac.pi(obs.permute(0, 3, 1, 2).contiguous(), act)
        # import pdb; pdb.set_trace()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs.permute(0, 3, 1, 2).contiguous()) - ret) ** 2).mean()

    def load_state(self, states):
        self.ac.pi.load_state_dict(states["pi"])
        self.ac.v.load_state_dict(states["v"])
        self.pi_optimizer.load_state_dict(states["pi_optimizer"])
        self.vf_optimizer.load_state_dict(states["vf_optimizer"])

    def get_state(self):
        return {"pi": self.ac.pi.state_dict(),
        "v": self.ac.v.state_dict(),
        "pi_optimizer": self.pi_optimizer.state_dict(),
        "vf_optimizer": self.vf_optimizer.state_dict()}

    def load_state_dict(self, path_ac, path_piot, path_vopt):
        self.ac.load_state_dict(torch.load(path_ac))
        self.pi_optimizer.load_state_dict(torch.load(path_piot))
        self.vf_optimizer.load_state_dict(torch.load(path_vopt))

    def get_prob(self, obs):
        return self.ac.get_prob(obs)


    def test(self, nb_test_episodes, length_test_episode):
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        gif = []
        for _ in range(nb_test_episodes):
            for t in range(length_test_episode):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))

                next_o, r, d, _ = self.env.step(a)

                o = next_o

                ep_len += 1
                timeout = ep_len == self.max_ep_len
                terminal = d or timeout

                if terminal:
                    break
        # import pdb; pdb.set_trace()
        return gif

    def update(self, train=True):
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            if train:
                self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
            if train:
                loss_pi.backward()
                # mpi_avg_grads(ac.pi)    # average grads across MPI processes
                if self.max_grad_norm >= 0:
                    nn.utils.clip_grad_norm_(self.ac.pi.parameters(),
                                             self.max_grad_norm)
                if self.clip_grad >= 0:
                    nn.utils.clip_grad_value_(self.ac.v.parameters(),
                                              self.clip_grad)
                self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            if train:
                self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            if train:
                loss_v.backward()
                if self.max_grad_norm >= 0:
                    nn.utils.clip_grad_norm_(self.ac.v.parameters(),
                                             self.max_grad_norm)
                if self.clip_grad >= 0:
                    nn.utils.clip_grad_value_(self.ac.v.parameters(),
                                              self.clip_grad)
                # mpi_avg_grads(ac.v)    # average grads across MPI processes
                self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                          KL=kl, Entropy=ent, ClipFrac=cf,
                          DeltaLossPi=(loss_pi.item() - pi_l_old),
                          DeltaLossV=(loss_v.item() - v_l_old))

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=64)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--cpu', type=int, default=4)
#     parser.add_argument('--steps', type=int, default=4000)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='ppo')
#     args = parser.parse_args()
#
#     # mpi_fork(args.cpu)  # run parallel code with mpi
#
#     # from spinup.utils.run_utils import setup_logger_kwargs
#     # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
#
#     ppo(lambda : gym.make(args.env), actor_critic=CNNActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
#         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)